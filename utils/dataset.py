"""
LISA-Gemma3 デュアルストリーム・データパイプライン
仕様書第5章に従った実装
"""

import glob
import os
import random
from typing import Dict, List, Tuple, Optional, Any
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torch.utils.data
from pycocotools import mask
from transformers import AutoProcessor

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .vqa_dataset import VQADataset

# デフォルト設定
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SEG_TOKEN = "<SEG>"
IGNORE_INDEX = -100

def collate_fn_gemma3(
    batch: List[Tuple], 
    gemma_processor: AutoProcessor, 
    local_rank: int = -1
) -> Dict[str, Any]:
    """
    Gemma-3のマルチモーダル処理に最適化されたcollate関数
    """
    # バッチデータの分離
    image_paths = []
    images = []  # PIL Images
    text_prompts = []
    masks_list = []
    labels_list = []
    
    for item in batch:
        image_path, image, text_prompt, mask, label = item
        image_paths.append(image_path)
        images.append(image)
        text_prompts.append(text_prompt)
        masks_list.append(mask)
        labels_list.append(label)
    
    # Gemma-3のチャットテンプレートでバッチ処理
    batch_messages = []
    for i in range(len(images)):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[i]},
                    {"type": "text", "text": text_prompts[i]}
                ]
            }
        ]
        batch_messages.append(messages)
    
    # バッチ全体のチャットテンプレート処理
    processed_inputs = []
    for messages in batch_messages:
        inputs = gemma_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        processed_inputs.append(inputs)
    
    # バッチの統一処理
    max_length = max(inputs["input_ids"].shape[1] for inputs in processed_inputs)
    
    # パディング処理
    batch_input_ids = []
    batch_attention_masks = []
    batch_pixel_values = []
    
    for inputs in processed_inputs:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        
        # パディング
        current_length = input_ids.shape[1]
        if current_length < max_length:
            pad_length = max_length - current_length
            input_ids = F.pad(input_ids, (0, pad_length), value=gemma_processor.tokenizer.pad_token_id)
            attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
        
        batch_input_ids.append(input_ids)
        batch_attention_masks.append(attention_mask)
        batch_pixel_values.append(pixel_values)
    
    return {
        "image_paths": image_paths,
        "images": images,  # PIL Images for SAM processing
        "input_ids": torch.cat(batch_input_ids, dim=0),
        "attention_mask": torch.cat(batch_attention_masks, dim=0),
        "pixel_values": torch.cat(batch_pixel_values, dim=0),
        "masks": masks_list,
        "labels": labels_list,
    }

class LisaGemma3Dataset(torch.utils.data.Dataset):
    """
    LISA-Gemma3用のメインデータセットクラス
    デュアルストリーム処理（Gemma用とSAM用の画像を同時に処理）
    """
    
    def __init__(
        self,
        base_image_dir: str,
        gemma_processor: AutoProcessor,
        samples_per_epoch: int = 500 * 8 * 2 * 10,
        precision: str = "bf16",
        gemma_image_size: int = 896,  # Gemma-3の推奨サイズ
        sam_image_size: int = 1024,   # SAMの推奨サイズ
        num_classes_per_sample: int = 3,
        exclude_val: bool = False,
        dataset: str = "sem_seg||refer_seg||vqa||reason_seg",
        sample_rate: List[float] = [9, 3, 3, 1],
        sem_seg_data: str = "ade20k||cocostuff||partimagenet||pascal_part",
        refer_seg_data: str = "refclef||refcoco||refcoco+||refcocog",
        vqa_data: str = "llava_instruct_150k",
        reason_seg_data: str = "ReasonSeg|train",
        explanatory: float = 0.1,
    ):
        self.base_image_dir = base_image_dir
        self.gemma_processor = gemma_processor
        self.samples_per_epoch = samples_per_epoch
        self.precision = precision
        self.gemma_image_size = gemma_image_size
        self.sam_image_size = sam_image_size
        self.num_classes_per_sample = num_classes_per_sample
        self.exclude_val = exclude_val
        self.explanatory = explanatory
        
        # サンプルレートの正規化
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()
        
        # データセットの初期化
        self.datasets = dataset.split("||")
        self.all_datasets = []
        
        # Semantic Segmentation Dataset
        if "sem_seg" in self.datasets:
            self.all_datasets.append(
                SemSegDataset(
                    base_image_dir,
                    gemma_processor.tokenizer,
                    None,  # vision_tower は使用しない
                    samples_per_epoch,
                    precision,
                    gemma_image_size,
                    num_classes_per_sample,
                    exclude_val,
                    sem_seg_data,
                )
            )
        
        # Referring Segmentation Dataset
        if "refer_seg" in self.datasets:
            self.all_datasets.append(
                ReferSegDataset(
                    base_image_dir,
                    gemma_processor.tokenizer,
                    None,  # vision_tower は使用しない
                    samples_per_epoch,
                    precision,
                    gemma_image_size,
                    num_classes_per_sample,
                    exclude_val,
                    refer_seg_data,
                )
            )
        
        # VQA Dataset
        if "vqa" in self.datasets:
            self.all_datasets.append(
                VQADataset(
                    base_image_dir,
                    gemma_processor.tokenizer,
                    None,  # vision_tower は使用しない
                    samples_per_epoch,
                    precision,
                    gemma_image_size,
                    num_classes_per_sample,
                    exclude_val,
                    vqa_data,
                )
            )
        
        # Reasoning Segmentation Dataset
        if "reason_seg" in self.datasets:
            self.all_datasets.append(
                ReasonSegDataset(
                    base_image_dir,
                    gemma_processor.tokenizer,
                    None,  # vision_tower は使用しない
                    samples_per_epoch,
                    precision,
                    gemma_image_size,
                    num_classes_per_sample,
                    exclude_val,
                    reason_seg_data,
                )
            )
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx) -> Tuple[str, Image.Image, str, torch.Tensor, torch.Tensor]:
        """
        アイテムを取得してデュアルストリーム処理用の形式で返す
        
        Returns:
            image_path: 画像のパス
            image: PIL Image（Gemma-3とSAMの両方で使用）
            text_prompt: テキストプロンプト
            mask: セグメンテーションマスク
            label: ラベルテンソル
        """
        # ランダムにデータセットを選択
        dataset_idx = np.random.choice(len(self.all_datasets), p=self.sample_rate)
        dataset = self.all_datasets[dataset_idx]
        
        # データセットからサンプルを取得
        sample_idx = random.randint(0, len(dataset) - 1)
        
        try:
            # 各データセットの__getitem__の出力形式に応じて処理
            sample = dataset[sample_idx]
            
            if len(sample) >= 4:
                image_path = sample[0]
                image = sample[1]  # 通常はPIL ImageまたはTensor
                text_prompt = sample[2] if len(sample) > 2 else "Describe this image."
                mask = sample[3] if len(sample) > 3 else torch.zeros(1, 1024, 1024)
                label = sample[4] if len(sample) > 4 else torch.tensor(0)
            else:
                # フォールバック処理
                image_path = f"sample_{idx}"
                image = Image.new('RGB', (224, 224), color='red')
                text_prompt = "Describe this image."
                mask = torch.zeros(1, 1024, 1024)
                label = torch.tensor(0)
            
            # 画像がTensorの場合、PIL Imageに変換
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:  # CHW format
                    image = image.permute(1, 2, 0)  # HWC format
                image = image.cpu().numpy()
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                # numpy arrayの場合
                if isinstance(image, np.ndarray):
                    if image.dtype != np.uint8:
                        image = (image * 255).astype(np.uint8)
                    image = Image.fromarray(image)
                else:
                    # その他の場合はダミー画像
                    image = Image.new('RGB', (224, 224), color='blue')
            
            # マスクの処理
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(np.array(mask)).float()
            
            # ラベルの処理
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)
            
            return image_path, image, text_prompt, mask, label
            
        except Exception as e:
            print(f"データ取得エラー (idx={idx}, dataset_idx={dataset_idx}): {e}")
            # エラー時のフォールバック
            return (
                f"error_sample_{idx}",
                Image.new('RGB', (224, 224), color='gray'),
                "This is a fallback image due to data loading error.",
                torch.zeros(1, 1024, 1024),
                torch.tensor(0)
            )

class LisaGemma3ValDataset(torch.utils.data.Dataset):
    """
    LISA-Gemma3用の評価データセット
    """
    
    def __init__(
        self,
        base_image_dir: str,
        gemma_processor: AutoProcessor,
        val_dataset: str,
        gemma_image_size: int = 896,
        sam_image_size: int = 1024,
    ):
        self.base_image_dir = base_image_dir
        self.gemma_processor = gemma_processor
        self.gemma_image_size = gemma_image_size
        self.sam_image_size = sam_image_size
        
        # 評価用データセットの初期化
        if "refer_seg" in val_dataset.lower():
            self.dataset = ReferSegDataset(
                base_image_dir,
                gemma_processor.tokenizer,
                None,
                1000,  # サンプル数
                "bf16",
                gemma_image_size,
                3,
                False,
                val_dataset,
            )
        elif "sem_seg" in val_dataset.lower():
            self.dataset = SemSegDataset(
                base_image_dir,
                gemma_processor.tokenizer,
                None,
                1000,  # サンプル数
                "bf16",
                gemma_image_size,
                3,
                False,
                val_dataset,
            )
        else:
            raise ValueError(f"不明な評価データセット: {val_dataset}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Tuple[str, Image.Image, str, torch.Tensor, torch.Tensor]:
        """評価用サンプルの取得"""
        try:
            sample = self.dataset[idx]
            
            # 訓練用データセットと同じ形式で処理
            if len(sample) >= 4:
                image_path = sample[0]
                image = sample[1]
                text_prompt = sample[2] if len(sample) > 2 else "Segment the object in this image."
                mask = sample[3] if len(sample) > 3 else torch.zeros(1, 1024, 1024)
                label = sample[4] if len(sample) > 4 else torch.tensor(0)
            else:
                raise ValueError("評価データセットのサンプル形式が無効です")
            
            # 画像の型変換（訓練用と同じ処理）
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    image = image.permute(1, 2, 0)
                image = image.cpu().numpy()
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    if image.dtype != np.uint8:
                        image = (image * 255).astype(np.uint8)
                    image = Image.fromarray(image)
                else:
                    image = Image.new('RGB', (224, 224), color='blue')
            
            # マスクとラベルの処理
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(np.array(mask)).float()
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)
            
            return image_path, image, text_prompt, mask, label
            
        except Exception as e:
            print(f"評価データ取得エラー (idx={idx}): {e}")
            # エラー時のフォールバック
            return (
                f"val_error_sample_{idx}",
                Image.new('RGB', (224, 224), color='yellow'),
                "This is a fallback validation image.",
                torch.zeros(1, 1024, 1024),
                torch.tensor(0)
            )
