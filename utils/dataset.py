"""
LISA-Gemma3 デュアルストリーム・データパイプライン
仕様書第3章に従った実装
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
from torchvision import transforms

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .vqa_dataset import VQADataset

# 設定ファイルのインポート - 実行時にどの設定が使われているかを判定
try:
    import config_small_test as config  # 小規模テスト用設定を優先
except ImportError:
    try:
        import config_linux as config  # フォールバック：通常設定
    except ImportError:
        # 設定ファイルが見つからない場合のデフォルト値
        class DefaultConfig:
            MODEL_MAX_LENGTH = 2048
        config = DefaultConfig()

# デフォルト設定
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SEG_TOKEN = "[SEG]"
IGNORE_INDEX = -100

def preprocess_sam_image(image: Image.Image, target_size: int = 1024) -> torch.Tensor:
    """
    SAM用画像前処理：1024x1024にリサイズ・パディング・正規化
    """
    # 1. 最長辺を1024にリサイズ
    w, h = image.size
    if max(w, h) != target_size:
        if w > h:
            new_w, new_h = target_size, int(h * target_size / w)
        else:
            new_w, new_h = int(w * target_size / h), target_size
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    # 2. 1024x1024にパディング
    w, h = image.size
    pad_w = (target_size - w) // 2
    pad_h = (target_size - h) // 2
    
    # パディング用の新しい画像を作成
    padded_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    padded_image.paste(image, (pad_w, pad_h))
    
    # 3. テンソル化と正規化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[123.675/255, 116.28/255, 103.53/255],  # SAMの標準正規化値
            std=[58.395/255, 57.12/255, 57.375/255]
        )
    ])
    
    return transform(padded_image)

def preprocess_mask(mask: np.ndarray, target_size: int = 1024) -> torch.Tensor:
    """
    マスクの前処理
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # マスクの形状を検証
    if mask.size == 0:
        raise ValueError("空のマスクです")
    
    if mask.ndim < 2:
        raise ValueError(f"マスクの次元が不正です: {mask.ndim}D (最低2D必要)")
    
    # マスクを2次元に変換
    if mask.ndim == 2:
        h, w = mask.shape
        mask_2d = mask
    elif mask.ndim == 3:
        if mask.shape[0] == 1:  # (1, H, W)
            mask_2d = mask[0]
            h, w = mask_2d.shape
        elif mask.shape[-1] == 1:  # (H, W, 1)
            mask_2d = mask[:, :, 0]
            h, w = mask_2d.shape
        elif mask.shape[0] == 3 or mask.shape[-1] == 3:  # RGB マスク
            # RGB マスクの場合、最初のチャンネルを使用
            if mask.shape[0] == 3:  # (3, H, W)
                mask_2d = mask[0]
                h, w = mask_2d.shape
            else:  # (H, W, 3)
                mask_2d = mask[:, :, 0]
                h, w = mask_2d.shape
        else:
            # その他の場合、最後の2次元を使用
            mask_2d = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[0]
            h, w = mask_2d.shape
    else:
        raise ValueError(f"サポートされていないマスクの次元: {mask.ndim}D")
    
    # サイズの検証
    if h == 0 or w == 0:
        raise ValueError(f"無効なマスクサイズ: {h}x{w}")
    
    # マスクのリサイズ（OpenCVのバグ回避のためPillowを使用）
    if (h, w) != (target_size, target_size):
        try:
            # numpy -> PIL Image -> リサイズ -> numpy
            mask_uint8 = mask_2d.astype(np.uint8)
            mask_pil = Image.fromarray(mask_uint8, mode='L')  # グレースケール
            mask_resized_pil = mask_pil.resize((target_size, target_size), Image.NEAREST)
            mask_2d = np.array(mask_resized_pil)
        except Exception as e:
            raise ValueError(f"マスクリサイズ失敗 (元サイズ: {h}x{w}, target: {target_size}x{target_size}): {e}")
    
    # テンソル化 (常に1次元追加して (1, H, W) 形式にする)
    if mask_2d.ndim == 2:
        mask_2d = mask_2d[None, ...]  # (H, W) -> (1, H, W)
    
    return torch.from_numpy(mask_2d).float()

class HybridDataset(torch.utils.data.Dataset):
    """
    仕様書第3章.2 HybridDatasetの実装
    デュアルストリーム処理：Gemma用とSAM用の2系統前処理を同時実行
    """

    def __init__(
        self,
        base_image_dir: str,
        gemma_processor: AutoProcessor,
        samples_per_epoch: int = 500 * 8 * 2 * 10,
        precision: str = "bf16",
        gemma_image_size: int = 896,  # Gemma-3のSigLIPエンコーダ要求仕様
        sam_image_size: int = 1024,   # SAM-ViTエンコーダ要求仕様
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

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        仕様書第3章.2 デュアル前処理の実装
        
        Returns:
            Dict containing:
                - images_for_gemma: Gemma-3用前処理済み画像テンソル (3, 896, 896)
                - images_for_sam: SAM用前処理済み画像テンソル (3, 1024, 1024)
                - input_ids: トークン化されたテキスト
                - labels: ラベル
                - seg_token_mask: [SEG]トークンの位置マスク
                - ground_truth_mask: 正解マスク
                - has_mask: マスクデータの有無フラグ
        """
        # ランダムにデータセットを選択
        dataset_idx = np.random.choice(len(self.all_datasets), p=self.sample_rate)
        dataset = self.all_datasets[dataset_idx]
        
        # データセットからサンプルを取得
        sample_idx = random.randint(0, len(dataset) - 1)
        
        # 各データセットの__getitem__の出力形式に応じて処理
        sample = dataset[sample_idx]
        
        if len(sample) < 4:
            raise ValueError(f"データセットサンプルの形式が不正です: {len(sample)} 要素 (最低4要素必要)")
        
        image_path = sample[0]
        image = sample[1]  # PIL Image
        text_prompt = sample[2] if len(sample) > 2 else "Describe this image. [SEG]"
        mask = sample[3] if len(sample) > 3 else None
        label = sample[4] if len(sample) > 4 else torch.tensor(0)
        
        # 画像の型チェックと変換
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:  # CHW format
                image = image.permute(1, 2, 0)  # HWC format
            image = image.cpu().numpy()
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(f"サポートされていない画像型: {type(image)}")
        
        # 画像サイズの検証
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError(f"無効な画像サイズ: {image.size}")
        
        # ===================================================================
        # デュアルストリーム前処理（仕様書第3章の核心部分）
        # ===================================================================
        
        # パスウェイA: Gemma-3用前処理
        # AutoProcessorを使用してGemma-3のSigLIPエンコーダ仕様に合わせる
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        
        try:
            gemma_processed = self.gemma_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
        except Exception as e:
            raise RuntimeError(f"Gemma前処理に失敗 (idx={idx}): {e}")
        
        # Gemma用画像テンソル (3, 896, 896)
        images_for_gemma = gemma_processed['pixel_values'].squeeze(0)
        input_ids = gemma_processed['input_ids'].squeeze(0)
        attention_mask = gemma_processed['attention_mask'].squeeze(0)
        
        # パスウェイB: SAM用前処理
        # カスタム前処理でSAM-ViT仕様に合わせる
        try:
            images_for_sam = preprocess_sam_image(image, self.sam_image_size)  # (3, 1024, 1024)
        except Exception as e:
            raise RuntimeError(f"SAM前処理に失敗 (idx={idx}): {e}")
        
        # [SEG]トークンの位置を特定
        seg_token_id = self.gemma_processor.tokenizer.convert_tokens_to_ids("[SEG]")
        if seg_token_id is None:
            raise ValueError("[SEG]トークンがトークナイザーに見つかりません")
        
        seg_token_mask = (input_ids == seg_token_id)
        
        # マスクの前処理
        has_mask = mask is not None and (isinstance(mask, (torch.Tensor, np.ndarray)) and mask.sum() > 0)
        if has_mask:
            try:
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)
                ground_truth_mask = preprocess_mask(mask_np, self.sam_image_size)
            except Exception as e:
                raise RuntimeError(f"マスク前処理に失敗 (idx={idx}): {e}")
        else:
            ground_truth_mask = torch.zeros(1, self.sam_image_size, self.sam_image_size)
        
        # ラベルの処理
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        
        return {
            "images_for_gemma": images_for_gemma,      # (3, 896, 896)
            "images_for_sam": images_for_sam,          # (3, 1024, 1024)
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
            "seg_token_mask": seg_token_mask,
            "ground_truth_mask": ground_truth_mask,    # (1, 1024, 1024)
            "has_mask": has_mask,
            "image_path": image_path,
            "text_prompt": text_prompt,
        }

def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    仕様書第3章.3 バッチの結合 (collate_fn)
    デュアルストリーム対応のカスタムcollate関数
    
    最大シーケンス長は設定ファイルのMODEL_MAX_LENGTHを自動的に使用:
    - config_small_test.py が利用可能な場合: 512 (メモリ効率優先)
    - config_linux.py のみの場合: 2048 (通常設定)
    - 設定ファイルなしの場合: 2048 (デフォルト)
    """
    # 各キーごとにデータを収集
    images_for_gemma = []
    images_for_sam = []
    input_ids = []
    attention_masks = []
    labels = []
    seg_token_masks = []
    ground_truth_masks = []
    has_masks = []
    image_paths = []
    text_prompts = []
    
    for item in batch:
        images_for_gemma.append(item["images_for_gemma"])
        images_for_sam.append(item["images_for_sam"])
        input_ids.append(item["input_ids"])
        attention_masks.append(item["attention_mask"])
        
        # labelsの次元を統一（常に1次元テンソルにする）
        label = item["labels"]
        if isinstance(label, torch.Tensor):
            if label.dim() == 0:  # スカラーテンソル
                label = label.unsqueeze(0)  # (1,) に変換
            elif label.dim() > 1:  # 多次元テンソル
                label = label.flatten()  # 1次元に変換
            # label.dim() == 1 の場合はそのまま
        else:
            label = torch.tensor([label])  # スカラー値を1次元テンソルに変換
        labels.append(label)
        
        seg_token_masks.append(item["seg_token_mask"])
        if item["has_mask"]:
            ground_truth_masks.append(item["ground_truth_mask"])
        has_masks.append(item["has_mask"])
        image_paths.append(item["image_path"])
        text_prompts.append(item["text_prompt"])
    
    # テンソルのスタック
    images_for_gemma = torch.stack(images_for_gemma)  # (B, 3, 896, 896)
    images_for_sam = torch.stack(images_for_sam)      # (B, 3, 1024, 1024)
    
    # テキストシーケンスとラベルの長さ統一（パディング）
    max_length = max(ids.size(0) for ids in input_ids)
    
    # ラベルの最大長も考慮
    max_label_length = max(label.size(0) for label in labels)
    # ラベルとinput_idsの長さを統一する場合
    unified_max_length = max(max_length, max_label_length)
    
    # 最大長制限を適用（メモリ不足を防ぐため）
    # 設定ファイルのMODEL_MAX_LENGTHを使用
    unified_max_length = min(unified_max_length, config.MODEL_MAX_LENGTH)
    
    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []
    padded_seg_token_masks = []
    
    pad_token_id = 0  # パディングトークンID
    
    for i in range(len(input_ids)):
        current_length = input_ids[i].size(0)
        pad_length = unified_max_length - current_length
        
        if pad_length > 0:
            # input_idsのパディング
            padded_input_ids.append(
                F.pad(input_ids[i], (0, pad_length), value=pad_token_id)
            )
            padded_attention_masks.append(
                F.pad(attention_masks[i], (0, pad_length), value=0)
            )
            padded_seg_token_masks.append(
                F.pad(seg_token_masks[i], (0, pad_length), value=False)
            )
        else:
            padded_input_ids.append(input_ids[i][:unified_max_length])
            padded_attention_masks.append(attention_masks[i][:unified_max_length])
            padded_seg_token_masks.append(seg_token_masks[i][:unified_max_length])
        
        # labelsのパディング（常に統一長にする）
        current_label_length = labels[i].size(0)
        label_pad_length = unified_max_length - current_label_length
        
        if label_pad_length > 0:
            padded_labels.append(
                F.pad(labels[i], (0, label_pad_length), value=-100)
            )
        elif label_pad_length < 0:
            padded_labels.append(labels[i][:unified_max_length])  # 切り詰め
        else:
            padded_labels.append(labels[i])  # 既に適切な長さ
    
    # マスクが存在するサンプルのみをスタック
    if ground_truth_masks:
        ground_truth_masks = torch.stack(ground_truth_masks)
    else:
        ground_truth_masks = None

    return {
        "images_for_gemma": images_for_gemma,           # (B, 3, 896, 896)
        "images_for_sam": images_for_sam,               # (B, 3, 1024, 1024)
        "input_ids": torch.stack(padded_input_ids),     # (B, unified_max_length)
        "attention_mask": torch.stack(padded_attention_masks),  # (B, unified_max_length)
        "labels": torch.stack(padded_labels).long(),    # (B, unified_max_length) - 必ずlong型に変換
        "seg_token_mask": torch.stack(padded_seg_token_masks),  # (B, unified_max_length)
        "ground_truth_mask": ground_truth_masks,        # (num_masks, 1, 1024, 1024) or None
        "has_mask": has_masks,                          # List[bool]
        "image_paths": image_paths,                     # List[str]
        "text_prompts": text_prompts,                   # List[str]
    }

# 後方互換性のためのエイリアス
LisaGemma3Dataset = HybridDataset
collate_fn_gemma3 = collate_fn

class LisaGemma3ValDataset(torch.utils.data.Dataset):
    """
    LISA-Gemma3用の評価データセット
    デュアルストリーム対応
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
            # ReasonSegをデフォルトとする
            self.dataset = ReasonSegDataset(
                        base_image_dir,
                gemma_processor.tokenizer,
                None,
                1000,
                "bf16",
                gemma_image_size,
                3,
                False,
                "ReasonSeg|val",
            )

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """評価用サンプルの取得（デュアルストリーム対応）"""
        sample = self.dataset[idx]
        
        # 訓練用データセットと同じ形式で処理
        if len(sample) < 4:
            raise ValueError(f"評価データセットのサンプル形式が不正です: {len(sample)} 要素 (最低4要素必要)")
        
        image_path = sample[0]
        image = sample[1]
        text_prompt = sample[2] if len(sample) > 2 else "Segment the object in this image. [SEG]"
        mask = sample[3] if len(sample) > 3 else None
        label = sample[4] if len(sample) > 4 else torch.tensor(0)
        
        # 画像の型変換
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(f"サポートされていない画像型: {type(image)}")
        
        # 画像サイズの検証
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError(f"無効な画像サイズ: {image.size}")
        
        # デュアルストリーム前処理（訓練用と同じ）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        
        try:
            gemma_processed = self.gemma_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
        except Exception as e:
            raise RuntimeError(f"評価用Gemma前処理に失敗 (idx={idx}): {e}")
        
        images_for_gemma = gemma_processed['pixel_values'].squeeze(0)
        input_ids = gemma_processed['input_ids'].squeeze(0)
        attention_mask = gemma_processed['attention_mask'].squeeze(0)
        
        try:
            images_for_sam = preprocess_sam_image(image, self.sam_image_size)
        except Exception as e:
            raise RuntimeError(f"評価用SAM前処理に失敗 (idx={idx}): {e}")
        
        seg_token_id = self.gemma_processor.tokenizer.convert_tokens_to_ids("[SEG]")
        if seg_token_id is None:
            raise ValueError("[SEG]トークンがトークナイザーに見つかりません")
        
        seg_token_mask = (input_ids == seg_token_id)
        
        # マスクの処理
        has_mask = mask is not None and (isinstance(mask, (torch.Tensor, np.ndarray)) and mask.sum() > 0)
        if has_mask:
            try:
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)
                ground_truth_mask = preprocess_mask(mask_np, self.sam_image_size)
            except Exception as e:
                raise RuntimeError(f"評価用マスク前処理に失敗 (idx={idx}): {e}")
        else:
            ground_truth_mask = torch.zeros(1, self.sam_image_size, self.sam_image_size)
        
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        
        return {
            "images_for_gemma": images_for_gemma,
            "images_for_sam": images_for_sam,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
            "seg_token_mask": seg_token_mask,
            "ground_truth_mask": ground_truth_mask,
            "has_mask": has_mask,
            "image_path": image_path,
            "text_prompt": text_prompt,
        }
