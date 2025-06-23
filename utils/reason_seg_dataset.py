import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .constants import (
    DEFAULT_IMAGE_TOKEN, 
    DEFAULT_SEG_TOKEN,
    ANSWER_LIST, 
    EXPLANATORY_QUESTION_LIST, 
    LONG_QUESTION_LIST,
    SHORT_QUESTION_LIST,
    SAM_PIXEL_MEAN,
    SAM_PIXEL_STD,
    SAM_IMAGE_SIZE,
    DEFAULT_IGNORE_LABEL
)
from .data_processing import get_mask_from_json

# 簡単な会話クラス（LLaVAの代替）
class SimpleConversation:
    def __init__(self):
        self.roles = ["USER", "ASSISTANT"]
        self.messages = []
        self.sep = "\n"
        self.sep2 = "</s>"
    
    def copy(self):
        new_conv = SimpleConversation()
        new_conv.messages = self.messages.copy()
        return new_conv
    
    def append_message(self, role, message):
        self.messages.append([role, message])
    
    def get_prompt(self):
        if len(self.messages) == 0:
            return ""
        
        prompt = ""
        for i, (role, message) in enumerate(self.messages):
            if i == 0:
                prompt += f"{role}: {message}"
            else:
                prompt += f"{self.sep}{role}: {message}"
        
        return prompt + self.sep2

# デフォルト会話インスタンス
default_conversation = SimpleConversation()


class ReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor(SAM_PIXEL_MEAN).view(-1, 1, 1)
    pixel_std = torch.Tensor(SAM_PIXEL_STD).view(-1, 1, 1)
    img_size = SAM_IMAGE_SIZE
    ignore_label = DEFAULT_IGNORE_LABEL

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower=None,  # Gemma-3では使用しない
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "bf16",
        image_size: int = SAM_IMAGE_SIZE,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        
        # Gemma-3では画像変換を簡略化
        self.target_size = image_size

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        # 複数データセット対応（例：ReasonSeg|train||AnotherDataset|val）
        self.reason_seg_datasets = reason_seg_data.split("||") if "||" in reason_seg_data else [reason_seg_data]
        all_images = []
        all_jsons = []
        
        for dataset_spec in self.reason_seg_datasets:
            dataset_name, splits = dataset_spec.split("|")
            splits = splits.split("_")
            dataset_images = []
            
            for split in splits:
                images_split = glob.glob(
                    os.path.join(
                        base_image_dir, "reason_seg", dataset_name, split, "*.jpg"
                    )
                )
                dataset_images.extend(images_split)
            
            dataset_jsons = [path.replace(".jpg", ".json") for path in dataset_images]
            
            # 存在確認
            valid_images = []
            valid_jsons = []
            for img, json_path in zip(dataset_images, dataset_jsons):
                if os.path.exists(img) and os.path.exists(json_path):
                    valid_images.append(img)
                    valid_jsons.append(json_path)
            
            if len(valid_images) > 0:
                all_images.extend(valid_images)
                all_jsons.extend(valid_jsons)
                print(f"ReasonSegデータセット '{dataset_name}' ({splits}): {len(valid_images)} サンプル")
            else:
                print(f"警告: ReasonSegデータセット '{dataset_name}' に有効なサンプルがありません")
        
        self.reason_seg_data = (all_images, all_jsons)
        print(f"ReasonSeg総サンプル数: {len(all_images)}")

        if explanatory != -1:
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
            self.img_to_explanation = {}
            
            # 各データセットのexplanatoryファイルを読み込み
            for dataset_spec in self.reason_seg_datasets:
                dataset_name, _ = dataset_spec.split("|")
                explanatory_path = os.path.join(
                        base_image_dir,
                        "reason_seg",
                        dataset_name,
                        "explanatory",
                        "train.json",
                    )
                
                try:
                    with open(explanatory_path) as f:
                        items = json.load(f)
                    for item in items:
                        img_name = item["image"]
                        self.img_to_explanation[img_name] = {
                            "query": item["query"],
                            "outputs": item["outputs"],
                        }
                    print(f"explanatory '{dataset_name}': {len(items)} 説明")
                except FileNotFoundError:
                    print(f"警告: explanatoryファイルが見つかりません: {explanatory_path}")
                except Exception as e:
                    print(f"警告: explanatory読み込みエラー ({dataset_name}): {e}")
            
            if len(self.img_to_explanation) == 0:
                print("警告: explanatory機能を無効にして続行します。")
                self.explanatory = -1  # explanatory機能を無効化
            else:
                print(f"explanatory総数: {len(self.img_to_explanation)}")
        
        # 画像ファイルの存在確認も追加
        if len(all_images) == 0:
            raise FileNotFoundError(f"ReasonSegデータセットに有効な画像ファイルが見つかりません")

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        images, jsons = self.reason_seg_data
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_path = jsons[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        # PIL Imageに変換（Gemma-3プロセッサー用）
        pil_image = Image.fromarray(image)

        try:
            mask, sents, is_sentence = get_mask_from_json(json_path, image)
            if len(sents) >= self.num_classes_per_sample:
                sampled_inds = np.random.choice(
                    list(range(len(sents))), size=self.num_classes_per_sample, replace=False
                )
            else:
                sampled_inds = list(range(len(sents)))
            sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
            sampled_masks = [
                (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
            ]
        except Exception as e:
            print(f"マスク読み込みエラー: {e}")
            # フォールバック処理
            sampled_sents = ["object"]
            sampled_masks = [np.zeros(ori_size, dtype=np.float32)]
            is_sentence = False

        # 会話形式の生成（オリジナルLISAに準拠）
        questions = []
        answers = []
        for text in sampled_sents:
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                question = question_template.format(sent=text)
            else:
                question_template = random.choice(self.short_question_list)
                question = question_template.format(class_name=text.lower())

            questions.append(question)

            # 説明付き回答の処理
            img_name = image_path.split("/")[-1]
            if self.explanatory != -1 and img_name in self.img_to_explanation:
                if random.random() < self.explanatory:
                    # 説明付き回答
                    answer = self.img_to_explanation[img_name]["outputs"]
                    answer = random.choice(self.answer_list) + " {}".format(answer)
                else:
                    # SEGトークンのみ
                    answer = random.choice(self.answer_list)
            else:
                # SEGトークンのみ
                answer = random.choice(self.answer_list)

            answers.append(answer)

        # 会話プロンプトの構築（Gemma-3形式）
        if len(questions) > 0 and len(answers) > 0:
            # 最初の質問と回答を使用
            conversation_text = f"<start_of_turn>user\n{questions[0]}<end_of_turn>\n<start_of_turn>model\n{answers[0]}<end_of_turn>"
        else:
            # フォールバック
            conversation_text = f"<start_of_turn>user\nSegment the object in this image.<end_of_turn>\n<start_of_turn>model\n{random.choice(self.answer_list)}<end_of_turn>"

        # マスクの処理
        if len(sampled_masks) > 0:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
        else:
            masks = torch.zeros(1, *ori_size)

        # ラベルの生成
        label = torch.ones(*ori_size) * self.ignore_label

        return (
            image_path,
            pil_image,  # PIL Image形式で返す
            conversation_text,  # 会話形式のテキスト
            masks,
            label,
        )
