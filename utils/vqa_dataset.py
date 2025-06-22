import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from .constants import (
    DEFAULT_IMAGE_TOKEN,
    SAM_PIXEL_MEAN,
    SAM_PIXEL_STD,
    SAM_IMAGE_SIZE,
    DEFAULT_IGNORE_LABEL
)

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


def preprocess_multimodal(source, mm_use_im_start_end=False):
    """Gemma-3用の簡略化されたマルチモーダル前処理"""
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
    return source


class VQADataset(torch.utils.data.Dataset):
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
        vqa_data="llava_instruct_150k",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        
        # Gemma-3では画像変換を簡略化
        self.target_size = image_size

        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")
        
        # VQAデータファイルの存在確認
        vqa_json_path = os.path.join(DATA_DIR, "{}.json".format(vqa_data))
        if not os.path.exists(vqa_json_path):
            raise FileNotFoundError(f"必須VQAデータファイルが見つかりません: {vqa_json_path}")
        
        with open(vqa_json_path) as f:
            vqa_data = json.load(f)
        self.vqa_data = vqa_data

        if len(self.vqa_data) == 0:
            raise ValueError(f"VQAデータが空です: {vqa_json_path}")

        print("vqa_data: ", len(self.vqa_data))

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
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])
        
        # 画像が存在しない場合はダミーデータを返す
        if not os.path.exists(image_path):
            return (
                f"missing_{idx}",
                Image.new('RGB', (224, 224), color='gray'),
                "This image is missing.",
                torch.zeros(1, 224, 224),
                torch.tensor(0)
            )
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        # PIL Imageに変換（Gemma-3用）
        pil_image = Image.fromarray(image)

        conv = default_conversation.copy()
        source = item["conversations"]
        source = preprocess_multimodal(source, mm_use_im_start_end=False)
        
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        
        if len(source) > 0 and roles.get(source[0]["from"]) != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles.get(sentence["from"], conv.roles[j % 2])
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

        # VQAデータセットではマスクは不要
        masks = torch.zeros(1, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label

        # Gemma-3用の形式で返す
        if len(conversations) > 0:
            text_prompt = conversations[0]
        else:
            text_prompt = "Describe this image."

        return (
            image_path,
            pil_image,  # PIL Image形式で返す
            text_prompt,
            masks,
            label,
        )
