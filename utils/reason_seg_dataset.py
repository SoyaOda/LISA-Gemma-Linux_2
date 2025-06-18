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

        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
                )
            )
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples: ", len(images))

        if explanatory != -1:
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
            self.img_to_explanation = {}
            explanatory_path = os.path.join(
                    base_image_dir,
                    "reason_seg",
                    reason_seg_data,
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
            print("len(self.img_to_explanation): ", len(self.img_to_explanation))
            except FileNotFoundError:
                print(f"警告: explanatoryファイルが見つかりません: {explanatory_path}")
                print("explanatory機能を無効にして続行します。")
                self.explanatory = -1  # explanatory機能を無効化
        
        # 画像ファイルの存在確認も追加
        if len(images) == 0:
            raise FileNotFoundError(f"画像ファイルが見つかりません: {os.path.join(base_image_dir, 'reason_seg', reason_seg_data)}")

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
        # PIL Imageに変換（Gemma-3用）
        pil_image = Image.fromarray(image)

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

        # 画像の簡単なリサイズ（SAM用）
        height, width = image.shape[:2]
        if height > width:
            new_height = self.target_size
            new_width = int(width * self.target_size / height)
        else:
            new_width = self.target_size
            new_height = int(height * self.target_size / width)
        
        image = cv2.resize(image, (new_width, new_height))
        resize = image.shape[:2]

        image_name = image_path.split("/")[-1]
        if self.explanatory != -1 and image_name in self.img_to_explanation:
            if random.random() < self.explanatory:
                choice = 2
            else:
                choice = random.randint(0, 1)

        questions = []
        answers = []
        for text in sampled_sents:
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))
            else:
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

            # add explanation if applicable
            img_name = image_path.split("/")[-1]
            if self.explanatory != -1 and img_name in self.img_to_explanation:
                if choice == 0:  # [SEG] token
                    answers.append(random.choice(self.answer_list))
                elif choice == 1:  # [SEG] token + text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    answer = random.choice(self.answer_list) + " {}".format(answer)
                    questions[-1] = (
                        DEFAULT_IMAGE_TOKEN
                        + "\n"
                        + text
                        + " {}".format(random.choice(self.explanatory_question_list))
                    )
                    answers.append(answer)
                elif choice == 2:  # vanilla text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    questions[-1] = DEFAULT_IMAGE_TOKEN + "\n" + text
                    answers.append(answer)
                else:
                    raise ValueError("Not implemented yet.")
            else:
                answers.append(random.choice(self.answer_list))

            conversations = []
            conv = default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        image_name = image_path.split("/")[-1]
        if (
            self.explanatory != -1
            and image_name in self.img_to_explanation
            and choice == 2
        ):
            masks = torch.rand(0, *ori_size)
            label = torch.ones(ori_size) * self.ignore_label
        else:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        # Gemma-3用の形式で返す
        if len(conversations) > 0:
            text_prompt = conversations[0]  # 最初の会話を使用
        else:
            text_prompt = "Describe this image."

        return (
            image_path,
            pil_image,  # PIL Image形式で返す
            text_prompt,
            masks,
            label,
        )
