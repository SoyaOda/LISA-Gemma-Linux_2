import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.segment_anything.utils.transforms import ResizeLongestSide

from .grefer import G_REFER
from .refer import REFER
from .constants import ANSWER_LIST, SHORT_QUESTION_LIST, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SYSTEM_PROMPT

# 簡単な会話クラス（LLaVA依存を削除）
class SimpleConversation:
    def __init__(self):
        self.messages = []
        self.roles = ["human", "gpt"]
    
    def copy(self):
        new_conv = SimpleConversation()
        new_conv.messages = self.messages.copy()
        return new_conv
    
    def append_message(self, role, message):
        self.messages.append([role, message])
    
    def get_prompt(self):
        if len(self.messages) >= 2:
            return f"<start_of_turn>user\n{self.messages[0][1]}<end_of_turn>\n<start_of_turn>model\n{self.messages[1][1]}<end_of_turn>\n"
        return ""

# デフォルト会話テンプレート
default_conversation = SimpleConversation()


class ReferSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        model_name=None,  # Gemma3モデル名（未使用だが互換性のため）
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        processor=None,  # 親クラスから渡されるプロセッサ（未使用だが互換性のため）
        image_processor=None,  # 親クラスから渡される画像プロセッサ（未使用だが互換性のため）
    ):
        """初期化
        
        Args:
            base_image_dir: ベースとなる画像ディレクトリ
            tokenizer: トークナイザ
            model_name: Gemma3モデル名（互換性のため）
            samples_per_epoch: エポックあたりのサンプル数
            precision: 精度
            image_size: 画像サイズ
            num_classes_per_sample: サンプルあたりのクラス数
            exclude_val: 検証データを除外するか
            refer_seg_data: 参照セグメンテーションデータ
            processor: 親クラスから渡されるプロセッサ（互換性のため）
            image_processor: 親クラスから渡される画像プロセッサ（互換性のため）
        """
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        # CLIPImageProcessorは使用しないが、互換性のため残す
        # self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        # データセットの存在確認
        DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        if not os.path.exists(DATA_DIR):
            raise FileNotFoundError(f"参照セグメンテーションデータディレクトリが見つかりません: {DATA_DIR}")

        self.refer_seg_ds_list = refer_seg_data.split("||")
        self.refer_seg_data = {}
        
        # 各データセットの初期化
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
            
            ref_ids_train = refer_api.getRefIds(split="train")
            if len(ref_ids_train) == 0:
                print(f"警告: データセット {ds} にトレーニングデータが見つかりません")
                continue
                
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

            # 画像パスの構築と存在確認
            valid_images = []
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/mscoco/images/train2014", item["file_name"]
                    )
                
                # ファイルの存在確認
                if os.path.exists(item["file_name"]):
                    valid_images.append(item)
                else:
                    print(f"警告: 画像ファイルが見つかりません: {item['file_name']}")
            
            if len(valid_images) == 0:
                print(f"警告: データセット {ds} に有効な画像が見つかりません")
                continue
                
            refer_seg_ds["images"] = valid_images
            refer_seg_ds["annotations"] = refer_api.Anns

            print(
                "dataset {} (refs {}) (train split) has {} images and {} annotations.".format(
                    ds,
                    splitBy,
                    len(refer_seg_ds["images"]),
                    len(refer_seg_ds["annotations"]),
                )
            )

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [ref]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_data[ds] = refer_seg_ds

        # 有効なデータセットがあるかチェック
        if len(self.refer_seg_data) == 0:
            raise ValueError("有効な参照セグメンテーションデータセットが見つかりません")

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
        # データセットをランダムに選択
        available_datasets = list(self.refer_seg_data.keys())
        if len(available_datasets) == 0:
            raise RuntimeError("利用可能なデータセットがありません")
            
        ds_idx = random.randint(0, len(available_datasets) - 1)
        ds = available_datasets[ds_idx]
        refer_seg_ds = self.refer_seg_data[ds]
        
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        
        if len(images) == 0:
            return self.__getitem__(0)
            
        idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        refs = img2refs.get(image_id, [])
        
        if len(refs) == 0:
            return self.__getitem__(0)

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
                
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
            
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_classes = sampled_sents
        
        # 画像の読み込み
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"画像の読み込みに失敗: {image_path}")
                return self.__getitem__(0)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"画像処理エラー: {e}")
            return self.__getitem__(0)

        # PIL Imageとして返すためのコピーを保存
        pil_image = Image.fromarray(image)

        # SAM用の前処理
        image = self.transform.apply_image(image)
        resize = image.shape[:2]

        # 質問と回答の生成
        questions = []
        answers = []
        for text in sampled_classes:
            text = text.strip()
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

        # 会話の生成
        conversations = []
        i = 0
        while i < len(questions):
            conv = default_conversation.copy()
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        # テンソル前処理
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # マスクの処理
        masks = []
        flag = False
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                else:
                    m_final = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]
                        if len(ann["segmentation"]) == 0:
                            m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = mask.frPyObjects(
                                    ann["segmentation"], image_info["height"], image_info["width"]
                                )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(m, axis=2)
                            m = m.astype(np.uint8)
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]
            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
            else:
                if type(ann["segmentation"][0]) == list:  # polygon
                    rle = mask.frPyObjects(
                        ann["segmentation"], image_info["height"], image_info["width"]
                    )
                else:
                    rle = ann["segmentation"]
                    for i in range(len(rle)):
                        if not isinstance(rle[i]["counts"], bytes):
                            rle[i]["counts"] = rle[i]["counts"].encode()
                m = mask.decode(rle)
                m = np.sum(m, axis=2)
                m = m.astype(np.uint8)
            masks.append(m)

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        # 戻り値の形式を統一（PIL Image、テキストプロンプト、マスク、ラベルを返す）
        text_prompt = conversations[0] if conversations else questions[0] if questions else ""
        
        return (
            image_path,  # 画像パス
            pil_image,   # PIL Image
            text_prompt, # テキストプロンプト
            masks,       # マスク
            label        # ラベル
        )
