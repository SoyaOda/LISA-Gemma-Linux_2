import glob
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor

from model.segment_anything.utils.transforms import ResizeLongestSide

from .constants import ANSWER_LIST, SHORT_QUESTION_LIST, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SYSTEM_PROMPT
from .data_processing import get_mask_from_json

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


def init_mapillary(base_image_dir):
    """Mapillaryデータセットの初期化"""
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    
    if not os.path.exists(mapillary_data_root):
        print(f"警告: Mapillaryディレクトリが見つかりません: {mapillary_data_root}")
        return [], [], []
    
    config_path = os.path.join(mapillary_data_root, "config_v2.0.json")
    if not os.path.exists(config_path):
        print(f"警告: Mapillary設定ファイルが見つかりません: {config_path}")
        return [], [], []
    
    try:
        with open(config_path) as f:
            mapillary_classes = json.load(f)["labels"]
        mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
        mapillary_classes = np.array(mapillary_classes)
        
        labels_dir = os.path.join(mapillary_data_root, "training", "v2.0", "labels")
        if not os.path.exists(labels_dir):
            print(f"警告: Mapillaryラベルディレクトリが見つかりません: {labels_dir}")
            return [], [], []
            
        mapillary_labels = sorted(glob.glob(os.path.join(labels_dir, "*.png")))
        
        if len(mapillary_labels) == 0:
            print(f"警告: Mapillaryラベルファイルが見つかりません")
            return [], [], []
        
        mapillary_images = [
            x.replace(".png", ".jpg").replace("v2.0/labels", "images")
            for x in mapillary_labels
        ]
        
        # 存在確認
        valid_images = []
        valid_labels = []
        for img, label in zip(mapillary_images, mapillary_labels):
            if os.path.exists(img) and os.path.exists(label):
                valid_images.append(img)
                valid_labels.append(label)
        
        print("mapillary: ", len(valid_images))
        return mapillary_classes, valid_images, valid_labels
    except Exception as e:
        print(f"Mapillary初期化エラー: {e}")
        return [], [], []


def init_ade20k(base_image_dir):
    """ADE20Kデータセットの初期化"""
    ade_path = os.path.join(base_image_dir, "ade20k")
    
    if not os.path.exists(ade_path):
        print(f"警告: ADE20Kディレクトリが見つかりません: {ade_path}")
        return [], [], []
    
    # クラスリストの読み込み
    classes_file = "utils/ade20k_classes.json"
    if not os.path.exists(classes_file):
        print(f"警告: ADE20Kクラスファイルが見つかりません: {classes_file}")
        return [], [], []
    
    try:
        with open(classes_file, "r") as f:
            ade20k_classes = json.load(f)
        ade20k_classes = np.array(ade20k_classes)
        
        images_dir = os.path.join(ade_path, "images", "training")
        if not os.path.exists(images_dir):
            print(f"警告: ADE20K画像ディレクトリが見つかりません: {images_dir}")
            return [], [], []
            
        image_ids = sorted(os.listdir(images_dir))
        ade20k_image_ids = []
        for x in image_ids:
            if x.endswith(".jpg"):
                ade20k_image_ids.append(x[:-4])
                
        ade20k_images = []
        for image_id in ade20k_image_ids:
            ade20k_images.append(
                os.path.join(images_dir, "{}.jpg".format(image_id))
            )
            
        ade20k_labels = [
            x.replace(".jpg", ".png").replace("images", "annotations")
            for x in ade20k_images
        ]
        
        # 存在確認
        valid_images = []
        valid_labels = []
        for img, label in zip(ade20k_images, ade20k_labels):
            if os.path.exists(img) and os.path.exists(label):
                valid_images.append(img)
                valid_labels.append(label)
        
        print("ade20k: ", len(valid_images))
        return ade20k_classes, valid_images, valid_labels
    except Exception as e:
        print(f"ADE20K初期化エラー: {e}")
        return [], [], []


def init_cocostuff(base_image_dir):
    """COCOStuffデータセットの初期化"""
    classes_file = "utils/cocostuff_classes.txt"
    if not os.path.exists(classes_file):
        print(f"警告: COCOStuffクラスファイルが見つかりません: {classes_file}")
        return [], [], []
    
    try:
        cocostuff_classes = []
        with open(classes_file) as f:
            for line in f.readlines()[1:]:
                cocostuff_classes.append(line.strip().split(": ")[-1])
        cocostuff_classes = np.array(cocostuff_classes)

        labels_dir = os.path.join(base_image_dir, "cocostuff", "train2017")
        if not os.path.exists(labels_dir):
            print(f"警告: COCOStuffラベルディレクトリが見つかりません: {labels_dir}")
            return [], [], []

        cocostuff_labels = glob.glob(os.path.join(labels_dir, "*.png"))
        cocostuff_images = [
            x.replace(".png", ".jpg").replace("cocostuff", "coco") for x in cocostuff_labels
        ]

        # 存在確認
        valid_images = []
        valid_labels = []
        for img, label in zip(cocostuff_images, cocostuff_labels):
            if os.path.exists(img) and os.path.exists(label):
                valid_images.append(img)
                valid_labels.append(label)

        print("cocostuff: ", len(valid_images))
        return cocostuff_classes, valid_images, valid_labels
    except Exception as e:
        print(f"COCOStuff初期化エラー: {e}")
        return [], [], []


def init_paco_lvis(base_image_dir):
    """PACO LVISデータセットの初期化"""
    annotations_path = os.path.join(base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json")
    
    if not os.path.exists(annotations_path):
        print(f"警告: PACO LVISアノテーションファイルが見つかりません: {annotations_path}")
        return {}, [], None
    
    try:
        coco_api_paco_lvis = COCO(annotations_path)
        all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
        class_map_paco_lvis = {}
        for cat in all_classes:
            cat_split = cat["name"].strip().split(":")
            if len(cat_split) == 1:
                name = cat_split[0].split("_(")[0]
            else:
                assert len(cat_split) == 2
                obj, part = cat_split
                obj = obj.split("_(")[0]
                part = part.split("_(")[0]
                name = (obj, part)
            class_map_paco_lvis[cat["id"]] = name
        img_ids = coco_api_paco_lvis.getImgIds()
        print("paco_lvis: ", len(img_ids))
        return class_map_paco_lvis, img_ids, coco_api_paco_lvis
    except Exception as e:
        print(f"PACO LVIS初期化エラー: {e}")
        return {}, [], None


def init_pascal_part(base_image_dir):
    """Pascal Partデータセットの初期化"""
    annotations_path = os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    
    if not os.path.exists(annotations_path):
        print(f"警告: Pascal Partアノテーションファイルが見つかりません: {annotations_path}")
        return {}, [], None
    
    try:
        coco_api_pascal_part = COCO(annotations_path)
        all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
        class_map_pascal_part = {}
        for cat in all_classes:
            cat_main, cat_part = cat["name"].strip().split(":")
            name = (cat_main, cat_part)
            class_map_pascal_part[cat["id"]] = name
        img_ids = coco_api_pascal_part.getImgIds()
        print("pascal_part: ", len(img_ids))
        return class_map_pascal_part, img_ids, coco_api_pascal_part
    except Exception as e:
        print(f"Pascal Part初期化エラー: {e}")
        return {}, [], None


class SemSegDataset(torch.utils.data.Dataset):
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
        sem_seg_data="ade20k||cocostuff||mapillary",
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
            sem_seg_data: セマンティックセグメンテーションデータ
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

        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        valid_datasets = []
        
        for ds in self.sem_seg_datas:
            try:
                if ds == "ade20k":
                    classes, images, labels = init_ade20k(base_image_dir)
                elif ds == "cocostuff":
                    classes, images, labels = init_cocostuff(base_image_dir)
                elif ds == "mapillary":
                    classes, images, labels = init_mapillary(base_image_dir)
                elif ds == "paco_lvis":
                    classes, images, labels = init_paco_lvis(base_image_dir)
                elif ds == "pascal_part":
                    classes, images, labels = init_pascal_part(base_image_dir)
                else:
                    print(f"警告: 不明なデータセット: {ds}")
                    continue
                
                if len(images) > 0:
                    self.data2list[ds] = (images, labels)
                    self.data2classes[ds] = classes
                    valid_datasets.append(ds)
                else:
                    print(f"警告: データセット {ds} に有効なデータがありません")
            except Exception as e:
                print(f"データセット {ds} の初期化でエラー: {e}")
                continue

        if len(valid_datasets) == 0:
            raise ValueError("有効なセマンティックセグメンテーションデータセットが見つかりません")

        self.sem_seg_datas = valid_datasets
        print(f"有効なデータセット: {self.sem_seg_datas}")

        if "cocostuff" in self.sem_seg_datas and "cocostuff" in self.data2classes:
            self.cocostuff_class2index = {
                c: i for i, c in enumerate(self.data2classes["cocostuff"])
            }

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
        if len(self.sem_seg_datas) == 0:
            raise RuntimeError("利用可能なデータセットがありません")
            
        ds_idx = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds_idx]

        try:
            if ds in ["paco_lvis", "pascal_part"]:
                return self._get_vlpart_item(ds)
            else:
                return self._get_semseg_item(ds)
        except Exception as e:
            print(f"データセット {ds} からのアイテム取得でエラー: {e}")
            return self.__getitem__(0)

    def _get_vlpart_item(self, ds):
        """VLPartデータセット（paco_lvis, pascal_part）からアイテムを取得"""
        class_map = self.data2classes[ds]
        img_ids, coco_api = self.data2list[ds]
        
        if len(img_ids) == 0:
            raise RuntimeError(f"{ds}にデータがありません")
            
        idx = random.randint(0, len(img_ids) - 1)
        img_id = img_ids[idx]
        image_info = coco_api.loadImgs([img_id])[0]
        file_name = image_info["file_name"]
        
        if ds == "pascal_part":
            image_path = os.path.join(self.base_image_dir, "vlpart", "pascal_part", "VOCdevkit", "VOC2010", "JPEGImages", file_name)
        else:  # paco_lvis
            image_path = os.path.join(self.base_image_dir, "coco", file_name)

        if not os.path.exists(image_path):
            print(f"画像が見つかりません: {image_path}")
            return self.__getitem__(0)

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

        # アノテーションの取得
        ann_ids = coco_api.getAnnIds(imgIds=[img_id])
        anns = coco_api.loadAnns(ann_ids)
        
        if len(anns) == 0:
            print(f"アノテーションが見つかりません: {img_id}")
            return self.__getitem__(0)

        # クラスとマスクの選択
        if len(anns) >= self.num_classes_per_sample:
            sampled_anns = np.random.choice(anns, size=self.num_classes_per_sample, replace=False)
        else:
            sampled_anns = anns

        # マスクの作成
        masks = []
        sampled_classes = []
        for ann in sampled_anns:
            try:
                mask_data = coco_api.annToMask(ann)
                masks.append(mask_data)
                class_name = class_map[ann["category_id"]]
                if isinstance(class_name, tuple):
                    class_name = f"{class_name[0]} {class_name[1]}"
                sampled_classes.append(str(class_name).lower())
            except Exception as e:
                print(f"マスク作成エラー: {e}")
                continue

        if len(masks) == 0:
            return self.__getitem__(0)

        # テキストプロンプトの生成
        question_template = random.choice(self.short_question_list)
        class_name = random.choice(sampled_classes)
        text_prompt = question_template.format(class_name=class_name)

        # マスクをテンソルに変換
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,  # 画像パス
            pil_image,   # PIL Image
            text_prompt, # テキストプロンプト
            masks,       # マスク
            label        # ラベル
        )

    def _get_semseg_item(self, ds):
        """セマンティックセグメンテーションデータセット（ade20k, cocostuff, mapillary）からアイテムを取得"""
        images, labels = self.data2list[ds]
        classes = self.data2classes[ds]
        
        if len(images) == 0:
            raise RuntimeError(f"{ds}にデータがありません")
            
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        label_path = labels[idx]

        if not os.path.exists(image_path):
            print(f"画像が見つかりません: {image_path}")
            return self.__getitem__(0)
            
        if not os.path.exists(label_path):
            print(f"ラベルが見つかりません: {label_path}")
            return self.__getitem__(0)

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

        # ラベルの読み込み
        try:
            label = Image.open(label_path)
            label = np.array(label)
        except Exception as e:
            print(f"ラベル読み込みエラー: {e}")
            return self.__getitem__(0)

        # クラスの選択とマスクの作成
        unique_labels = np.unique(label)
        if ds == "cocostuff":
            # COCOStuffの場合、特定のクラスのみを使用
            valid_labels = [l for l in unique_labels if l < len(classes) and l != 255]
        else:
            valid_labels = [l for l in unique_labels if l < len(classes) and l != 0 and l != 255]

        if len(valid_labels) == 0:
            return self.__getitem__(0)

        # サンプリング
        if len(valid_labels) >= self.num_classes_per_sample:
            sampled_labels = np.random.choice(valid_labels, size=self.num_classes_per_sample, replace=False)
        else:
            sampled_labels = valid_labels

        # マスクとクラス名の作成
        masks = []
        sampled_classes = []
        for label_id in sampled_labels:
            mask = (label == label_id).astype(np.uint8)
            masks.append(mask)
            if label_id < len(classes):
                sampled_classes.append(classes[label_id])
            else:
                sampled_classes.append("unknown")

        if len(masks) == 0:
            return self.__getitem__(0)

        # テキストプロンプトの生成
        question_template = random.choice(self.short_question_list)
        class_name = random.choice(sampled_classes)
        text_prompt = question_template.format(class_name=class_name.lower())

        # マスクをテンソルに変換
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        label_tensor = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,  # 画像パス
            pil_image,   # PIL Image
            text_prompt, # テキストプロンプト
            masks,       # マスク
            label_tensor # ラベル
        )
