# LISA-Llama4 移行実装仕様書

**プロジェクト:** LISA-Gemma → LISA-Llama4-Scout-17B-16E-Instruct  
**作成日:** 2025 年 1 月 16 日  
**バージョン:** v2.0 - 公式実装対応版

## 📋 目次

1. [概要と移行戦略](#概要と移行戦略)
2. [公式実装に基づく技術仕様](#公式実装に基づく技術仕様)
3. [統合実装戦略](#統合実装戦略)
4. [完全設定実装](#完全設定実装)
5. [検証手順](#検証手順)

---

## 📊 概要と移行戦略

### 移行の背景

- **現状:** Gemma-3-4b-it + SAM による LISA 実装
- **目標:** Llama-4-Scout-17B-16E-Instruct + SAM による高性能 LISA 実装
- **新戦略:** 公式実装に基づく包括的設定による一括移行

### 主要な変更点

1. **VLM の変更:** Gemma-3-4b-it → Llama-4-Scout-17B-16E-Instruct
2. **アーキテクチャ:** Dense → MoE (16 experts) + Native Multimodal
3. **実装アプローチ:** 段階的修正 → 公式仕様に基づく包括的実装
4. **設定構造:** 完全な text_config + vision_config + MoE 設定

---

## 🔍 公式実装に基づく技術仕様

### Llama-4 公式アーキテクチャ

| 項目                  | Llama-4-Scout-17B-16E-Instruct         |
| --------------------- | -------------------------------------- |
| **パラメータ数**      | 17B (activated) / 109B (total)         |
| **アーキテクチャ**    | MoE (16 experts) + Early Fusion        |
| **コンテキスト長**    | 10M tokens                             |
| **ビジョン処理**      | Native Multimodal (Early Fusion)       |
| **入力解像度**        | 最大 1120×1120 (可変)                  |
| **モデルクラス**      | `Llama4ForConditionalGeneration`       |
| **プロセッサー**      | `AutoProcessor`                        |
| **注意機構**          | `attn_implementation="flex_attention"` |
| **必要 transformers** | ≥4.51.0                                |

### 公式実装パターン

```python
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="flex_attention",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# マルチモーダル入力処理
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]
}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)
```

---

## 🎯 統合実装戦略

### 新アプローチ: 公式ベース + SAM 統合

```python
# 基本構造
class Llama4LisaConfig(PretrainedConfig):
    model_type = "llama4_lisa"

    def __init__(
        self,
        # 基本Llama-4設定
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=10_000_000,  # 10M context

        # MoE設定
        num_experts=16,
        num_experts_per_tok=1,

        # ビジョン設定
        vision_config=None,
        text_config=None,

        # LISA固有設定
        sam_checkpoint_path="./sam_vit_h_4b8939.pth",
        train_mask_decoder=True,
        out_dim=256,

        # 損失重み
        ce_loss_weight=1.0,
        dice_loss_weight=0.5,
        bce_loss_weight=2.0,
        **kwargs
    ):
        # 標準Llama-4設定の初期化
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings

        # MoE設定
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # ネイティブマルチモーダル設定
        if text_config is None:
            text_config = {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_key_value_heads,
                "max_position_embeddings": max_position_embeddings,
                "model_type": "llama",
                "torch_dtype": "bfloat16"
            }

        if vision_config is None:
            vision_config = {
                "hidden_size": 1024,
                "image_size": 1120,
                "intermediate_size": 4096,
                "num_attention_heads": 16,
                "num_hidden_layers": 24,
                "num_channels": 3,
                "patch_size": 14,
                "model_type": "siglip_vision_model"
            }

        self.text_config = Llama4TextConfig(**text_config)
        self.vision_config = Llama4VisionConfig(**vision_config)

        # LISA固有パラメータ
        self.sam_checkpoint_path = sam_checkpoint_path
        self.train_mask_decoder = train_mask_decoder
        self.out_dim = out_dim
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight

        super().__init__(**kwargs)

class Llama4TextConfig(PretrainedConfig):
    model_type = "llama"

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=10_000_000,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        attention_bias=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias

        super().__init__(**kwargs)

class Llama4VisionConfig(PretrainedConfig):
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1024,
        image_size=1120,
        intermediate_size=4096,
        num_attention_heads=16,
        num_hidden_layers=24,
        num_channels=3,
        patch_size=14,
        attention_dropout=0.0,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout

        super().__init__(**kwargs)
```

### デュアルストリーム統合戦略

```python
class Llama4LisaForConditionalGeneration(Llama4ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        # SAM統合 (セグメンテーション専用)
        self.visual_model = build_sam_vit_h(config.sam_checkpoint_path)

        # SAMは凍結 (Llama-4ビジョンと独立動作)
        for param in self.visual_model.parameters():
            param.requires_grad = False

        # マスクデコーダーのみ学習可能
        if config.train_mask_decoder:
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # ビジョン統合層
        self.vision_integration = nn.Sequential(
            nn.Linear(config.vision_config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.out_dim)
        )

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        sam_images=None,
        seg_token_positions=None,
        ground_truth_masks=None,
        **kwargs
    ):
        # 1. Llama-4ネイティブフォワード
        llama4_outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **kwargs
        )

        # 2. セグメンテーション処理 (SAM)
        if sam_images is not None and seg_token_positions is not None:
            seg_outputs = self._process_segmentation(
                sam_images,
                llama4_outputs.hidden_states,
                seg_token_positions,
                ground_truth_masks
            )

            return Llama4LisaOutput(
                **llama4_outputs,
                **seg_outputs
            )

        return llama4_outputs
```

---

## 🚀 完全設定実装

### config_llama4.py (完全版)

```python
import os
from dataclasses import dataclass

@dataclass
class Llama4LisaTrainingConfig:
    """Llama4-LISA学習設定"""

    # === 基本モデル設定 ===
    MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    MODEL_TYPE = "llama4_lisa"

    # === ハードウェア設定 ===
    DEVICE_MAP = "auto"
    TORCH_DTYPE = "bfloat16"
    ATTN_IMPLEMENTATION = "flex_attention"  # Llama-4推奨

    # === トレーニング設定 ===
    BATCH_SIZE = 1  # A100対応
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 10
    WARMUP_RATIO = 0.03
    WEIGHT_DECAY = 0.0
    MAX_GRAD_NORM = 1.0

    # === SAM設定 ===
    SAM_CHECKPOINT_PATH = "/lambda/nfs/lisa-gemma-project-fs/data/weights/sam_vit_h_4b8939.pth"
    TRAIN_MASK_DECODER = True
    OUT_DIM = 256

    # === 損失重み ===
    CE_LOSS_WEIGHT = 1.0
    DICE_LOSS_WEIGHT = 0.5
    BCE_LOSS_WEIGHT = 2.0

    # === データセット設定 ===
    DATASET_BASE_DIR = "/lambda/nfs/lisa-gemma-project-fs/data"
    SAMPLES_PER_EPOCH = 1000
    SAMPLE_RATE = [9, 3, 3, 1]  # sem_seg, refer_seg, vqa, reason_seg

    # === 特殊トークン ===
    SEG_TOKEN = "[SEG]"
    IMAGE_TOKEN = "<image>"

    # === 出力設定 ===
    OUTPUT_DIR = "/lambda/nfs/lisa-gemma-project-fs/outputs/llama4_lisa"
    LOGGING_DIR = "/lambda/nfs/lisa-gemma-project-fs/logs/llama4_lisa"
    SAVE_STEPS = 100
    EVAL_STEPS = 50

    # === WandB設定 ===
    WANDB_PROJECT = "lisa-llama4"
    WANDB_NAME = "llama4-scout-17b-lisa"

    def __post_init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.LOGGING_DIR, exist_ok=True)
```

---

## ✅ 検証手順

### ステップ 1: 包括的設定確認

```bash
python verify_llama4_complete_config.py
```

### ステップ 2: 統合アーキテクチャ確認

```bash
python verify_llama4_integrated_architecture.py
```

### ステップ 3: エンドツーエンドテスト

```bash
python verify_llama4_end_to_end.py
```

---

## 🚨 重要な変更点

### 1. 実装戦略の根本的変更

- **旧:** 段階的パラメータ追加
- **新:** 公式実装に基づく包括的設定

### 2. ビジョン処理の統合

- **Llama-4 ネイティブビジョン:** 理解・推論用
- **SAM:** セグメンテーション専用
- **統合層:** 特徴融合

### 3. 設定構造の完全化

- 全必要パラメータを事前定義
- 公式仕様準拠
- エラー駆動修正の廃止

---

**最終更新:** 2025 年 1 月 16 日  
**作成者:** Claude-4 + 公式実装調査  
**戦略:** 包括的実装アプローチ
