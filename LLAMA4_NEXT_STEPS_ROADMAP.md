# LISA-Llama4 移行作業ロードマップ v3.0 (緊急修正版)

**作成日:** 2025 年 1 月 16 日  
**Lambda Cloud IP:** 150.136.95.127  
**重大発見:** 現在の検証は不十分 - 実データセット使用の本格検証が必要

---

## 🚨 **緊急事態: 現状問題の発見**

### ❌ **現在の検証スクリプトの重大問題**

| **問題**       | **現状**         | **本来あるべき姿**     |
| -------------- | ---------------- | ---------------------- |
| **データ使用** | ダミーデータのみ | 実際の HybridDataset   |
| **モデル統合** | 設定確認のみ     | 完全な Llama4+SAM 統合 |
| **処理検証**   | シミュレーション | 実際の processor 処理  |
| **損失計算**   | ダミー値         | 実フォワードパス       |
| **学習検証**   | 表面的ループ     | 実バッチ過学習         |

### 🔍 **Web 調査による重要発見**

**Llama-4-Scout-17B-16E-Instruct 公式仕様:**

- **アーキテクチャ**: MoE (17B activated/109B total, 16 experts)
- **Native Multimodal**: Early fusion (text+image 統合処理)
- **重要設定**: `attn_implementation="flex_attention"`, `torch_dtype=torch.bfloat16`
- **Context**: 10M tokens, transformers >= 4.51.0 必須
- **公式パターン**: `Llama4ForConditionalGeneration` + `AutoProcessor`

### 📋 **Gemma-LISA/Original-LISA との実装比較**

| **要素**         | **Original-LISA**    | **Gemma-LISA**   | **現在の Llama4-LISA**         |
| ---------------- | -------------------- | ---------------- | ------------------------------ |
| **統合方式**     | LlavaLlama 継承      | Gemma3 継承      | Llama4 継承（不完全）          |
| **SAM 統合**     | build_sam_vit_h 使用 | 同じ             | 同じ（実装不足）               |
| **プロジェクタ** | text_hidden_fcs      | mlp_projector    | vision_integration（問題あり） |
| **検証方式**     | 実データセット       | 実 HybridDataset | ダミーデータ（問題）           |

---

## 🎯 **修正版戦略: 実データセット使用の本格検証**

### **Phase 1: llama4_lisa.py 完全実装** (2-3 時間)

#### ステップ 1.1: 公式仕様準拠の完全実装

**修正対象:** `model/llama4_lisa.py`

**実装内容:**

```python
class Llama4LisaForConditionalGeneration(Llama4ForConditionalGeneration):
    def __init__(self, config):
        # 1. 公式仕様でLlama4初期化
        super().__init__(config)

        # 2. 公式推奨設定適用
        self.config.attn_implementation = "flex_attention"
        self.config.torch_dtype = torch.bfloat16

        # 3. SAM統合 (Original-LISA/Gemma-LISA方式)
        self.visual_model = build_sam_vit_h(config.sam_checkpoint_path)

        # 4. プロジェクタ実装 (Gemma-LISA準拠)
        self.mlp_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.out_dim),
        )

        # 5. SEGトークン統合
        self._setup_seg_token()

    def forward(self, **kwargs):
        # Original-LISA/Gemma-LISAのforward実装パターン準拠
        # 実際のデュアルストリーム処理
```

#### ステップ 1.2: ファクトリ関数の完全実装

```python
def create_llama4_lisa_model(**kwargs):
    # 公式仕様準拠でLlama4+SAMを統合
    model = Llama4ForConditionalGeneration.from_pretrained(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        attn_implementation="flex_attention",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    # SAM統合処理...
```

### **Phase 2: 実データセット検証スクリプト作成** (2-3 時間)

#### ステップ 2.1: 実データセット整合性検証

**新規作成:** `verify_llama4_dataset_integrity.py`

**Gemma-LISA ベース実装:**

```python
# 実際のHybridDatasetを使用
from utils.dataset import HybridDataset
dataset = HybridDataset(datasets=["sem_seg", "refer_seg", "vqa"], ...)

# 実サンプルで検証
for i in range(10):
    sample = dataset[i]
    # 画像・マスク・テキストの整合性確認
    # Lambda Cloud上の実データ使用
```

#### ステップ 2.2: 実入力フォーマット検証

**新規作成:** `verify_llama4_input_formatting.py`

**Gemma-LISA ベース実装:**

```python
# 実際のprocessorで25サンプル処理
processor = AutoProcessor.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")
for sample in dataset:
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_tensors="pt"
    )
    # トークン化・ラベルマスキング検証
```

#### ステップ 2.3: 実モデルアーキテクチャ検証

**新規作成:** `verify_llama4_architecture_real.py`

**完全モデル初期化:**

```python
# 実際のLlama4+SAM統合
model = create_llama4_lisa_model()
# 実メモリ使用量測定
# 実コンポーネント検証
```

#### ステップ 2.4: 実損失・勾配検証

**新規作成:** `verify_llama4_loss_gradients_real.py`

**実フォワードパス:**

```python
# HybridDatasetから実バッチ作成
dataloader = DataLoader(dataset, batch_size=1)
batch = next(iter(dataloader))

# 実際のフォワードパス
outputs = model(**batch)
loss = outputs.loss

# 実際の勾配伝播
loss.backward()
# 全パラメータの勾配検証
```

#### ステップ 2.5: 実バッチ過学習テスト

**新規作成:** `overfit_llama4_real_batch.py`

**実データ過学習:**

```python
# HybridDatasetから実バッチ
real_batch = next(iter(dataloader))

# 20イテレーション実過学習
for epoch in range(20):
    outputs = model(**real_batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    # 実損失減少確認
```

### **Phase 3: 完全統合検証** (1-2 時間)

#### ステップ 3.1: エンドツーエンド実検証

**新規作成:** `verify_llama4_end_to_end_real.py`

**完全パイプライン:**

```python
# 1. 実データセット → HybridDataset
# 2. 実プロセッサ → Llama4 processor
# 3. 実モデル → Llama4LisaForConditionalGeneration
# 4. 実フォワードパス → デュアルストリーム処理
# 5. 実損失計算 → CE + Dice + BCE
```

### **Phase 4: フル学習実行** (4-8 時間)

#### ステップ 4.1: 実環境での学習開始

```bash
# 修正版での本格学習
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.95.127 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && tmux new-session -d -s lisa_llama4_real_training 'python train_llama4.py'"
```

---

## 🚀 **緊急実行計画**

### **即座に実行すべき作業 (優先度: 最高)**

1. **llama4_lisa.py 完全書き換え** (2 時間)
2. **実データセット検証スクリプト 5 本作成** (3 時間)
3. **実検証実行** (1 時間)

### **修正版実装チェックリスト**

#### Phase 1: 完全実装 🔄 **修正中**

- [ ] `llama4_lisa.py` 公式仕様準拠書き換え
- [ ] Gemma-LISA/Original-LISA パターン採用
- [ ] ファクトリ関数完全実装
- [ ] 設定クラス完全修正

#### Phase 2: 実データセット検証 ❌ **未着手**

- [ ] `verify_llama4_dataset_integrity.py` 作成
- [ ] `verify_llama4_input_formatting.py` 作成
- [ ] `verify_llama4_architecture_real.py` 作成
- [ ] `verify_llama4_loss_gradients_real.py` 作成
- [ ] `overfit_llama4_real_batch.py` 作成

#### Phase 3: 完全検証 ❌ **未着手**

- [ ] `verify_llama4_end_to_end_real.py` 作成・実行
- [ ] 全検証スクリプト PASS 確認

#### Phase 4: 実学習 ❌ **未実行**

- [ ] 修正版での学習開始
- [ ] 実損失減少確認

---

## 🔄 **戦略変更の根拠**

### ❌ **現行アプローチの根本的問題**

1. **表面的検証**: ダミーデータでは実際の問題を検出不可
2. **統合不足**: llama4_lisa.py が実際には機能しない可能性
3. **仕様乖離**: Web 調査で判明した公式仕様との差異

### ✅ **修正アプローチの利点**

1. **実データ検証**: HybridDataset で実際の処理確認
2. **完全統合**: Gemma-LISA/Original-LISA の実装パターン踏襲
3. **公式準拠**: Web 調査で確認した公式仕様に完全準拠

---

## 📅 **修正版タイムライン**

- **Phase 1:** 2-3 時間（完全実装）
- **Phase 2:** 2-3 時間（実データセット検証）
- **Phase 3:** 1-2 時間（完全統合検証）
- **Phase 4:** 4-8 時間（実学習）

**合計推定時間:** 9-16 時間

---

## 🎯 **修正版成功判定基準**

### 即座の成功基準

- [ ] llama4_lisa.py が実際の Llama4+SAM 統合を実現
- [ ] 実データセット検証で全項目 PASS
- [ ] 実バッチ過学習で損失減少確認

### 最終成功基準

- [ ] エンドツーエンド実検証 PASS
- [ ] 実環境での学習正常開始
- [ ] 実損失減少と WandB ログ確認

---

**戦略更新理由:** 現在の検証はダミーデータのみで、実際の llama4_lisa 統合検証になっていない  
**次のアクション:** llama4_lisa.py の完全実装と Gemma-LISA 準拠の実データセット検証スクリプト作成
