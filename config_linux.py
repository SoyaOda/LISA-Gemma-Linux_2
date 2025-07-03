# config_linux.py
import os
from pathlib import Path

# ==============================================================================
# 1. パスおよびモデル識別子設定 (Lambda Cloud環境)
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent

# データセットベースディレクトリ (Lambda Cloud の NFSパスなど)
DATASET_BASE_DIR = os.environ.get("LISA_DATASET_BASE_DIR", "/lambda/nfs/lisa-gemma-project-fs/data/dataset")
# SAMチェックポイントのパス
SAM_CHECKPOINT_PATH = os.environ.get("LISA_SAM_CHECKPOINT_PATH", "/lambda/nfs/lisa-gemma-project-fs/data/weights/sam_vit_h_4b8939.pth")

# Hugging Faceキャッシュディレクトリ（オプション環境変数）
HF_CACHE_DIR = os.environ.get('HF_HOME', None)

# ==============================================================================
# 2. モデル識別子およびモデル設定
# ==============================================================================
# 使用するLlama4モデル
LLAMA_MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# Llama-4-Scout-17B-16E-Instruct特有の設定（Webリサーチ準拠）
# 注意: flex_attentionにバグがあるため、現在はeagerが推奨（2025年実装状況）
ATTN_IMPLEMENTATION = "eager"           # flex_attentionバグ回避のため
DEVICE_MAP = "auto"                     # meta tensor対策  
TORCH_DTYPE = "bfloat16"               # 推奨精度

# モデルサイズ/バリエーションに応じた設定
LLAMA_MODEL_CONFIGS = {
    "scout": {
        "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "hidden_size": 5120,
        "recommended_batch_size": 1,  # Scoutは17B (MoE全体109B) -> 高メモリ消費
        "memory_gb_estimate": 80.0   # BF16での推定（単GPUに載せるには4bit量子化推奨）
    },
    "maverick": {
        "model_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "hidden_size": 5120,
        "recommended_batch_size": 1,
        "memory_gb_estimate": 160.0  # Maverick (402B total)はFP8で複数GPU前提
    }
}

# ログ・出力ディレクトリ
LOG_BASE_DIR = str(PROJECT_ROOT / "runs")
WEIGHTS_DIR = str(PROJECT_ROOT / "weights")

# ==============================================================================
# 3. モデルハイパーパラメータ
# ==============================================================================
# 画像サイズ設定
LLAMA_IMAGE_SIZE = 448   # Llama4 Visionが扱うタイルサイズ (px)
SAM_IMAGE_SIZE = 1024    # SAMエンコーダー入力サイズ (px)
# モデルの最大トークン長 (Llama4は128kまで可能)
MODEL_MAX_LENGTH = 131072
# MLPプロジェクタ出力次元 (SAM prompt embedと一致)
SEG_PROJECTION_DIM = 256
# セグメンテーション特別トークン
SEG_TOKEN = "[SEG]"
# Llama4 Scoutのテキスト隠れ層サイズ
LLAMA_HIDDEN_SIZE = 5120

# ==============================================================================
# 4. トレーニング設定
# ==============================================================================
LEARNING_RATE = 1e-4
EPOCHS = 10
STEPS_PER_EPOCH = 500
WEIGHT_DECAY = 1e-2
BETA1 = 0.9
BETA2 = 0.95

# バッチサイズと勾配蓄積
BATCH_SIZE_PER_GPU = 1  # Scoutモデルは大きいため、GPUあたり1に設定
GRADIENT_ACCUMULATION_STEPS = 8  # 実質バッチサイズ = BATCH_SIZE_PER_GPU * accumulation * GPU数

# システム・最適化設定
MIXED_PRECISION = True    # bf16混合精度
GRADIENT_CHECKPOINTING = True  # メモリ節約のための勾配チェックポイント
DATALOADER_NUM_WORKERS = 4

# ==============================================================================
# 5. LoRAファインチューニング設定
# ==============================================================================
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# ターゲットモジュール: Llama4のAttentionとFFNプロジェクション層
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# ==============================================================================
# 6. データセット設定
# ==============================================================================
# データセット混合比率 (例: sem_seg:refer_seg:vqa:reason_seg = 9:3:3:1)
DATASET_SAMPLE_RATES = "9,3,3,1"
# 1エポックあたり各データセットから使用するサンプル数 (デバッグ用に小さめ設定)
SAMPLES_PER_EPOCH = 500
# 推論（generate）時の最大新出トークン数
MAX_NEW_TOKENS = 100

# データセット種別ごとの利用データ
SEM_SEG_DATA = "ade20k||cocostuff||mapillary||pascal_part||paco_lvis"
REFER_SEG_DATA = "refclef||refcoco||refcoco+||refcocog"
VQA_DATA = "llava_instruct_150k"
REASON_SEG_DATA = "ReasonSeg|train"
VAL_DATASET = "ReasonSeg|val"

# ==============================================================================
# 7. データセットディレクトリ構造 (データ存在チェック用)
# ==============================================================================
DATASET_STRUCTURE = {
    "sem_seg": {
        "ade20k": {
            "path": "ade20k", "images": "images", "annotations": "annotations",
            "required_files": ["images", "annotations"]
        },
        "cocostuff": {
            "path": "cocostuff", "images": "train2017", "annotations": "train2017",
            "required_files": ["train2017"]
        },
        "mapillary": {
            "path": "mapillary", "images": "training/images", "annotations": "training/labels",
            "config": "config_v2.0.json", "required_files": ["training", "config_v2.0.json"]
        },
        "pascal_part": {
            "path": "vlpart/pascal_part", "json_file": "train.json",
            "images": "VOCdevkit/VOC2010/JPEGImages", "required_files": ["train.json", "VOCdevkit"]
        },
        "paco_lvis": {
            "path": "vlpart/paco", "annotations": "annotations",
            "required_files": ["annotations"]
        }
    },
    "refer_seg": {
        "base_path": "refer_seg",
        "datasets": {
            "refcoco": {"annotations": "refcoco", "images": "images/mscoco/images/train2014"},
            "refcoco+": {"annotations": "refcoco+", "images": "images/mscoco/images/train2014"},
            "refcocog": {"annotations": "refcocog", "images": "images/mscoco/images/train2014"},
            "refclef": {"annotations": "refclef", "images": "images/saiapr_tc-12"}
        },
        "required_files": ["refcoco", "refcoco+", "refcocog", "refclef", "images"]
    },
    "vqa": {
        "llava_instruct_150k": {
            "path": "llava_dataset", "json_file": "llava_instruct_150k.json",
            "required_files": ["llava_instruct_150k.json"]
        }
    },
    "reason_seg": {
        "ReasonSeg": {
            "path": "reason_seg/ReasonSeg", "train": "train", "val": "val", "explanatory": "explanatory",
            "required_files": ["train", "val", "explanatory"]
        }
    },
    "vlpart": {
        "paco": {"path": "vlpart/paco", "annotations": "annotations", "required_files": ["annotations"]},
        "pascal_part": {"path": "vlpart/pascal_part", "annotations": "train.json", "images": "VOCdevkit",
                        "required_files": ["train.json", "VOCdevkit"]}
    }
}

# ==============================================================================
# 8. ユーティリティ関数
# ==============================================================================
def get_model_config(key: str = "scout"):
    """モデルバリエーションに応じた設定取得"""
    if key not in LLAMA_MODEL_CONFIGS:
        raise ValueError(f"未サポートのモデル種別: {key}. 選択可能: {list(LLAMA_MODEL_CONFIGS.keys())}")
    return LLAMA_MODEL_CONFIGS[key]

def get_dataset_paths() -> dict:
    """データセットパスのディクショナリ作成"""
    base = Path(DATASET_BASE_DIR)
    paths = {"sem_seg": {}, "refer_seg": {}, "vqa": {}, "reason_seg": {}}
    # Semantic Segmentation
    for name, cfg in DATASET_STRUCTURE["sem_seg"].items():
        paths["sem_seg"][name] = str(base / cfg["path"])
    # Referring Segmentation
    refer_base = base / DATASET_STRUCTURE["refer_seg"]["base_path"]
    for name, cfg in DATASET_STRUCTURE["refer_seg"]["datasets"].items():
        paths["refer_seg"][name] = str(refer_base)
    # VQA
    for name, cfg in DATASET_STRUCTURE["vqa"].items():
        paths["vqa"][name] = str(base / cfg["path"] / cfg["json_file"])
    # Reasoning Segmentation
    for name, cfg in DATASET_STRUCTURE["reason_seg"].items():
        paths["reason_seg"][name] = str(base / cfg["path"])
    return paths

def get_required_weights() -> dict:
    """必要な重みファイル情報"""
    return {
        "sam_vit_h": {
            "path": SAM_CHECKPOINT_PATH,
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "size_gb": 2.39,
            "description": "SAM (ViT-H) モデルチェックポイント"
        }
    }

def check_dataset_structure(dataset_name: str, dataset_type: str) -> tuple[bool, list]:
    """指定データセットの構造チェック"""
    base = Path(DATASET_BASE_DIR)
    missing = []
    if dataset_type not in DATASET_STRUCTURE or dataset_name not in DATASET_STRUCTURE[dataset_type]:
        return False, [f"不明なデータセット: {dataset_type}/{dataset_name}"]
    cfg = DATASET_STRUCTURE[dataset_type][dataset_name]
    dataset_path = base / cfg.get("path", "")
    if not dataset_path.exists():
        missing.append(f"データセットディレクトリ未発見: {dataset_path}")
        return False, missing
    for req in cfg.get("required_files", []):
        item_path = dataset_path / req
        if not item_path.exists():
            missing.append(f"必要ファイル/フォルダ未発見: {item_path}")
    return (len(missing) == 0), missing

def check_all_paths() -> bool:
    """重要パスとファイルの存在チェック"""
    errors = []
    base = Path(DATASET_BASE_DIR)
    if not base.exists():
        errors.append(f"データセットベースディレクトリが存在しません: {base}")
    sam_path = Path(SAM_CHECKPOINT_PATH)
    if not sam_path.exists():
        errors.append(f"SAMチェックポイントが存在しません: {sam_path}")
        info = get_required_weights()["sam_vit_h"]
        errors.append(f"ダウンロードコマンド例: wget {info['url']} -O {sam_path}")
    critical = [("ReasonSeg", "reason_seg"), ("ade20k", "sem_seg"), ("llava_instruct_150k", "vqa")]
    for name, dtype in critical:
        ok, missing = check_dataset_structure(name, dtype)
        if not ok:
            for m in missing:
                errors.append(f"{name}: {m}")
    if errors:
        print("❌ 設定エラー:")
        for err in errors:
            print(f"  - {err}")
        print("\n対応策:")
        print("1. 必要に応じ環境変数 LISA_DATASET_BASE_DIR, LISA_SAM_CHECKPOINT_PATH を設定してパスを修正")
        print("2. あるいは、本ファイル内のパスを直接編集")
        print("3. データセットおよびSAM重みファイルをダウンロード・配置してください")
        raise FileNotFoundError("必須リソースが不足しています。上記対応策を実施してください。")
    return True

def print_dataset_info():
    """データセット情報の表示"""
    print("=== データセットパス情報 ===")
    print(f"Base Dir: {DATASET_BASE_DIR}")
    print(f"SAM Checkpoint: {SAM_CHECKPOINT_PATH}")
    paths = get_dataset_paths()
    for dtype, datasets in paths.items():
        print(f"\n[{dtype.upper()}]")
        for name, path in datasets.items():
            status = "✓" if Path(path).exists() else "✗"
            print(f"  {status} {name}: {path}")

if __name__ == "__main__":
    print("=== LISA-Llama4 設定検証 ===")
    try:
        if check_all_paths():
            print("✅ パス/ファイル設定OK")
            print_dataset_info()
    except FileNotFoundError as e:
        print(f"❌ エラー: {e}")
        exit(1)