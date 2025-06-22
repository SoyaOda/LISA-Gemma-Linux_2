import os

# ==============================================================================
# 小規模テスト用設定ファイル
# 本格学習前の動作確認用
# ==============================================================================

# ==============================================================================
# 1. PATHS AND IDENTIFIERS（config_linux.pyと同じ）
# ==============================================================================
# データセットのベースディレクトリ
DATASET_BASE_DIR = "/mnt/h/download/LISA-dataset/dataset"

# 事前学習済みSAMモデルのチェックポイントへのパス
SAM_CHECKPOINT_PATH = "/mnt/c/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth"

# Hugging Faceモデル識別子
GEMMA_MODEL_ID = "google/gemma-3-4b-it"

# ログと出力の保存先
LOG_BASE_DIR = "./runs"

# ==============================================================================
# 2. MODEL CONFIGURATION（config_linux.pyと同じ）
# ==============================================================================
# 画像サイズ設定
GEMMA_IMAGE_SIZE = 896
SAM_IMAGE_SIZE = 1024
MODEL_MAX_LENGTH = 512  # より短く設定してメモリ使用量を削減
SEG_PROJECTION_DIM = 256
SEG_TOKEN = "[SEG]"
# Gemma-3-4bの隠れ層サイズ
GEMMA_HIDDEN_SIZE = 2560

# ==============================================================================
# 3. TRAINING HYPERPARAMETERS（小規模テスト用に調整）
# ==============================================================================
# 小規模テスト用の設定
LEARNING_RATE = 2e-4  # 少し高めの学習率で早期収束確認
EPOCHS = 2  # 短期テスト用
STEPS_PER_EPOCH = 20  # 大幅削減（通常500 → 20）
WEIGHT_DECAY = 1e-2
BETA1 = 0.9
BETA2 = 0.95

# 損失関数の重み
CE_LOSS_WEIGHT = 1.0
DICE_LOSS_WEIGHT = 0.5
BCE_LOSS_WEIGHT = 2.0

# ==============================================================================
# 4. LoRA CONFIGURATION（config_linux.pyと同じ）
# ==============================================================================
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# ==============================================================================
# 5. DATASET CONFIGURATION（小規模テスト用に制限）
# ==============================================================================
# より小さなサンプル数（通常は thousands → 40サンプル）
SMALL_TEST_SAMPLES_PER_EPOCH = 40  # 小規模テスト用

# エイリアス（後方互換性のため）
SMALL_TEST_EPOCHS = EPOCHS  # 小規模テスト用エポック数
SMALL_TEST_STEPS_PER_EPOCH = STEPS_PER_EPOCH  # 小規模テスト用ステップ/エポック

# データセットの混合比率（単純化）
DATASET_SAMPLE_RATES = "2,1,1,1"  # より均等な分布でテスト

# 使用するデータセットを制限（最も安定したもののみ）
SEM_SEG_DATA = "ade20k"  # 1つのデータセットのみ
REFER_SEG_DATA = "refcoco"  # 1つのデータセットのみ
VQA_DATA = "llava_instruct_150k"  # そのまま
REASON_SEG_DATA = "ReasonSeg|train"  # そのまま（最小）
VAL_DATASET = "ReasonSeg|val"

# ==============================================================================
# 6. UTILITY FUNCTIONS
# ==============================================================================
def get_dataset_paths():
    """データセットへの完全なパスを構築して返す"""
    return {
        "sem_seg": {
            "ade20k": os.path.join(DATASET_BASE_DIR, "ade20k"),
        },
        "refer_seg": {
            "refcoco": os.path.join(DATASET_BASE_DIR, "refer_seg"),
        },
        "vqa": {
            "llava_instruct_150k": os.path.join(DATASET_BASE_DIR, "llava_dataset", "llava_instruct_150k.json"),
        },
        "reason_seg": {
            "ReasonSeg": os.path.join(DATASET_BASE_DIR, "reason_seg", "ReasonSeg"),
        }
    }

def check_paths():
    """重要なパスが存在するかを検証する"""
    paths_to_check = [
        ("データセットベースディレクトリ", DATASET_BASE_DIR),
        ("SAMチェックポイント", SAM_CHECKPOINT_PATH),
    ]
    
    dataset_paths = get_dataset_paths()
    paths_to_check.extend([
        ("ADE20k データセット", dataset_paths["sem_seg"]["ade20k"]),
        ("ReasonSeg データセット", dataset_paths["reason_seg"]["ReasonSeg"]),
        ("LLaVA VQA データセット", dataset_paths["vqa"]["llava_instruct_150k"]),
        ("RefCOCO データセット", dataset_paths["refer_seg"]["refcoco"]),
    ])

    missing_paths = []
    for name, path in paths_to_check:
        if not os.path.exists(path):
            missing_paths.append((name, path))
    
    if missing_paths:
        print("エラー: 以下の必須パスまたはファイルが見つかりません:")
        for name, path in missing_paths:
            print(f"  - {name}: {path}")
        print("\nデータセットは必須です。プログラムを終了します。")
        raise FileNotFoundError(f"必須ファイルが見つかりません。設定を確認してください。")
    
    return True

if __name__ == "__main__":
    print("=== 小規模テスト設定検証 ===")
    print(f"サンプル数: {SMALL_TEST_SAMPLES_PER_EPOCH}")
    print(f"エポック数: {EPOCHS}")
    print(f"ステップ/エポック: {STEPS_PER_EPOCH}")
    print(f"総ステップ数: {EPOCHS * STEPS_PER_EPOCH}")
    if check_paths():
        print("✓ 必須パスの検証に成功しました。")
    else:
        print("✗ 必須パスの検証に失敗しました。") 