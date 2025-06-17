import os

# ==============================================================================
# 1. PATHS AND IDENTIFIERS
# ==============================================================================
# プロジェクトのルートディレクトリからの相対パスでデータセットのベースディレクトリを指定
# 例: /home/user/LISA-Gemma3/datasets
# 注意: WSLの '/mnt/h/...' のようなパスではなく、Linuxネイティブの絶対パスまたは相対パスを使用すること
DATASET_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

# 事前学習済みSAMモデルのチェックポイントへのパス
# 例: /home/user/LISA-Gemma3/weights/sam_vit_h_4b8939.pth
SAM_CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "sam_vit_h_4b8939.pth")

# Hugging Faceモデル識別子
GEMMA_MODEL_ID = "google/gemma-2-2b-it"

# ログと出力の保存先
LOG_BASE_DIR = "./runs"

# ==============================================================================
# 2. MODEL CONFIGURATION
# ==============================================================================
# 画像サイズ設定
# GEMMA_IMAGE_SIZEはGemma-2のSigLIPエンコーダの要求仕様 (896x896)
GEMMA_IMAGE_SIZE = 896
# SAM_IMAGE_SIZEはSAM-ViTエンコーダの要求仕様 (1024x1024)
SAM_IMAGE_SIZE = 1024
# モデルが処理するトークンの最大長
MODEL_MAX_LENGTH = 2048
# MLPプロジェクタからSAMデコーダへの出力次元 (SAMのプロンプト埋め込み次元と一致)
SEG_PROJECTION_DIM = 256

# ==============================================================================
# 3. TRAINING HYPERPARAMETERS
# ==============================================================================
# DeepSpeed設定ファイルで "auto" を使用するため、ここではコメントアウト。
# TrainingArgumentsまたはdeepspeed configで直接設定することを推奨。
# BATCH_SIZE_PER_GPU = 2
# GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
EPOCHS = 10
STEPS_PER_EPOCH = 500
WEIGHT_DECAY = 1e-2
BETA1 = 0.9
BETA2 = 0.95

# 損失関数の重み
CE_LOSS_WEIGHT = 1.0
DICE_LOSS_WEIGHT = 0.5
BCE_LOSS_WEIGHT = 2.0

# ==============================================================================
# 4. LoRA CONFIGURATION (Parameter-Efficient Fine-Tuning)
# ==============================================================================
LORA_R = 32  # ランク
LORA_ALPHA = 64  # LoRAのスケーリング係数 (r * 2 が一般的)
LORA_DROPOUT = 0.05

# CRITICAL: Gemmaアーキテクチャに適合したターゲットモジュール
# ユーザー提供のスクリプトにあった "q_proj,k_proj,v_proj" は不完全。
# GemmaではAttention層とFFN層の両方をターゲットにする必要がある。
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
# 5. DATASET CONFIGURATION
# ==============================================================================
# データセットの混合比率
DATASET_SAMPLE_RATES = "9,3,3,1"  # sem_seg, refer_seg, vqa, reason_seg
# 使用するデータセットの指定
SEM_SEG_DATA = "ade20k||cocostuff||mapillary"
REFER_SEG_DATA = "refclef||refcoco||refcoco+||refcocog"
VQA_DATA = "llava_instruct_150k"
REASON_SEG_DATA = "ReasonSeg|train"
VAL_DATASET = "ReasonSeg|val"

# ==============================================================================
# 6. UTILITY FUNCTIONS
# ==============================================================================
def get_dataset_paths():
    """データセットへの完全なパスを構築して返す"""
    return {
        "sem_seg": {
            "ade20k": os.path.join(DATASET_BASE_DIR, "ade20k"),
            "cocostuff": os.path.join(DATASET_BASE_DIR, "cocostuff"),
            "mapillary": os.path.join(DATASET_BASE_DIR, "mapillary"),
        },
        "refer_seg": {
            "refcoco": os.path.join(DATASET_BASE_DIR, "refer_seg"),
            "refcoco+": os.path.join(DATASET_BASE_DIR, "refer_seg"),
            "refcocog": os.path.join(DATASET_BASE_DIR, "refer_seg"),
            "refclef": os.path.join(DATASET_BASE_DIR, "refer_seg"),
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
    paths_to_check = [SAM_CHECKPOINT_PATH]
    
    dataset_paths = get_dataset_paths()
    # 簡単な存在チェック
    paths_to_check.append(dataset_paths["sem_seg"]["ade20k"])
    paths_to_check.append(dataset_paths["vqa"]["llava_instruct_150k"])

    missing_paths = [path for path in paths_to_check if not os.path.exists(path)]
    
    if missing_paths:
        print("警告: 以下の必須パスまたはファイルが見つかりません:")
        for path in missing_paths:
            print(f"  - {path}")
        print("\nconfig_linux.pyのパス設定と、データセットが正しく配置されているか確認してください。")
        return False
    
    return True

if __name__ == "__main__":
    print("=== プロジェクト設定検証 ===")
    if check_paths():
        print("✓ 必須パスの検証に成功しました。")
    else:
        print("✗ 必須パスの検証に失敗しました。上記のエラーメッセージを確認してください。") 