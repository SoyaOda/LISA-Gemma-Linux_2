# ==============================================================================
# Lambda Cloud Configuration for LISA-Gemma Project
# ==============================================================================

# ==============================================================================
# 1. PATHS AND IDENTIFIERS (Lambda Cloud Environment)
# ==============================================================================
# Lambda Cloud Persistent Filesystem ベースパス
FILESYSTEM_BASE = "/lambda/nfs/lisa-gemma-project-fs"

# データセットのベースディレクトリ (移行後)
DATASET_BASE_DIR = f"{FILESYSTEM_BASE}/data/dataset"

# 事前学習済みSAMモデルのチェックポイントへのパス (移行後)
SAM_CHECKPOINT_PATH = f"{FILESYSTEM_BASE}/data/weights/sam_vit_h_4b8939.pth"

# プロジェクトコードのベースディレクトリ
PROJECT_BASE_DIR = f"{FILESYSTEM_BASE}/code/LISA-Gemma-Linux"

# アーティファクト保存用ディレクトリ
ARTIFACTS_BASE_DIR = f"{FILESYSTEM_BASE}/artifacts"
CHECKPOINTS_DIR = f"{ARTIFACTS_BASE_DIR}/checkpoints"
LOGS_DIR = f"{ARTIFACTS_BASE_DIR}/logs"
FINAL_MODELS_DIR = f"{ARTIFACTS_BASE_DIR}/final_models"

# Python仮想環境
VENV_PATH = f"{FILESYSTEM_BASE}/venvs/lisa_gemma_venv"

# ==============================================================================
# 2. MODEL CONFIGURATION
# ==============================================================================
# Gemma モデル設定
GEMMA_MODEL_NAME = "google/gemma-2-7b-it"
GEMMA_LOCAL_CACHE_DIR = f"{FILESYSTEM_BASE}/model_cache/gemma"

# SAM モデル設定
SAM_MODEL_TYPE = "vit_h"  # vit_h, vit_l, vit_b
SAM_LOCAL_CACHE_DIR = f"{FILESYSTEM_BASE}/model_cache/sam"

# ==============================================================================
# 3. TRAINING CONFIGURATION
# ==============================================================================
# デフォルト学習設定
DEFAULT_OUTPUT_DIR = f"{CHECKPOINTS_DIR}/default_run"
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 5

# LoRA設定
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1

# ==============================================================================
# 4. HARDWARE CONFIGURATION  
# ==============================================================================
# GPU設定（Lambda Cloud環境に応じて自動調整）
import torch
if torch.cuda.is_available():
    NUM_GPUS = torch.cuda.device_count()
    DEVICE_MAP = "auto"
    TORCH_DTYPE = torch.bfloat16  # H100/A100 最適化
else:
    NUM_GPUS = 0
    DEVICE_MAP = "cpu"
    TORCH_DTYPE = torch.float32

# ==============================================================================
# 5. DEEPSPEED CONFIGURATION
# ==============================================================================
# DeepSpeed設定ファイルパス
DEEPSPEED_CONFIG_LOCAL = f"{PROJECT_BASE_DIR}/ds_config_local.json"
DEEPSPEED_CONFIG_CLOUD = f"{PROJECT_BASE_DIR}/ds_config_cloud.json"
DEEPSPEED_CONFIG_ADVANCED = f"{PROJECT_BASE_DIR}/ds_config_advanced.json"

# ==============================================================================
# 6. HUGGING FACE CONFIGURATION
# ==============================================================================
# Hugging Face設定
import os
HF_TOKEN = os.getenv("HF_TOKEN", "your_token_here")  # 環境変数から取得
HF_CACHE_DIR = f"{FILESYSTEM_BASE}/hf_cache"

# ==============================================================================
# 7. LOGGING AND MONITORING
# ==============================================================================
# ログ設定
TENSORBOARD_LOG_DIR = f"{LOGS_DIR}/tensorboard"
CSV_LOG_PATH = f"{LOGS_DIR}/training_log.csv"
HPO_LOG_PATH = f"{LOGS_DIR}/hpo_log.csv"

# ==============================================================================
# 8. UTILITY FUNCTIONS
# ==============================================================================
def get_output_dir(experiment_name="default"):
    """実験名に基づいた出力ディレクトリを生成"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{CHECKPOINTS_DIR}/{experiment_name}_{timestamp}"

def get_config_summary():
    """設定の概要を表示"""
    return f"""
=== Lambda Cloud LISA-Gemma Configuration ===
Dataset: {DATASET_BASE_DIR}
SAM Weights: {SAM_CHECKPOINT_PATH}
Project: {PROJECT_BASE_DIR}
Artifacts: {ARTIFACTS_BASE_DIR}
Virtual Env: {VENV_PATH}
GPUs: {NUM_GPUS}
Device Map: {DEVICE_MAP}
Torch Dtype: {TORCH_DTYPE}
===============================================
"""

if __name__ == "__main__":
    print(get_config_summary()) 