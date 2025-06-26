# Lambda Cloud Environment Configuration
import os
from pathlib import Path

# ==================== Lambda Cloud Specific Paths ====================
# Persistent Filesystem root
FILESYSTEM_ROOT = "/lambda/nfs/lisa-gemma-project-fs"

# Data paths (Lambda Cloud specific)
DATA_ROOT = f"{FILESYSTEM_ROOT}/data"
DATASET_ROOT = f"{DATA_ROOT}/dataset"

# Code and project paths
CODE_ROOT = f"{FILESYSTEM_ROOT}/code/LISA-Gemma-Linux"
PROJECT_ROOT = CODE_ROOT

# Virtual environment
VENV_ROOT = f"{FILESYSTEM_ROOT}/venvs"
VENV_NAME = "lisa_gemma_venv"
VENV_PATH = f"{VENV_ROOT}/{VENV_NAME}"

# Output and artifacts paths
ARTIFACTS_ROOT = f"{FILESYSTEM_ROOT}/artifacts"
CHECKPOINT_ROOT = f"{ARTIFACTS_ROOT}/checkpoints"
LOGS_ROOT = f"{ARTIFACTS_ROOT}/logs"
FINAL_MODELS_ROOT = f"{ARTIFACTS_ROOT}/final_models"

# ==================== Dataset Paths ====================
# Individual dataset paths
REFER_SEG_PATH = f"{DATASET_ROOT}/refer_seg"
REASON_SEG_PATH = f"{DATASET_ROOT}/reason_seg"
LLAVA_DATASET_PATH = f"{DATASET_ROOT}/llava_dataset"
COCO_PATH = f"{DATASET_ROOT}/coco"
COCOSTUFF_PATH = f"{DATASET_ROOT}/cocostuff"
ADE20K_PATH = f"{DATASET_ROOT}/ade20k"
MAPILLARY_PATH = f"{DATASET_ROOT}/mapillary"
VLPART_PATH = f"{DATASET_ROOT}/vlpart"

# SAM weights path
SAM_WEIGHTS_PATH = f"{DATA_ROOT}/weights/sam_vit_h_4b8939.pth"

# ==================== Lambda Cloud Specific Settings ====================
# GPU settings
GPU_COUNT = 1  # Will be overridden based on instance type
GPU_TYPE = "A10"  # Will be detected automatically

# Training settings optimized for Lambda Cloud
TRAINING_BATCH_SIZE = 4  # Conservative for A10
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
MAX_EPOCHS = 3
LEARNING_RATE = 1e-4

# Model settings
MODEL_NAME = "google/gemma-2-9b-it"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# ==================== Environment Detection ====================
def detect_environment():
    """Detect the current Lambda Cloud environment"""
    import subprocess
    
    # Detect GPU type and count
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_names = result.stdout.strip().split('\n')
            GPU_COUNT = len(gpu_names)
            GPU_TYPE = gpu_names[0].split()[-1] if gpu_names else "Unknown"
            return {
                'gpu_count': GPU_COUNT,
                'gpu_type': GPU_TYPE,
                'gpu_names': gpu_names
            }
    except:
        pass
    
    return {
        'gpu_count': 1,
        'gpu_type': 'Unknown',
        'gpu_names': ['Unknown']
    }

# ==================== Path Validation ====================
def validate_paths():
    """Validate that all required paths exist"""
    required_paths = [
        FILESYSTEM_ROOT,
        DATA_ROOT,
        DATASET_ROOT,
        CODE_ROOT,
        VENV_PATH,
        SAM_WEIGHTS_PATH
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print("❌ Missing required paths:")
        for path in missing_paths:
            print(f"  - {path}")
        return False
    
    print("✅ All required paths exist")
    return True

# ==================== Utility Functions ====================
def get_experiment_name(prefix="exp"):
    """Generate unique experiment name with timestamp"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"

def get_output_dir(experiment_name=None):
    """Get output directory for current experiment"""
    if experiment_name is None:
        experiment_name = get_experiment_name()
    return f"{CHECKPOINT_ROOT}/{experiment_name}"

def get_log_dir(experiment_name=None):
    """Get log directory for current experiment"""
    if experiment_name is None:
        experiment_name = get_experiment_name()
    return f"{LOGS_ROOT}/{experiment_name}"

# ==================== SSH Command Helpers ====================
def generate_ssh_command(command, use_tmux=False, session_name=None):
    """Generate SSH command for remote execution"""
    ssh_base = f"ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58"
    
    if use_tmux:
        session_name = session_name or "lisa_training"
        # Create tmux session if it doesn't exist, or attach to existing
        tmux_cmd = f"tmux new-session -d -s {session_name} '{command}' || tmux send-keys -t {session_name} '{command}' Enter"
        return f"{ssh_base} \"{tmux_cmd}\""
    else:
        return f"{ssh_base} \"{command}\""

# ==================== Development Workflow Helpers ====================
def sync_code_to_lambda():
    """Generate rsync command to sync code to Lambda"""
    return f"""rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \\
  ./ ubuntu@150.136.47.58:{CODE_ROOT}/"""

def sync_results_from_lambda():
    """Generate rsync command to sync results from Lambda"""
    return f"""rsync -avz ubuntu@150.136.47.58:{ARTIFACTS_ROOT}/ ./lambda_results/"""

if __name__ == "__main__":
    print("🚀 Lambda Cloud Configuration")
    print("=" * 50)
    
    # Environment detection
    env_info = detect_environment()
    print(f"GPU Count: {env_info['gpu_count']}")
    print(f"GPU Type: {env_info['gpu_type']}")
    
    # Path validation
    print("\n📁 Path Validation:")
    validate_paths()
    
    print(f"\n📂 Key Paths:")
    print(f"  Code Root: {CODE_ROOT}")
    print(f"  Data Root: {DATA_ROOT}")
    print(f"  Checkpoints: {CHECKPOINT_ROOT}")
    print(f"  Logs: {LOGS_ROOT}") 