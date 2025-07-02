"""
Llama4-LISA Complete Training Configuration
Based on official Llama-4-Scout-17B-16E-Instruct specifications
Optimized for Lambda Cloud A100 environment
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

@dataclass
class Llama4LisaTrainingConfig:
    """Complete Llama4-LISA Training Configuration"""
    
    # =====================================================
    # Base Model Configuration (Official Llama-4 Spec)
    # =====================================================
    MODEL_ID: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    MODEL_TYPE: str = "llama4_lisa"
    
    # =====================================================
    # Hardware & Environment Settings
    # =====================================================
    DEVICE_MAP: str = "auto"
    TORCH_DTYPE: str = "bfloat16"
    ATTN_IMPLEMENTATION: str = "flex_attention"  # Llama-4 recommended
    
    # Lambda Cloud A100 Optimization
    USE_GRADIENT_CHECKPOINTING: bool = True
    DATALOADER_PIN_MEMORY: bool = True
    DATALOADER_NUM_WORKERS: int = 4
    
    # =====================================================
    # Training Hyperparameters
    # =====================================================
    BATCH_SIZE: int = 1  # A100-40GB optimized
    GRADIENT_ACCUMULATION_STEPS: int = 8  # Effective batch size = 8
    LEARNING_RATE: float = 5e-5
    NUM_EPOCHS: int = 10
    WARMUP_RATIO: float = 0.03
    WEIGHT_DECAY: float = 0.0
    MAX_GRAD_NORM: float = 1.0
    
    # Learning Rate Scheduler
    LR_SCHEDULER_TYPE: str = "cosine"
    NUM_WARMUP_STEPS: Optional[int] = None  # Will be calculated from warmup_ratio
    
    # =====================================================
    # SAM Configuration (Lambda Cloud 環境統一)
    # =====================================================
    SAM_CHECKPOINT_PATH: str = "/lambda/nfs/lisa-gemma-project-fs/data/weights/sam_vit_h_4b8939.pth"
    TRAIN_MASK_DECODER: bool = True
    OUT_DIM: int = 256  # SEG_PROJECTION_DIM と統一
    
    # =====================================================
    # Loss Configuration (Gemma-LISA と統一)
    # =====================================================
    CE_LOSS_WEIGHT: float = 1.0
    DICE_LOSS_WEIGHT: float = 0.5
    BCE_LOSS_WEIGHT: float = 2.0
    
    # =====================================================
    # Dataset Configuration (Lambda Cloud 環境統一)
    # =====================================================
    DATASET_BASE_DIR: str = "/lambda/nfs/lisa-gemma-project-fs/data/dataset"
    SAMPLES_PER_EPOCH: int = 1000
    SAMPLE_RATE: List[int] = field(default_factory=lambda: [9, 3, 3, 1])  # sem_seg, refer_seg, vqa, reason_seg
    
    # Input Processing
    MAX_SEQ_LENGTH: int = 2048  # Context length for training
    IMAGE_SIZE: int = 1024  # SAM image size
    
    # =====================================================
    # Special Tokens
    # =====================================================
    SEG_TOKEN: str = "[SEG]"
    IMAGE_TOKEN: str = "<image>"
    
    # =====================================================
    # Training Output Configuration
    # =====================================================
    OUTPUT_DIR: str = "/lambda/nfs/lisa-gemma-project-fs/outputs/llama4_lisa"
    LOGGING_DIR: str = "/lambda/nfs/lisa-gemma-project-fs/logs/llama4_lisa"
    
    # Checkpointing
    SAVE_STEPS: int = 100
    EVAL_STEPS: int = 50
    SAVE_TOTAL_LIMIT: int = 3
    LOAD_BEST_MODEL_AT_END: bool = True
    
    # Evaluation
    EVALUATION_STRATEGY: str = "steps"
    EVAL_ACCUMULATION_STEPS: int = 1
    
    # =====================================================
    # Logging & Monitoring
    # =====================================================
    LOGGING_STRATEGY: str = "steps"
    LOGGING_STEPS: int = 10
    
    # WandB Configuration
    WANDB_PROJECT: str = "lisa-llama4"
    WANDB_NAME: str = "llama4-scout-17b-lisa"
    WANDB_ENTITY: Optional[str] = None
    REPORT_TO: List[str] = field(default_factory=lambda: ["wandb"])
    
    # =====================================================
    # Model-specific Configuration
    # =====================================================
    # Llama-4 Architecture Parameters (17B activated, 109B total)
    VOCAB_SIZE: int = 128256
    HIDDEN_SIZE: int = 4096
    INTERMEDIATE_SIZE: int = 14336
    NUM_HIDDEN_LAYERS: int = 32
    NUM_ATTENTION_HEADS: int = 32
    NUM_KEY_VALUE_HEADS: int = 8
    MAX_POSITION_EMBEDDINGS: int = 10_000_000  # 10M context
    
    # MoE Configuration
    NUM_EXPERTS: int = 16
    NUM_EXPERTS_PER_TOK: int = 1
    
    # Vision Configuration (SigLIP-based)
    VISION_HIDDEN_SIZE: int = 1024
    VISION_IMAGE_SIZE: int = 1120
    VISION_PATCH_SIZE: int = 14
    VISION_NUM_ATTENTION_HEADS: int = 16
    VISION_NUM_HIDDEN_LAYERS: int = 24
    
    # =====================================================
    # Performance Optimization
    # =====================================================
    USE_FLASH_ATTENTION: bool = True
    USE_XFORMERS: bool = False  # Disabled in favor of flex_attention
    FP16: bool = False  # Using bfloat16 instead
    BF16: bool = True
    
    # Memory Optimization
    REMOVE_UNUSED_COLUMNS: bool = False
    DATALOADER_DROP_LAST: bool = True
    GROUP_BY_LENGTH: bool = False  # Disabled for multimodal
    
    # =====================================================
    # Advanced Training Settings
    # =====================================================
    # Optimizer
    OPTIM: str = "adamw_torch"
    ADAM_BETA1: float = 0.9
    ADAM_BETA2: float = 0.999
    ADAM_EPSILON: float = 1e-8
    
    # Regularization
    DROPOUT_RATE: float = 0.1
    ATTENTION_DROPOUT: float = 0.0
    
    # =====================================================
    # Paths and File Management
    # =====================================================
    CACHE_DIR: str = "/lambda/nfs/lisa-gemma-project-fs/cache"
    HF_HOME: str = "/lambda/nfs/lisa-gemma-project-fs/cache/huggingface"
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Create directories
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.LOGGING_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.HF_HOME, exist_ok=True)
        
        # Set environment variables
        os.environ["HF_HOME"] = self.HF_HOME
        os.environ["TRANSFORMERS_CACHE"] = self.CACHE_DIR
        
        # Calculate warmup steps if not provided
        if self.NUM_WARMUP_STEPS is None:
            total_steps = (self.SAMPLES_PER_EPOCH // (self.BATCH_SIZE * self.GRADIENT_ACCUMULATION_STEPS)) * self.NUM_EPOCHS
            self.NUM_WARMUP_STEPS = int(total_steps * self.WARMUP_RATIO)
        
        # Validate SAM checkpoint path
        if not os.path.exists(self.SAM_CHECKPOINT_PATH):
            print(f"⚠️  Warning: SAM checkpoint not found at {self.SAM_CHECKPOINT_PATH}")
            print("   Please ensure the SAM checkpoint is downloaded")
    
    def to_training_args_dict(self):
        """Convert to transformers TrainingArguments compatible dict"""
        return {
            "output_dir": self.OUTPUT_DIR,
            "logging_dir": self.LOGGING_DIR,
            "num_train_epochs": self.NUM_EPOCHS,
            "per_device_train_batch_size": self.BATCH_SIZE,
            "gradient_accumulation_steps": self.GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": self.LEARNING_RATE,
            "weight_decay": self.WEIGHT_DECAY,
            "max_grad_norm": self.MAX_GRAD_NORM,
            "warmup_ratio": self.WARMUP_RATIO,
            "lr_scheduler_type": self.LR_SCHEDULER_TYPE,
            "save_steps": self.SAVE_STEPS,
            "eval_steps": self.EVAL_STEPS,
            "save_total_limit": self.SAVE_TOTAL_LIMIT,
            "load_best_model_at_end": self.LOAD_BEST_MODEL_AT_END,
            "evaluation_strategy": self.EVALUATION_STRATEGY,
            "logging_strategy": self.LOGGING_STRATEGY,
            "logging_steps": self.LOGGING_STEPS,
            "report_to": self.REPORT_TO,
            "dataloader_pin_memory": self.DATALOADER_PIN_MEMORY,
            "dataloader_num_workers": self.DATALOADER_NUM_WORKERS,
            "dataloader_drop_last": self.DATALOADER_DROP_LAST,
            "gradient_checkpointing": self.USE_GRADIENT_CHECKPOINTING,
            "bf16": self.BF16,
            "fp16": self.FP16,
            "remove_unused_columns": self.REMOVE_UNUSED_COLUMNS,
            "group_by_length": self.GROUP_BY_LENGTH,
            "optim": self.OPTIM,
            "adam_beta1": self.ADAM_BETA1,
            "adam_beta2": self.ADAM_BETA2,
            "adam_epsilon": self.ADAM_EPSILON,
        }
    
    def to_model_config_dict(self):
        """Convert to model configuration dict"""
        return {
            "vocab_size": self.VOCAB_SIZE,
            "hidden_size": self.HIDDEN_SIZE,
            "intermediate_size": self.INTERMEDIATE_SIZE,
            "num_hidden_layers": self.NUM_HIDDEN_LAYERS,
            "num_attention_heads": self.NUM_ATTENTION_HEADS,
            "num_key_value_heads": self.NUM_KEY_VALUE_HEADS,
            "max_position_embeddings": self.MAX_POSITION_EMBEDDINGS,
            "num_experts": self.NUM_EXPERTS,
            "num_experts_per_tok": self.NUM_EXPERTS_PER_TOK,
            "sam_checkpoint_path": self.SAM_CHECKPOINT_PATH,
            "train_mask_decoder": self.TRAIN_MASK_DECODER,
            "out_dim": self.OUT_DIM,
            "ce_loss_weight": self.CE_LOSS_WEIGHT,
            "dice_loss_weight": self.DICE_LOSS_WEIGHT,
            "bce_loss_weight": self.BCE_LOSS_WEIGHT,
            "seg_token": self.SEG_TOKEN,
            "image_token": self.IMAGE_TOKEN,
        }
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("=" * 60)
        print("🚀 LLAMA4-LISA TRAINING CONFIGURATION")
        print("=" * 60)
        
        print(f"📱 Model: {self.MODEL_ID}")
        print(f"🎯 Model Type: {self.MODEL_TYPE}")
        print(f"💾 Device Map: {self.DEVICE_MAP}")
        print(f"🔢 Data Type: {self.TORCH_DTYPE}")
        print(f"⚡ Attention: {self.ATTN_IMPLEMENTATION}")
        
        print("\n" + "=" * 40)
        print("📊 TRAINING PARAMETERS")
        print("=" * 40)
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"Gradient Accumulation: {self.GRADIENT_ACCUMULATION_STEPS}")
        print(f"Effective Batch Size: {self.BATCH_SIZE * self.GRADIENT_ACCUMULATION_STEPS}")
        print(f"Learning Rate: {self.LEARNING_RATE}")
        print(f"Epochs: {self.NUM_EPOCHS}")
        print(f"Warmup Ratio: {self.WARMUP_RATIO}")
        
        print("\n" + "=" * 40)
        print("🎭 SAM CONFIGURATION")
        print("=" * 40)
        print(f"Checkpoint: {self.SAM_CHECKPOINT_PATH}")
        print(f"Train Mask Decoder: {self.TRAIN_MASK_DECODER}")
        print(f"Output Dim: {self.OUT_DIM}")
        
        print("\n" + "=" * 40)
        print("⚖️  LOSS WEIGHTS")
        print("=" * 40)
        print(f"CE Loss: {self.CE_LOSS_WEIGHT}")
        print(f"Dice Loss: {self.DICE_LOSS_WEIGHT}")
        print(f"BCE Loss: {self.BCE_LOSS_WEIGHT}")
        
        print("\n" + "=" * 40)
        print("📁 PATHS")
        print("=" * 40)
        print(f"Output Dir: {self.OUTPUT_DIR}")
        print(f"Logging Dir: {self.LOGGING_DIR}")
        print(f"Dataset Dir: {self.DATASET_BASE_DIR}")
        
        print("=" * 60)

# =====================================================
# Global Configuration Instance
# =====================================================

# Create default configuration
config = Llama4LisaTrainingConfig()

# =====================================================
# Environment-specific Overrides
# =====================================================

def get_lambda_cloud_config():
    """Get Lambda Cloud optimized configuration"""
    lambda_config = Llama4LisaTrainingConfig()
    
    # Lambda Cloud specific paths (Gemma-LISA と統一)
    lambda_config.SAM_CHECKPOINT_PATH = "/lambda/nfs/lisa-gemma-project-fs/data/weights/sam_vit_h_4b8939.pth"
    lambda_config.DATASET_BASE_DIR = "/lambda/nfs/lisa-gemma-project-fs/data/dataset"
    lambda_config.OUTPUT_DIR = "/lambda/nfs/lisa-gemma-project-fs/outputs/llama4_lisa"
    lambda_config.LOGGING_DIR = "/lambda/nfs/lisa-gemma-project-fs/logs/llama4_lisa"
    lambda_config.CACHE_DIR = "/lambda/nfs/lisa-gemma-project-fs/cache"
    lambda_config.HF_HOME = "/lambda/nfs/lisa-gemma-project-fs/cache/huggingface"
    
    # A100-40GB optimizations
    lambda_config.BATCH_SIZE = 1
    lambda_config.GRADIENT_ACCUMULATION_STEPS = 8
    lambda_config.USE_GRADIENT_CHECKPOINTING = True
    lambda_config.DATALOADER_NUM_WORKERS = 4
    
    return lambda_config

def get_debug_config():
    """Get debug configuration for rapid testing"""
    debug_config = Llama4LisaTrainingConfig()
    
    # Minimal settings for debugging
    debug_config.SAMPLES_PER_EPOCH = 10
    debug_config.NUM_EPOCHS = 1
    debug_config.SAVE_STEPS = 5
    debug_config.EVAL_STEPS = 5
    debug_config.LOGGING_STEPS = 1
    debug_config.MAX_SEQ_LENGTH = 512
    
    return debug_config

# =====================================================
# Configuration Factory
# =====================================================

def create_config(config_type: str = "default"):
    """
    Create configuration based on environment type
    
    Args:
        config_type: "default", "lambda", or "debug"
    
    Returns:
        Llama4LisaTrainingConfig instance
    """
    if config_type == "lambda":
        return get_lambda_cloud_config()
    elif config_type == "debug":
        return get_debug_config()
    else:
        return Llama4LisaTrainingConfig()

# =====================================================
# Validation Functions
# =====================================================

def validate_config(config: Llama4LisaTrainingConfig) -> bool:
    """Validate configuration settings"""
    issues = []
    
    # Check required paths
    if not os.path.exists(config.SAM_CHECKPOINT_PATH):
        issues.append(f"SAM checkpoint not found: {config.SAM_CHECKPOINT_PATH}")
    
    if not os.path.exists(config.DATASET_BASE_DIR):
        issues.append(f"Dataset directory not found: {config.DATASET_BASE_DIR}")
    
    # Check batch size
    if config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS < 1:
        issues.append("Effective batch size must be at least 1")
    
    # Check learning rate
    if config.LEARNING_RATE <= 0:
        issues.append("Learning rate must be positive")
    
    if issues:
        print("❌ Configuration validation failed:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ Configuration validation passed")
    return True

# =====================================================
# Main Execution
# =====================================================

if __name__ == "__main__":
    # Print configuration summary
    config.print_config_summary()
    
    # Validate configuration
    validate_config(config) 