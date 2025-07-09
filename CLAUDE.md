# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

オリジナルLISA（オリジナルレポジトリ：./Original-LISA-Code/）のVLMをLlavaからgemma-3-4b-it（2025年6月時点でのGoogleのVLMの最新モデル）に変更して、学習するプロジェクト（./Gemma_LISA-Code/）を実装し、train.pyまで完了していたが、そのプロジェクトからさらにVLMをLlama-4-Scout-17B-16E-Instructに変更して学習するように修正している。

This repository contains LISA-Gemma-Linux, a computer vision project that implements Large Language Instructed Segmentation Assistant (LISA) with multiple model backends including Gemma-3 and Llama-4. The project focuses on multimodal learning combining vision and language for image segmentation tasks.

## Development Rules & Guidelines

### Lambda Cloud Development Workflow
Lambda Cloud環境での実行を行うので、ローカルファイルの修正を行うたびに、以下のコマンド例を参考に、lambda上に転送し、lambda上で実行すること

```bash
# File transfer to Lambda Cloud
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@<ip address>:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

# Remote execution
ssh -i ~/.ssh/lambda_cloud_key ubuntu@<ip address> "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 python -u <script>.py 2>&1"
```

### Core Development Principles

1. **Reference Existing Implementations**: 最も重要な方針として、本プロジェクトは既存の実際に実行できたオリジナルLISAとGemma-LISAにおいて、VLMをLlama-4-Scout-17B-16E-Instructに変更したプロジェクトである。原則としてGemma-LISAプロジェクト（./Gemma_LISA-Code/）とオリジナルLISA（./Original-LISA-Code/）を参考に進めて

2. **Web Research for Official Methods**: 
   - Llama-4-Scout-17B-16E-Instructの使い方については、積極的にWebリサーチを行い、test_llama4_standalone.py, verify_llama4_loss_and_gradients.py, overfit_llama4_single_batch.pyを参照して、公式や非公式の実際に動くコードを積極的に参照して、カスタムコードでなく用意されている方法でシンプルに実装すること
   - SAMの使い方については、積極的にWebリサーチを行い、公式や非公式の実際に動くコードを積極的に参照して、カスタムコードでなく用意されている方法でシンプルに実装すること

3. **Error Handling Strategy**: フォールバック的なコードはエラーを隠蔽するので、エラーを出して止め、一つ一つデバッグするように実装すること

4. **Reference Integration Models**: 統合モデル(llama4_lisa.pyやその参照元ファイル)についてはgemma_lisa.pyやOriginal-LISA-Code（オリジナルのLISAでLlavaとSAMの統合モデルプロジェクト）を参考にして

### Code Quality Standards

- Always run tests before making significant changes
- Use the unified configuration system in `config_linux.py`
- Follow existing code style and naming conventions
- Add detailed docstrings for complex functions
- Never commit sensitive information (API keys, tokens)

## Implementation Status

### [実装完了部分]
- llama4_lisa.pyでLlama-4-Scout-17B-16E-InstructとSAMの統合モデルの実装を完了した
- test_llama4_lisa_standalone.py, verify_llama4_lisa_gradients.py, overfit_llama4_lisa_batch.pyでLlama-4-Scout-17B-16E-InstructとSAMの統合モデルの学習前の検証スクリプトの実装が完了した

## Architecture Components

### Model Implementations
- **LISA-Gemma3**: Integration with Google's Gemma-3-4B-IT model using dual-stream data pipeline
- **LISA-Llama4**: Integration with Meta's Llama-4-Scout-17B-16E-Instruct model  
- **Segment Anything Model (SAM)**: SAM ViT-H for segmentation tasks
- **Dual-stream Architecture**: Separate image processing pipelines for Gemma (896x896) and SAM (1024x1024)

### Core Components
- **MLP Projector**: Bridges between language model hidden states and SAM prompt embeddings
- **LoRA Configuration**: Parameter-efficient fine-tuning for both Gemma and Llama models
- **HybridDataset**: Unified dataset handling for multiple data sources (ReasonSeg, VQA, ReferSeg, SemSeg)
- **DeepSpeed Integration**: Distributed training with ZeRO Stage 2 optimization

## Common Development Commands

### Training Commands
```bash
# Llama-4 training (recommended)
python train_llama4.py --batch_size 1 --lr 1e-4 --exp_name "llama4_experiment"

# Llama-4 with DeepSpeed
python train_llama4_deepspeed.py --deepspeed_config ds_config_llama4_moe.json

# Gemma-3 training (legacy)
python Gemma_LISA-Code/train_simple_test.py --batch_size 1 --lr 1e-4
```

### Testing Commands
```bash
# Llama-4 standalone tests
python test_llama4_standalone.py          # Basic model test
python test_llama4_lisa_standalone.py     # Full LISA integration test
python test_llama4_inference_pipeline.py  # Inference pipeline test

# Verification scripts
python verify_llama4_lisa_gradients.py    # Gradient flow verification
python overfit_llama4_lisa_batch.py       # Learning capability test

# Dataset verification
python verify_dataset_integrity.py        # Dataset health check
python verify_model_architecture.py       # Model architecture check
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Check configuration
python config_linux.py                    # Verify paths and settings

# Environment verification
python verify_config_and_setup.py         # Full environment check
```

## Configuration Management

### Primary Configuration
- **config_linux.py**: Central configuration file containing all model, training, and path settings
- Key settings accessed via:
  - `get_lisa_model_config()`: Model architecture settings
  - `get_lora_config()`: LoRA fine-tuning parameters
  - `get_training_config()`: Training hyperparameters
  - `get_path_config()`: Dataset and checkpoint paths

### Model-Specific Settings
- **Llama-4**: Uses `meta-llama/Llama-4-Scout-17B-16E-Instruct` with 5120 hidden size
- **Gemma-3**: Uses `google/gemma-3-4b-it` with 2560 hidden size
- **SAM**: Uses ViT-H checkpoint with 256 prompt embedding dimension

### DeepSpeed Configurations
- **ds_config_local.json**: Local development (no CPU offload)
- **ds_config_cloud.json**: Cloud GPU training (CPU offload enabled)
- **ds_config_llama4_moe.json**: Llama-4 specific optimization

## Dataset Requirements

### Required Datasets
- **ReasonSeg**: Explanatory segmentation dataset
- **VQA**: LLaVA instruct 150k dataset
- **ReferSeg**: RefCOCO/RefCOCO+/RefCOCOg datasets
- **SemSeg**: ADE20k semantic segmentation

### SAM Checkpoint
- Required: SAM ViT-H checkpoint (`sam_vit_h_4b8939.pth`)
- Path configured in `SAM_CHECKPOINT_PATH` environment variable

## Development Workflow

1. **Environment Setup**: Run `python config_linux.py` to verify configuration
2. **Dataset Verification**: Use `verify_dataset_integrity.py` to check data
3. **Model Testing**: Run appropriate standalone tests before training
4. **Training**: Use DeepSpeed-enabled training scripts for best performance
5. **Lambda Cloud Deployment**: Transfer files and execute remotely for production training

## Memory and Performance Considerations

- **Mixed Precision**: Uses bfloat16 for memory efficiency
- **Gradient Checkpointing**: Enabled for large models
- **Batch Size**: Typically 1 per GPU for Llama-4, 2-4 for Gemma-3
- **Gradient Accumulation**: 8 steps recommended for effective batch size
- **DeepSpeed**: ZeRO Stage 2 for distributed training