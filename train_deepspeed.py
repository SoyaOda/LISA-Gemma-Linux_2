#!/usr/bin/env python3
"""
LISA-Gemma3 学習スクリプト
仕様書第4章「学習のオーケストレーション」に従った実装
"""

import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_linux import *
from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
from model.losses import CompositeLoss
from utils.dataset import LisaGemma3Dataset, collate_fn_gemma3, HybridDataset, collate_fn


def parse_args():
    """引数解析と設定（仕様書第4章.1）"""
    parser = argparse.ArgumentParser(description="LISA-Gemma3 Training with DeepSpeed")
    
    # 基本設定
    parser.add_argument("--local_rank", default=0, type=int, help="Local rank for distributed training")
    parser.add_argument("--exp_name", default="lisa-gemma3-run1", type=str, help="Experiment name")
    parser.add_argument("--log_dir", default=LOG_BASE_DIR, type=str, help="Log directory")
    
    # モデル設定
    parser.add_argument("--gemma_model_id", default=GEMMA_MODEL_ID, type=str, help="Gemma model ID")
    parser.add_argument("--sam_checkpoint_path", default=SAM_CHECKPOINT_PATH, type=str, help="SAM checkpoint path")
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    
    # 学習設定
    parser.add_argument("--epochs", default=EPOCHS, type=int, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", default=STEPS_PER_EPOCH, type=int, help="Steps per epoch")
    parser.add_argument("--batch_size", default=16, type=int, help="Global batch size")
    parser.add_argument("--grad_accumulation_steps", default=2, type=int, help="Gradient accumulation steps")
    parser.add_argument("--lr", default=LEARNING_RATE, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=WEIGHT_DECAY, type=float, help="Weight decay")
    parser.add_argument("--beta1", default=BETA1, type=float, help="Adam beta1")
    parser.add_argument("--beta2", default=BETA2, type=float, help="Adam beta2")
    
    # LoRA設定
    parser.add_argument("--lora_r", default=LORA_R, type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", default=LORA_ALPHA, type=int, help="LoRA alpha")
    parser.add_argument("--lora_dropout", default=LORA_DROPOUT, type=float, help="LoRA dropout")
    
    # 損失関数の重み
    parser.add_argument("--ce_loss_weight", default=CE_LOSS_WEIGHT, type=float, help="Cross entropy loss weight")
    parser.add_argument("--dice_loss_weight", default=DICE_LOSS_WEIGHT, type=float, help="Dice loss weight")
    parser.add_argument("--bce_loss_weight", default=BCE_LOSS_WEIGHT, type=float, help="BCE loss weight")
    
    # データセット設定
    parser.add_argument("--dataset_base_dir", default=DATASET_BASE_DIR, type=str, help="Dataset base directory")
    parser.add_argument("--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str, help="Datasets to use")
    parser.add_argument("--sample_rates", default=DATASET_SAMPLE_RATES, type=str, help="Dataset sample rates")
    parser.add_argument("--sem_seg_data", default=SEM_SEG_DATA, type=str, help="Semantic segmentation data")
    parser.add_argument("--refer_seg_data", default=REFER_SEG_DATA, type=str, help="Referring segmentation data")
    parser.add_argument("--vqa_data", default=VQA_DATA, type=str, help="VQA data")
    parser.add_argument("--reason_seg_data", default=REASON_SEG_DATA, type=str, help="Reasoning segmentation data")
    
    # DeepSpeed設定
    parser.add_argument("--deepspeed_config", default="ds_config.json", type=str, help="DeepSpeed config file")
    
    # その他
    parser.add_argument("--num_workers", default=4, type=int, help="Number of data loader workers")
    parser.add_argument("--save_interval", default=1, type=int, help="Save interval in epochs")
    parser.add_argument("--eval_interval", default=1, type=int, help="Evaluation interval in epochs")
    parser.add_argument("--resume", default="", type=str, help="Resume from checkpoint")
    
    return parser.parse_args()


def setup_model_and_lora(args):
    """モデルとLoRAの初期化（仕様書第4章.2-3）"""
    print("=== モデルとLoRAの設定 ===")
    
    # 1. LisaGemmaConfigの作成
    config = LisaGemmaConfig(
        gemma_model_id=args.gemma_model_id,
        sam_checkpoint_path=args.sam_checkpoint_path,
        seg_token="<SEG>",
        gemma_hidden_size=2560,
        sam_prompt_embed_dim=256,
    )
    
    # 2. モデルの初期化
    print(f"LISA-Gemmaモデルを初期化中... (precision: {args.precision})")
    model = LisaGemmaForCausalLM(config)
    
    # 3. 精度設定
    if args.precision == "bf16":
        model = model.to(torch.bfloat16)
    elif args.precision == "fp16":
        model = model.to(torch.float16)
    
    # 4. LoRA設定（仕様書第4章.2）
    print("LoRA設定を適用中...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,  # config_linux.pyから取得
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 5. LoRAアダプタの適用
    model = get_peft_model(model, lora_config)
    
    # 6. 訓練可能パラメータの確認
    try:
        model.print_trainable_parameters()
    except AttributeError:
        # PEFTモデルにprint_trainable_parametersがない場合
        pass
    
    param_info = model.get_trainable_parameters_info() if hasattr(model, 'get_trainable_parameters_info') else None
    if param_info:
        print(f"総パラメータ数: {param_info['total_parameters']:,}")
        print(f"訓練可能パラメータ数: {param_info['trainable_parameters']:,}")
        print(f"訓練可能な割合: {param_info['trainable_percentage']:.2f}%")
    else:
        # 手動でパラメータ数を計算
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"総パラメータ数: {total_params:,}")
        print(f"訓練可能パラメータ数: {trainable_params:,}")
        print(f"訓練可能な割合: {trainable_params/total_params*100:.2f}%")
    
    return model


def setup_dataset_and_dataloader(args, gemma_processor):
    """データセットとデータローダーの設定（デュアルストリーム対応）"""
    print("=== データセットとデータローダーの設定 ===")
    
    # サンプルレートの解析
    sample_rates = [float(x) for x in args.sample_rates.split(",")]
    
    # 総サンプル数の計算（分散学習対応）
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    samples_per_epoch = args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size
    
    print(f"ワールドサイズ: {world_size}")
    print(f"エポックあたりサンプル数: {samples_per_epoch:,}")
    
    # デュアルストリーム対応のHybridDatasetを使用
    train_dataset = HybridDataset(
        base_image_dir=args.dataset_base_dir,
        gemma_processor=gemma_processor,
        samples_per_epoch=samples_per_epoch,
        precision=args.precision,
        gemma_image_size=GEMMA_IMAGE_SIZE,
        sam_image_size=SAM_IMAGE_SIZE,
        dataset=args.dataset,
        sample_rate=sample_rates,
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
    )
    
    print(f"✅ デュアルストリーム訓練データセット作成完了: {len(train_dataset):,} サンプル")
    print(f"   - Gemma画像サイズ: {GEMMA_IMAGE_SIZE}x{GEMMA_IMAGE_SIZE}")
    print(f"   - SAM画像サイズ: {SAM_IMAGE_SIZE}x{SAM_IMAGE_SIZE}")
    
    return train_dataset


def setup_loss_function(args):
    """複合損失関数の設定（仕様書第4章.4）"""
    print("=== 複合損失関数の設定 ===")
    
    loss_fn = CompositeLoss(
        ce_loss_weight=args.ce_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        bce_loss_weight=args.bce_loss_weight
    )
    
    print(f"損失関数の重み - CE: {args.ce_loss_weight}, DICE: {args.dice_loss_weight}, BCE: {args.bce_loss_weight}")
    
    return loss_fn


def train_epoch(model_engine, train_dataloader, loss_fn, epoch, args, writer=None):
    """1エポックの学習（仕様書第4章.5 - デュアルストリーム対応）"""
    model_engine.train()
    
    total_loss = 0.0
    total_text_loss = 0.0
    total_mask_loss = 0.0
    step_count = 0
    
    print(f"=== エポック {epoch+1} 開始 ===")
    
    for step, batch in enumerate(train_dataloader):
        if step >= args.steps_per_epoch:
            break
            
        try:
            # デュアルストリーム・バッチの処理
            # バッチをGPUに転送
            device = model_engine.device
            batch_gpu = {}
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_gpu[key] = value.to(device)
                else:
                    batch_gpu[key] = value
            
            # デュアルストリーム・フォワードパス
            outputs = model_engine(
                input_ids=batch_gpu["input_ids"],
                attention_mask=batch_gpu["attention_mask"],
                images_for_gemma=batch_gpu["images_for_gemma"],  # (B, 3, 896, 896)
                images_for_sam=batch_gpu["images_for_sam"],      # (B, 3, 1024, 1024)
                labels=batch_gpu["labels"],
                generate_mask=True
            )
            
            # 複合損失の計算（仕様書第4章.4）
            composite_loss_result = loss_fn(
                outputs=outputs,
                ground_truth_masks=batch_gpu.get("ground_truth_mask"),
                has_masks=batch_gpu.get("has_mask", [])
            )
            
            total_loss_step = composite_loss_result["total_loss"]
            text_loss_step = composite_loss_result["text_loss"]
            mask_loss_step = composite_loss_result["mask_loss"]
            
            # バックワードパス
            model_engine.backward(total_loss_step)
            model_engine.step()
            
            # 統計の更新
            total_loss += total_loss_step.item()
            total_text_loss += text_loss_step.item() if text_loss_step is not None else 0.0
            total_mask_loss += mask_loss_step.item() if mask_loss_step is not None else 0.0
            step_count += 1
            
            # ログ出力
            if step % 10 == 0:
                avg_loss = total_loss / max(step_count, 1)
                avg_text_loss = total_text_loss / max(step_count, 1)
                avg_mask_loss = total_mask_loss / max(step_count, 1)
                
                print(f"Step {step:4d}/{args.steps_per_epoch} | "
                      f"Loss: {avg_loss:.4f} (Text: {avg_text_loss:.4f}, Mask: {avg_mask_loss:.4f})")
                
                if writer:
                    global_step = epoch * args.steps_per_epoch + step
                    writer.add_scalar("train/total_loss", avg_loss, global_step)
                    writer.add_scalar("train/text_loss", avg_text_loss, global_step)
                    writer.add_scalar("train/mask_loss", avg_mask_loss, global_step)
            
        except Exception as e:
            print(f"⚠️ ステップ {step} でエラー: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # エポック統計
    avg_loss = total_loss / max(step_count, 1)
    avg_text_loss = total_text_loss / max(step_count, 1)
    avg_mask_loss = total_mask_loss / max(step_count, 1)
    
    print(f"✅ エポック {epoch+1} 完了")
    print(f"   平均損失: {avg_loss:.4f} (Text: {avg_text_loss:.4f}, Mask: {avg_mask_loss:.4f})")
    
    return {
        "avg_loss": avg_loss,
        "avg_text_loss": avg_text_loss,
        "avg_mask_loss": avg_mask_loss,
        "steps": step_count
    }


def save_checkpoint(model_engine, epoch, args):
    """チェックポイントの保存"""
    if torch.distributed.get_rank() == 0:
        save_dir = os.path.join(args.log_dir, f"checkpoint_epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        
        # DeepSpeedチェックポイントの保存
        model_engine.save_checkpoint(save_dir)
        print(f"チェックポイントを保存しました: {save_dir}")


def main():
    """メイン関数（仕様書第4章統合）"""
    args = parse_args()
    
    # ログディレクトリの作成
    log_dir = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoardライター
    writer = SummaryWriter(log_dir) if torch.distributed.get_rank() == 0 else None
    
    print("=" * 60)
    print("🚀 LISA-Gemma3 学習開始")
    print("=" * 60)
    print(f"実験名: {args.exp_name}")
    print(f"ログディレクトリ: {log_dir}")
    print(f"エポック数: {args.epochs}")
    print(f"バッチサイズ: {args.batch_size}")
    print(f"学習率: {args.lr}")
    
    # 1. モデルとLoRAの設定
    model = setup_model_and_lora(args)
    
    # 2. Gemmaプロセッサーの取得
    gemma_processor = model.gemma_processor if hasattr(model, 'gemma_processor') else None
    if gemma_processor is None:
        from transformers import AutoProcessor
        gemma_processor = AutoProcessor.from_pretrained(args.gemma_model_id)
    
    # 3. データセットとデータローダーの設定（デュアルストリーム対応）
    train_dataset = setup_dataset_and_dataloader(args, gemma_processor)
    
    # 4. デュアルストリーム対応のcollate_fn
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // torch.distributed.get_world_size() if torch.distributed.is_initialized() else args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,  # デュアルストリーム対応
        pin_memory=True,
        drop_last=True
    )
    
    print(f"✅ デュアルストリーム・データローダー作成完了")
    print(f"   - バッチサイズ（GPU毎）: {train_dataloader.batch_size}")
    print(f"   - ワーカー数: {args.num_workers}")
    
    # 5. 損失関数の設定
    loss_fn = setup_loss_function(args)
    
    # 6. DeepSpeed初期化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=args.deepspeed_config,
        model_parameters=model.parameters(),
    )
    
    print(f"✅ DeepSpeed初期化完了")
    print(f"   - ZeRO Stage: {model_engine.zero_optimization_stage()}")
    print(f"   - 精度: {args.precision}")
    
    # 7. 学習ループ
    print("\n" + "=" * 60)
    print("📚 学習開始")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        # 1エポックの学習
        epoch_results = train_epoch(model_engine, train_dataloader, loss_fn, epoch, args, writer)
        
        # チェックポイントの保存
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model_engine, epoch, args)
        
        # TensorBoardログ
        if writer:
            writer.add_scalar("epoch/avg_loss", epoch_results["avg_loss"], epoch)
            writer.add_scalar("epoch/avg_text_loss", epoch_results["avg_text_loss"], epoch)
            writer.add_scalar("epoch/avg_mask_loss", epoch_results["avg_mask_loss"], epoch)
    
    # 最終チェックポイントの保存
    save_checkpoint(model_engine, args.epochs - 1, args)
    
    if writer:
        writer.close()
    
    print("\n" + "=" * 60)
    print("🎉 学習完了!")
    print("=" * 60)


if __name__ == "__main__":
    main() 