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
from utils.dataset import LisaGemma3Dataset, collate_fn_gemma3


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
    """データセットとデータローダーの設定"""
    print("=== データセットとデータローダーの設定 ===")
    
    # サンプルレートの解析
    sample_rates = [float(x) for x in args.sample_rates.split(",")]
    
    # 総サンプル数の計算（分散学習対応）
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    samples_per_epoch = args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size
    
    print(f"ワールドサイズ: {world_size}")
    print(f"エポックあたりサンプル数: {samples_per_epoch:,}")
    
    # データセットの作成
    train_dataset = LisaGemma3Dataset(
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
    
    print(f"訓練データセット作成完了: {len(train_dataset):,} サンプル")
    
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
    """1エポックの学習（仕様書第4章.5）"""
    model_engine.train()
    
    total_loss = 0.0
    total_text_loss = 0.0
    total_dice_loss = 0.0
    total_bce_loss = 0.0
    step_count = 0
    
    print(f"\n=== エポック {epoch+1}/{args.epochs} 開始 ===")
    
    for step, batch in enumerate(train_dataloader):
        if step >= args.steps_per_epoch:
            break
            
        step_count += 1
        
        # バッチをGPUに移動
        device = next(model_engine.parameters()).device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # collate_fn_gemma3からの出力を処理
        # batch keys: "image_paths", "images", "input_ids", "attention_mask", "pixel_values", "masks", "labels"
        
        # フォワードパス - LisaGemmaForCausalLMのforwardメソッドに合わせる
        outputs = model_engine(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"] if "labels" in batch else None,
        )
        
        # 損失計算用のバッチデータを準備
        batch_for_loss = {
            "labels": batch.get("labels"),
            "ground_truth_mask": batch.get("masks")
        }
        
        losses = loss_fn(outputs, batch_for_loss)
        total_loss_val = losses["total_loss"]
        
        # バックワードパス
        model_engine.backward(total_loss_val)
        model_engine.step()
        
        # 統計の更新
        total_loss += total_loss_val.item()
        if "text_loss" in losses:
            total_text_loss += losses["text_loss"].item()
        if "dice_loss" in losses:
            total_dice_loss += losses["dice_loss"].item()
        if "bce_loss" in losses:
            total_bce_loss += losses["bce_loss"].item()
        
        # ログ出力
        if step % 10 == 0:
            avg_loss = total_loss / step_count
            print(f"Step {step}/{args.steps_per_epoch}, Loss: {total_loss_val.item():.4f}, Avg Loss: {avg_loss:.4f}")
            
            # TensorBoardログ
            if writer and torch.distributed.get_rank() == 0:
                global_step = epoch * args.steps_per_epoch + step
                writer.add_scalar("train/total_loss", total_loss_val.item(), global_step)
                if "text_loss" in losses:
                    writer.add_scalar("train/text_loss", losses["text_loss"].item(), global_step)
                if "dice_loss" in losses:
                    writer.add_scalar("train/dice_loss", losses["dice_loss"].item(), global_step)
                if "bce_loss" in losses:
                    writer.add_scalar("train/bce_loss", losses["bce_loss"].item(), global_step)
    
    # エポック終了時の統計
    avg_total_loss = total_loss / step_count
    avg_text_loss = total_text_loss / step_count
    avg_dice_loss = total_dice_loss / step_count
    avg_bce_loss = total_bce_loss / step_count
    
    print(f"エポック {epoch+1} 完了:")
    print(f"  平均総損失: {avg_total_loss:.4f}")
    print(f"  平均テキスト損失: {avg_text_loss:.4f}")
    print(f"  平均DICE損失: {avg_dice_loss:.4f}")
    print(f"  平均BCE損失: {avg_bce_loss:.4f}")
    
    return avg_total_loss


def save_checkpoint(model_engine, epoch, args):
    """チェックポイントの保存"""
    if torch.distributed.get_rank() == 0:
        save_dir = os.path.join(args.log_dir, f"checkpoint_epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        
        # DeepSpeedチェックポイントの保存
        model_engine.save_checkpoint(save_dir)
        print(f"チェックポイントを保存しました: {save_dir}")


def main():
    """メイン関数（仕様書第4章.5）"""
    print("LISA-Gemma3 DeepSpeed学習開始")
    print("=" * 60)
    
    # 1. 引数解析
    args = parse_args()
    
    # 2. 必須パスの検証
    print("=== 設定の検証 ===")
    try:
        check_paths()
        print("✓ 必須パスの検証に成功しました")
    except FileNotFoundError as e:
        print(f"✗ 必須パスの検証に失敗しました: {e}")
        return
    
    # 3. ログディレクトリの作成
    log_dir = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    
    # 4. 分散学習の初期化
    deepspeed.init_distributed()
    
    # 5. Gemma-3プロセッサーの初期化
    print("=== Gemma-3プロセッサーの初期化 ===")
    gemma_processor = AutoProcessor.from_pretrained(args.gemma_model_id)
    
    # 6. モデルとLoRAの設定
    model = setup_model_and_lora(args)
    
    # 7. データセットとデータローダーの設定
    train_dataset = setup_dataset_and_dataloader(args, gemma_processor)
    
    # 8. 損失関数の設定
    loss_fn = setup_loss_function(args)
    
    # 9. DeepSpeedエンジンの初期化
    print("=== DeepSpeedエンジンの初期化 ===")
    
    # collate_fnの準備
    collate_fn = partial(collate_fn_gemma3, gemma_processor=gemma_processor)
    
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=collate_fn,
        config=args.deepspeed_config,
    )
    
    print(f"DeepSpeedエンジン初期化完了")
    print(f"使用GPU数: {torch.distributed.get_world_size()}")
    
    # 10. TensorBoardライターの初期化
    writer = None
    if torch.distributed.get_rank() == 0:
        writer = SummaryWriter(log_dir)
        print(f"TensorBoardログ: {log_dir}")
    
    # 11. 学習ループ
    print("\n=== 学習開始 ===")
    
    for epoch in range(args.epochs):
        # 1エポックの学習
        avg_loss = train_epoch(model_engine, train_dataloader, loss_fn, epoch, args, writer)
        
        # チェックポイントの保存
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model_engine, epoch, args)
        
        # 学習率のログ
        if writer and torch.distributed.get_rank() == 0:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("train/learning_rate", current_lr, epoch)
    
    # 12. 最終チェックポイントの保存
    save_checkpoint(model_engine, args.epochs - 1, args)
    
    # 13. TensorBoardライターのクローズ
    if writer:
        writer.close()
    
    print("\n=== 学習完了 ===")
    print(f"最終チェックポイント: {args.log_dir}/checkpoint_epoch_{args.epochs}")


if __name__ == "__main__":
    main() 