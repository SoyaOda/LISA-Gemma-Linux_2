#!/usr/bin/env python3
"""
LISA-Llama4 シングルプロセス学習スクリプト（overfit_llama4_lisa_batch.py成功パターン移植）
103Bモデルを1プロセスでModel Parallelism使用し、HybridDatasetで学習

実行方法:
# 最小時間テスト（10ステップ）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_llama4_lisa_single_process.py --exp_name quick_test --steps_per_epoch 10 --epochs 1

# 通常テスト
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_llama4_lisa_single_process.py --exp_name lisa_single --epochs 3
"""

import argparse
import os
import sys
import time
import json
import logging
import gc
from datetime import datetime
from pathlib import Path
import warnings
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # バックエンドを非対話モードに設定
import matplotlib.pyplot as plt
import numpy as np

# Disable warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# LISA-Llama4モデルとユーティリティ
from model.llama4_lisa import LisaLlama4ForCausalLM, LisaLlama4Config
from utils.dataset import HybridDataset, collate_fn
from utils.constants import DEFAULT_SEG_TOKEN
import config_linux

# 評価用メトリクス
from utils.utils import (
    AverageMeter, ProgressMeter, Summary, 
    dict_to_cuda, intersectionAndUnionGPU
)
from utils.hf_auth import ensure_hf_login

def setup_logging(log_dir: str):
    """ロギング設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'train.log'))
        ]
    )
    return logging.getLogger(__name__)

def setup_environment():
    """環境変数の設定（overfit成功パターン）"""
    # GPU表示順序を固定
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # メモリ最適化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"
    
    # HuggingFaceキャッシュ設定
    setup_hf_cache_dirs()

def setup_hf_cache_dirs():
    """HuggingFaceキャッシュディレクトリを自動設定・作成"""
    print("=== HuggingFaceキャッシュディレクトリ設定 ===")
    
    # 候補ディレクトリ（優先順位順）
    cache_candidates = [
        "/lambda/nfs/lisa-gemma-project-fs/cache/huggingface",
        os.path.expanduser("~/.cache/huggingface"),
        "/tmp/huggingface_cache"
    ]
    
    # 書き込み可能なディレクトリを探す
    hf_cache_dir = None
    for candidate in cache_candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            os.makedirs(os.path.join(candidate, "models"), exist_ok=True)
            os.makedirs(os.path.join(candidate, "datasets"), exist_ok=True)
            
            # 書き込みテスト
            test_file = os.path.join(candidate, "test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            hf_cache_dir = candidate
            break
        except (OSError, PermissionError):
            continue
    
    # 環境変数に設定
    if hf_cache_dir:
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
        os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_dir, "datasets")
        os.environ["TOKENIZERS_CACHE"] = os.path.join(hf_cache_dir, "tokenizers")
        print(f"✅ HuggingFaceキャッシュ設定: {hf_cache_dir}")
    else:
        print("⚠️ キャッシュディレクトリ設定に失敗")
    
    print("=== キャッシュディレクトリ設定完了 ===")

def create_model_and_tokenizer(logger):
    """モデルとトークナイザーの作成（overfit成功パターン完全移植）"""
    logger.info("=== LISA-Llama4統合モデル初期化 ===")
    
    try:
        # 動的コンパイルを無効化してGPU分散エラーを回避（成功した検証スクリプトと同じ設定）
        torch.compiler.disable()
        logger.info("動的コンパイル無効化: GPU分散エラー回避のため")
        
        # LISA統合モデル設定（config_linux統一設定を使用）
        lisa_config = LisaLlama4Config(**config_linux.get_lisa_model_config())
        
        # LISA統合モデル初期化（成功したoverfit手法）
        model = LisaLlama4ForCausalLM(lisa_config)
        
        logger.info(f"✓ LISA統合モデル初期化完了")
        logger.info(f"  - 総パラメータ: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
        
    except Exception as e:
        logger.error(f"LISA統合モデル初期化エラー: {e}")
        raise

def apply_lora_config(model, logger):
    """Web調査結果に基づくLoRA設定適用（device_map preservation対応）完全移植版"""
    logger.info("=== LoRA設定適用 ===")
    
    try:
        # PEFT適用前にdevice_mapを保存（Web調査：既知の問題対策）
        original_device_map = None
        original_device_map_location = None
        
        # device_mapの場所を特定して保存（成功パターン完全移植）
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            original_device_map = model.hf_device_map.copy()
            original_device_map_location = "direct"
            logger.info(f"✓ 元のdevice_map保存（直接アクセス）: {len(original_device_map)} エントリ")
        elif hasattr(model, 'llama_model') and hasattr(model.llama_model, 'hf_device_map') and model.llama_model.hf_device_map:
            original_device_map = model.llama_model.hf_device_map.copy()
            original_device_map_location = "llama_model"
            logger.info(f"✓ 元のdevice_map保存（llama_model経由）: {len(original_device_map)} エントリ")
        else:
            logger.warning("⚠️ device_mapが見つかりません。Model Parallelismが未設定の可能性があります。")
        
        # LoRA設定作成（config_linux統一設定を使用）
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            **config_linux.get_lora_config()
        )
        
        # LoRA適用
        model = get_peft_model(model, lora_config)
        
        # device_mapの復元試行（Web調査：PEFT既知問題の対策）成功パターン完全移植
        if original_device_map and original_device_map_location:
            # 複数のアクセス方法を試行
            restoration_success = False
            
            # 方法1: 直接アクセス確認
            if hasattr(model, 'hf_device_map') and model.hf_device_map:
                logger.info("✓ 直接device_mapアクセス確認済み")
                restoration_success = True
            
            # 方法2: base_model経由のアクセス
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map') and model.base_model.hf_device_map:
                logger.info("✓ base_model経由device_mapアクセス確認済み")
                restoration_success = True
            
            # 方法3: llama_model経由のアクセス（LISA統合モデル特有）
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and hasattr(model.base_model.llama_model, 'hf_device_map') and model.base_model.llama_model.hf_device_map:
                logger.info("✓ base_model.llama_model経由device_mapアクセス確認済み")
                restoration_success = True
            
            # 方法4: 手動復元
            if not restoration_success:
                logger.warning("⚠️ device_mapが失われました。手動復元を試行...")
                
                # 元の場所に基づいて復元
                if original_device_map_location == "llama_model":
                    if hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model'):
                        model.base_model.llama_model.hf_device_map = original_device_map
                        logger.info("✓ base_model.llama_model.hf_device_mapを手動復元")
                        restoration_success = True
                elif original_device_map_location == "direct":
                    if hasattr(model, 'base_model'):
                        model.base_model.hf_device_map = original_device_map
                        logger.info("✓ base_model.hf_device_mapを手動復元")
                        restoration_success = True
            
            if not restoration_success:
                logger.error("❌ device_mapの復元に失敗しました")
        
        # 学習可能パラメータ統計
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"✓ LoRA適用完了")
        logger.info(f"  - 学習可能パラメータ: {trainable_params:,}")
        logger.info(f"  - 全パラメータ: {total_params:,}")
        logger.info(f"  - 学習可能割合: {100 * trainable_params / total_params:.3f}%")
        
        # 最終device_map確認
        verify_device_map_after_lora(model, logger)
        
        return model
        
    except Exception as e:
        logger.error(f"LoRA適用エラー: {e}")
        raise

def create_dataset_and_dataloader(model, args, logger):
    """データセットとデータローダー作成"""
    logger.info("=== データセット初期化 ===")
    
    # HybridDatasetの初期化（llama_processorを渡す）
    dataset = HybridDataset(llama_processor=model.llama_processor)
    
    logger.info(f"✓ データセット初期化完了: {len(dataset)} サンプル")
    
    # DataLoader作成
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"✓ DataLoader作成完了: バッチサイズ={args.batch_size}, ワーカー数={args.workers}")
    
    return dataset, dataloader

def verify_device_map_after_lora(model, logger):
    """LoRA適用後のdevice_map確認（LISA統合モデル対応）完全移植版"""
    logger.info("=== LoRA適用後device_map確認 ===")
    
    device_map = None
    access_path = None
    
    # 複数のアクセス方法を試行（成功パターン完全移植）
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        device_map = model.hf_device_map
        access_path = "直接アクセス"
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map') and model.base_model.hf_device_map:
        device_map = model.base_model.hf_device_map
        access_path = "base_model経由"
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and hasattr(model.base_model.llama_model, 'hf_device_map') and model.base_model.llama_model.hf_device_map:
        device_map = model.base_model.llama_model.hf_device_map
        access_path = "base_model.llama_model経由"
    
    if device_map:
        logger.info(f"✓ {access_path}でhf_device_mapアクセス成功")
        logger.info(f"  - デバイスマップ: {dict(list(device_map.items())[:5])}...")
        logger.info(f"  - 使用GPU数: {len(set(device_map.values()))}")
        return True
    else:
        logger.error("❌ device_mapが見つかりません")
        logger.error("Model Parallelismが設定されていません。103Bモデルには必須です。")
        return False

def get_model_device(model):
    """モデルの適切なデバイスを取得（Model Parallelism対応）完全移植版"""
    # Model Parallelismのデバイスマップを探す（LISA統合モデル対応）
    device_map = None
    access_path = None
    
    # 複数のアクセス方法を試行（成功パターン完全移植）
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        device_map = model.hf_device_map
        access_path = "直接アクセス"
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map') and model.base_model.hf_device_map:
        device_map = model.base_model.hf_device_map
        access_path = "base_model経由"
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and hasattr(model.base_model.llama_model, 'hf_device_map') and model.base_model.llama_model.hf_device_map:
        device_map = model.base_model.llama_model.hf_device_map
        access_path = "base_model.llama_model経由"
    
    if device_map:
        # Model Parallelismの場合、最初のデバイスを返す
        return next(iter(device_map.values()))
    else:
        raise RuntimeError("Model Parallelismが設定されていません。103Bモデルには必須です。")

def train_epoch(model, dataloader, optimizer, scheduler, epoch, args, logger, writer=None):
    """1エポックの学習実行（overfit成功パターン）"""
    model.train()
    
    # メトリクス初期化
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(dataloader) if args.steps_per_epoch is None else args.steps_per_epoch,
        [losses],
        prefix=f"Epoch: [{epoch}]"
    )
    
    start_time = time.time()
    
    for step, batch in enumerate(dataloader):
        # ステップ数制限チェック
        if args.steps_per_epoch is not None and step >= args.steps_per_epoch:
            break
        
        # データを適切なデバイスに移動（Model Parallelism対応）完全移植版
        # Model Parallelismのdevice_mapを厳密にチェック（LISA統合モデル対応）
        device_map = None
        access_path = None
        
        # 複数のアクセス方法を試行（成功パターン完全移植）
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            device_map = model.hf_device_map
            access_path = "直接アクセス"
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map') and model.base_model.hf_device_map:
            device_map = model.base_model.hf_device_map
            access_path = "base_model経由"
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and hasattr(model.base_model.llama_model, 'hf_device_map') and model.base_model.llama_model.hf_device_map:
            device_map = model.base_model.llama_model.hf_device_map
            access_path = "base_model.llama_model経由"
        
        if not device_map:
            raise RuntimeError("Model Parallelismが設定されていません。103Bモデルには必須です。")
        
        if step == 0:
            logger.info(f"Model Parallelismデバイスマップ確認: {access_path}")
        
        first_device = next(iter(device_map.values()))
        batch = {k: v.to(first_device) if hasattr(v, 'to') else v for k, v in batch.items()}
        
        # フォワードパス（overfit成功手法）
        optimizer.zero_grad()
        
        # バッチ内容をデバッグ出力
        if step == 0:
            logger.info(f"バッチ内容: {list(batch.keys())}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    logger.info(f"  {key}: {type(value)}")
        
        # 実際のLISA統合モデル使用：完全なフォワードパス（SAM機能付き）
        # HybridDatasetの実際の出力形式に合わせて修正（引数重複エラー対策）
        model_inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch.get('attention_mask'),
            'images_for_llama': batch.get('images_for_llama'),  # Llama4用画像
            'images_for_sam': batch.get('images_for_sam'),      # SAM用画像
            'labels': batch.get('labels'),                      # 実際のラベル使用
            'ground_truth_mask': batch.get('ground_truth_mask'),  # マスク損失計算用
            'generate_mask': True  # SAM機能を有効化
        }
        
        # Noneの値を除去（引数重複エラー対策）
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
        
        model_outputs = model(**model_inputs)
        
        # CompositeLoss統合による損失取得（成功パターン完全移植）
        if isinstance(model_outputs, dict):
            # CompositeLossからの統一損失
            if 'text_loss' in model_outputs:
                loss = model_outputs['text_loss']
            # 予備処理：lossキーも確認
            elif 'loss' in model_outputs:
                loss = model_outputs['loss']
            # フォールバック：手動計算
            else:
                logits = model_outputs.get('logits')
                if logits is None:
                    raise ValueError("logitsが見つかりません")
                
                # 言語モデリング損失を手動計算
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["input_ids"][..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
        else:
            # 非辞書型出力の場合
            if hasattr(model_outputs, 'loss'):
                loss = model_outputs.loss
            else:
                raise ValueError("損失が見つかりません")
        
        # 逆伝播
        loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
        
        # パラメータ更新
        optimizer.step()
        scheduler.step()
        
        # メトリクス更新
        losses.update(loss.item(), batch['input_ids'].size(0))
        
        # ログ出力
        if step % args.print_freq == 0:
            progress.display(step)
            
            # TensorBoard記録
            if writer is not None:
                global_step = epoch * len(dataloader) + step
                writer.add_scalar('Train/Loss', losses.val, global_step)
                writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], global_step)
        
        # メモリクリーンアップ
        del model_outputs, loss
        torch.cuda.empty_cache()
        gc.collect()
    
    epoch_time = time.time() - start_time
    logger.info(f"✓ エポック {epoch} 完了: 平均損失={losses.avg:.4f}, 時間={epoch_time:.1f}秒")
    
    return losses.avg

def main():
    parser = argparse.ArgumentParser(description="LISA-Llama4 シングルプロセス学習（overfit成功パターン移植）")
    
    # 基本設定
    parser.add_argument("--exp_name", type=str, required=True, help="実験名")
    parser.add_argument("--batch_size", type=int, default=1, help="バッチサイズ")
    parser.add_argument("--epochs", type=int, default=3, help="エポック数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学習率")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="勾配クリッピング")
    parser.add_argument("--workers", type=int, default=4, help="データローダーワーカー数")
    parser.add_argument("--print_freq", type=int, default=10, help="ログ出力頻度")
    parser.add_argument("--save_freq", type=int, default=1, help="チェックポイント保存頻度")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="エポックあたりのステップ数（制限）")
    
    args = parser.parse_args()
    
    # 環境設定
    setup_environment()
    
    # ログディレクトリ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{args.exp_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # ロギング設定
    logger = setup_logging(log_dir)
    
    logger.info("=" * 80)
    logger.info("LISA-Llama4 シングルプロセス学習開始（overfit成功パターン移植）")
    logger.info("=" * 80)
    logger.info(f"実験名: {args.exp_name}")
    logger.info(f"バッチサイズ: {args.batch_size}")
    logger.info(f"エポック数: {args.epochs}")
    logger.info(f"学習率: {args.lr}")
    if args.steps_per_epoch:
        logger.info(f"ステップ制限: {args.steps_per_epoch}/エポック")
    
    try:
        # HuggingFace認証確認
        ensure_hf_login()
        
        # モデル初期化
        model = create_model_and_tokenizer(logger)
        model = apply_lora_config(model, logger)
        
        # データセットとデータローダー
        dataset, dataloader = create_dataset_and_dataloader(model, args, logger)
        
        # オプティマイザーとスケジューラー
        # 学習可能パラメータのみ対象
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        total_steps = len(dataloader) * args.epochs
        if args.steps_per_epoch:
            total_steps = args.steps_per_epoch * args.epochs
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6
        )
        
        logger.info(f"✓ オプティマイザー設定完了: 学習可能パラメータ={len(trainable_params):,}")
        
        # TensorBoard
        writer = SummaryWriter(f"./runs/{args.exp_name}_{timestamp}")
        
        # 学習ループ
        for epoch in range(args.epochs):
            avg_loss = train_epoch(
                model, dataloader, optimizer, scheduler, 
                epoch, args, logger, writer
            )
            
            # チェックポイント保存
            if epoch % args.save_freq == 0:
                checkpoint_dir = f"./checkpoints/{args.exp_name}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Model Parallelism対応でチェックポイント保存
                device_map = None
                if hasattr(model, 'hf_device_map') and model.hf_device_map:
                    device_map = model.hf_device_map
                elif hasattr(model, 'base_model') and hasattr(model.base_model, 'hf_device_map') and model.base_model.hf_device_map:
                    device_map = model.base_model.hf_device_map
                elif hasattr(model, 'base_model') and hasattr(model.base_model, 'llama_model') and hasattr(model.base_model.llama_model, 'hf_device_map') and model.base_model.llama_model.hf_device_map:
                    device_map = model.base_model.llama_model.hf_device_map
                
                if not device_map:
                    logger.warning("⚠️ Model Parallelismのdevice_mapが見つかりません。チェックポイント保存をスキップします。")
                    logger.info("✓ 学習完了（チェックポイント保存なし）")
                    continue
                
                # base_modelを通してアクセス
                if hasattr(model, 'base_model'):
                    model_state_dict = model.base_model.state_dict()
                else:
                    model_state_dict = model.state_dict()
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                }, f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt")
                
                logger.info(f"✓ チェックポイント保存: epoch_{epoch}.pt")
        
        logger.info("=" * 80)
        logger.info("✅ 学習完了!")
        logger.info(f"ログディレクトリ: {log_dir}")
        logger.info(f"チェックポイント: ./checkpoints/{args.exp_name}")
        logger.info(f"TensorBoard: tensorboard --logdir=./runs/{args.exp_name}_{timestamp}")
        logger.info("=" * 80)
        
        writer.close()
    
    except Exception as e:
        logger.error(f"❌ エラー: {e}")
        raise

if __name__ == "__main__":
    main()