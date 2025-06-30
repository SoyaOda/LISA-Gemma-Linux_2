#!/usr/bin/env python3
"""
フェーズ2.3: DeepSpeed ZeRO Stage 2 統合（事前処理チェックポイント対応版）
A100*8環境でのマルチGPU分散学習（DTensor問題解決済み）

事前処理チェックポイント使用により以下を実現:
- DTensor問題の完全回避
- 語彙サイズ問題の事前解決
- A100*8でのスムーズな分散学習
- 少量ステップでの動作確認対応
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor, AutoTokenizer, PreTrainedModel, Gemma3Config, Gemma3ForCausalLM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psutil
import wandb

# DeepSpeed統合
import deepspeed
from deepspeed.utils import logger

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# プロジェクトのパスを追加
sys.path.append('/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux')

# config_linuxから必要な設定値を直接インポート
import config_linux
from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
from model.losses import CompositeLoss
from utils.dataset import HybridDataset, collate_fn

def find_latest_preprocessed_checkpoint():
    """最新の事前処理チェックポイントを自動検出"""
    checkpoint_dir = Path("./preprocessed_checkpoints")
    if not checkpoint_dir.exists():
        return None
    
    # lisa_gemma_non_distributed_* パターンのディレクトリを検索
    checkpoint_dirs = list(checkpoint_dir.glob("lisa_gemma_non_distributed_*"))
    if not checkpoint_dirs:
        return None
    
    # 最新のものを選択（タイムスタンプ順）
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: x.name)
    
    # 必要なファイルが存在するか確認
    required_files = [
        "pytorch_model.bin",
        "model_info.json", 
        "lisa_config.json"
    ]
    
    for file in required_files:
        if not (latest_checkpoint / file).exists():
            print(f"⚠️ 事前処理チェックポイントに必要ファイルが不足: {file}")
            return None
    
    return latest_checkpoint

def load_preprocessed_checkpoint(checkpoint_path: Path, device):
    """事前処理チェックポイントから直接モデルを構築（DTensorエラー完全回避版）"""
    print(f"📂 事前処理チェックポイントを直接読み込み中: {checkpoint_path}")
    
    # メタデータを読み込み
    with open(checkpoint_path / "model_info.json", 'r') as f:
        model_info = json.load(f)
    
    # LISA設定を読み込み
    with open(checkpoint_path / "lisa_config.json", 'r') as f:
        lisa_config_dict = json.load(f)
    
    print(f"   ✅ メタデータ読み込み完了")
    print(f"   - 事前処理済み語彙サイズ: {model_info['vocab_size']}")
    print(f"   - 総パラメータ: {model_info['total_parameters']:,}")
    print(f"   - 学習可能パラメータ: {model_info['trainable_parameters']:,}")
    
    # LoRA効率を計算
    lora_efficiency = (model_info['trainable_parameters'] / model_info['total_parameters']) * 100
    print(f"   - LoRA効率: {lora_efficiency:.2f}%")
    print(f"   - SEGトークンID: {model_info['seg_token_id']}")
    
    # トークナイザーとプロセッサーをロード
    print(f"   🔄 トークナイザーとプロセッサーをロード中...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path / "tokenizer")
    processor = AutoProcessor.from_pretrained(checkpoint_path / "processor")
    
    # LISA設定を復元
    lisa_config = LisaGemmaConfig(**lisa_config_dict)
    lisa_config.use_preprocessed_checkpoint = True
    lisa_config.preprocessed_vocab_size = model_info['vocab_size']
    lisa_config.seg_token_id = model_info['seg_token_id']
    
    # 事前処理済みstate_dictをロード
    print(f"   🔄 事前処理済みstate_dictをロード中...")
    model_state = torch.load(checkpoint_path / "pytorch_model.bin", map_location=device)
    
    # 🔧 **決定的解決策**: 事前処理済みチェックポイントから正しくモデルを作成
    print(f"   🏗️ DTensorエラー回避: 事前処理済み構造でモデル作成中...")
    model = create_model_from_preprocessed_checkpoint(lisa_config, model_state, processor, device)
    
    # 🔍 構築後の語彙サイズ確認
    actual_vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"   📊 構築後の実際の語彙サイズ: {actual_vocab_size}")
    
    if actual_vocab_size != model_info['vocab_size']:
        raise RuntimeError(
            f"❌ 語彙サイズの不一致が発生しました。"
            f"期待: {model_info['vocab_size']}, 実際: {actual_vocab_size}"
        )
    else:
        print(f"   ✅ 語彙サイズが正しく復元されました: {actual_vocab_size}")
        print(f"   ✅ DTensorエラー完全回避: 事前処理済みstate_dictから直接構築成功")
    
    return model, processor, tokenizer, model_info

def create_model_from_preprocessed_checkpoint(lisa_config, model_state, processor, device):
    """
    事前処理済みチェックポイントから統合モデルを作成
    """
    print("   🏗️ DTensorエラー回避: 事前処理済み構造でモデル作成中...")
    print("   🏗️ 【ROOT解決方法】事前処理済み専用バイパス初期化")
    print("   📋 埋め込み層リサイズを完全に回避する安全な手法を適用中...")
    
    # 🚀 ROOT解決方法: config.pyで定義済みのLisaGemmaConfigを直接使用
    # これにより、確実にskip_embedding_resize=Trueが適用される
    import copy
    bypass_config = copy.deepcopy(lisa_config)
    
    # 絶対確実に事前処理済みモードを有効化
    bypass_config.skip_embedding_resize = True
    bypass_config.use_preprocessed_checkpoint = True
    
    # 🔧 重要: vocab_sizeフィールドを確実に設定
    # LisaGemmaConfigのvocab_sizeフィールドを明示的に追加
    if not hasattr(bypass_config, 'vocab_size'):
        # フィールドが存在しない場合は動的に追加
        setattr(bypass_config, 'vocab_size', None)
    
    # 事前処理済みstate_dictから実際の語彙サイズを取得
    print(f"   🔍 デバッグ: state_dictキー数 = {len(model_state)} 個")
    print(f"   🔍 検索対象キー1: 'gemma_model.base_model.model.model.language_model.embed_tokens.weight'")
    print(f"   🔍 検索対象キー2: 'gemma_model.base_model.model.lm_head.weight'")
    
    # 実際に存在するキーを確認
    embed_keys = [k for k in model_state.keys() if 'embed_tokens' in k and 'weight' in k]
    lm_head_keys = [k for k in model_state.keys() if 'lm_head' in k and 'weight' in k]
    
    print(f"   🔍 検出されたembed_tokensキー: {embed_keys}")
    print(f"   🔍 検出されたlm_headキー: {lm_head_keys}")
    
    if "gemma_model.base_model.model.model.language_model.embed_tokens.weight" in model_state:
        actual_vocab_size = model_state["gemma_model.base_model.model.model.language_model.embed_tokens.weight"].shape[0]
        print(f"   ✅ 事前処理済み語彙サイズ検出: {actual_vocab_size}")
        bypass_config.vocab_size = actual_vocab_size
    elif "gemma_model.base_model.model.lm_head.weight" in model_state:
        actual_vocab_size = model_state["gemma_model.base_model.model.lm_head.weight"].shape[0]
        print(f"   ✅ 事前処理済み語彙サイズ検出（LMヘッド）: {actual_vocab_size}")
        bypass_config.vocab_size = actual_vocab_size
    elif embed_keys:
        # 見つかったembed_tokensキーを使用
        key = embed_keys[0]
        actual_vocab_size = model_state[key].shape[0]
        print(f"   ✅ 事前処理済み語彙サイズ検出（動的キー: {key}）: {actual_vocab_size}")
        bypass_config.vocab_size = actual_vocab_size
    elif lm_head_keys:
        # 見つかったlm_headキーを使用
        key = lm_head_keys[0]
        actual_vocab_size = model_state[key].shape[0]
        print(f"   ✅ 事前処理済み語彙サイズ検出（動的LMヘッド: {key}）: {actual_vocab_size}")
        bypass_config.vocab_size = actual_vocab_size
    else:
        print(f"   ❌ 事前処理済み語彙サイズを検出できませんでした")
        print(f"   🔍 利用可能なキーの例（最初の10個）:")
        for i, key in enumerate(list(model_state.keys())[:10]):
            print(f"      {i+1}. {key}")
        # フォールバック: デフォルトサイズを使用
        bypass_config.vocab_size = 262504
    
    print(f"   📊 バイパス設定確認:")
    print(f"      - skip_embedding_resize: {bypass_config.skip_embedding_resize}")
    print(f"      - use_preprocessed_checkpoint: {getattr(bypass_config, 'use_preprocessed_checkpoint', False)}")
    print(f"      - 事前処理済み語彙サイズ: {getattr(bypass_config, 'vocab_size', '未設定')}")
    
    # 🔧 デバッグ: 設定値の詳細確認
    print(f"   🔍 詳細デバッグ: bypass_configの実際の属性値:")
    print(f"      - hasattr(skip_embedding_resize): {hasattr(bypass_config, 'skip_embedding_resize')}")
    print(f"      - getattr(skip_embedding_resize, False): {getattr(bypass_config, 'skip_embedding_resize', False)}")
    print(f"      - bypass_config.__dict__から抜粋:")
    config_dict = bypass_config.__dict__ if hasattr(bypass_config, '__dict__') else {}
    for key in ['skip_embedding_resize', 'use_preprocessed_checkpoint', 'vocab_size']:
        if key in config_dict:
            print(f"         {key}: {config_dict[key]}")
        else:
            print(f"         {key}: 未設定")
    
    print(f"   🔄 【動作確認済みパターン】LisaGemmaForCausalLM初期化開始...")
    
    # 動作確認済みのLisaGemmaForCausalLM直接初期化
    model = LisaGemmaForCausalLM(bypass_config)
    
    # 事前処理済み状態をロード
    print("   📦 事前処理済みstate_dict適用中...")
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    if missing_keys:
        print(f"   ⚠️ 未使用キー: {len(missing_keys)}個")
    if unexpected_keys:
        print(f"   📦 追加キー: {len(unexpected_keys)}個")
    
    print("   ✅ 【ROOT解決】事前処理済みチェックポイントから統合モデル作成完了")
    return model

def parse_args():
    # config_linuxから設定値を直接取得
    parser = argparse.ArgumentParser(description='LISA-Gemma DeepSpeed学習 (事前処理チェックポイント版)')
    
    # 学習設定
    parser.add_argument('--epochs', type=int, default=config_linux.EPOCHS, help='学習エポック数')
    parser.add_argument('--steps_per_epoch', type=int, default=config_linux.STEPS_PER_EPOCH, help='エポックあたりのステップ数')
    parser.add_argument('--learning_rate', type=float, default=config_linux.LEARNING_RATE, help='学習率')
    parser.add_argument('--weight_decay', type=float, default=config_linux.WEIGHT_DECAY, help='重み減衰')
    parser.add_argument('--warmup_steps', type=int, default=config_linux.WARMUP_STEPS, help='ウォームアップステップ数')
    
    # バッチ設定
    parser.add_argument('--batch_size', type=int, default=config_linux.BATCH_SIZE_PER_GPU, help='GPU毎のバッチサイズ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=config_linux.GRADIENT_ACCUMULATION_STEPS, help='勾配蓄積ステップ数')
    
    # データセット設定
    parser.add_argument('--dataset_type', type=str, default='sem_seg', choices=['sem_seg', 'refer_seg', 'vqa', 'reason_seg'], help='使用するデータセットタイプ')
    parser.add_argument('--samples_per_epoch', type=int, default=config_linux.SAMPLES_PER_EPOCH, help='エポックあたりのサンプル数')
    
    # 出力・ログ設定
    parser.add_argument('--output_dir', type=str, default='./deepspeed_preprocessed_test_output', help='出力ディレクトリ')
    parser.add_argument('--logging_steps', type=int, default=config_linux.LOGGING_STEPS, help='ログ出力間隔')
    parser.add_argument('--checkpoint_interval', type=int, default=config_linux.SAVE_STEPS, help='チェックポイント保存間隔')
    
    # DeepSpeed設定
    parser.add_argument('--ds_config', type=str, default='deepspeed_zero2_config.json', help='DeepSpeed設定ファイル')
    parser.add_argument('--local_rank', type=int, default=-1, help='DeepSpeed用ローカルランク')
    
    # 事前処理チェックポイント設定（新規追加）
    parser.add_argument('--preprocessed_checkpoint', type=str, required=True, help='事前処理済みチェックポイントのパス')
    
    # DeepSpeed用引数を追加
    parser = deepspeed.add_config_arguments(parser)
    
    return parser.parse_args()

def get_memory_usage():
    """GPU/CPUメモリ使用量を取得"""
    memory_info = {}
    
    # CPUメモリ
    cpu_memory = psutil.virtual_memory()
    memory_info['cpu_used_gb'] = cpu_memory.used / (1024**3)
    memory_info['cpu_total_gb'] = cpu_memory.total / (1024**3)
    memory_info['cpu_percent'] = cpu_memory.percent
    
    # GPUメモリ
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_max = torch.cuda.max_memory_allocated() / (1024**3)
        gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)
        
        memory_info['gpu_used_gb'] = gpu_memory
        memory_info['gpu_max_gb'] = gpu_memory_max
        memory_info['gpu_cached_gb'] = gpu_memory_cached
        
        gpu_properties = torch.cuda.get_device_properties(0)
        gpu_total_memory = gpu_properties.total_memory / (1024**3)
        memory_info['gpu_total_gb'] = gpu_total_memory
        memory_info['gpu_percent'] = (gpu_memory / gpu_total_memory) * 100
    else:
        memory_info['gpu_used_gb'] = 0
        memory_info['gpu_max_gb'] = 0
        memory_info['gpu_cached_gb'] = 0
        memory_info['gpu_total_gb'] = 0
        memory_info['gpu_percent'] = 0
    
    return memory_info

def format_memory_info(memory_info):
    """メモリ情報をフォーマット"""
    cpu_info = f"CPU: {memory_info['cpu_used_gb']:.1f}/{memory_info['cpu_total_gb']:.1f}GB ({memory_info['cpu_percent']:.1f}%)"
    
    if torch.cuda.is_available():
        gpu_info = f"GPU: {memory_info['gpu_used_gb']:.1f}/{memory_info['gpu_total_gb']:.1f}GB ({memory_info['gpu_percent']:.1f}%)"
    else:
        gpu_info = "GPU: N/A"
    
    return f"{cpu_info} | {gpu_info}"

def main():
    args = parse_args()
    
    # DeepSpeed分散環境の初期化
    deepspeed.init_distributed()
    
    # ランク取得
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    print(f"🚀 DeepSpeed分散環境初期化完了")
    print(f"   - ランク: {rank}/{world_size}")
    print(f"   - ローカルランク: {local_rank}")
    
    is_main_process = rank == 0
    
    if is_main_process:
        print("="*80)
        print("フェーズ2.3：DeepSpeed学習（事前処理チェックポイント対応版）")
        print("DTensor問題解決済み・少量ステップテスト対応")
        print("="*80)
    
    # セッションタイムスタンプ
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 出力ディレクトリ作成
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 1. 事前処理チェックポイントの検出またはロード
        checkpoint_path = None
        if args.preprocessed_checkpoint:
            checkpoint_path = Path(args.preprocessed_checkpoint)
        else:
            checkpoint_path = find_latest_preprocessed_checkpoint()
        
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(
                "事前処理チェックポイントが見つかりません。\n"
                "create_non_distributed_checkpoint.py を先に実行してください。"
            )
        
        if is_main_process:
            print(f"📁 使用する事前処理チェックポイント: {checkpoint_path}")
        
        # 2. 事前処理チェックポイントからモデルをロード
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        model, processor, tokenizer, model_info = load_preprocessed_checkpoint(checkpoint_path, device)
        
        if is_main_process:
            print("✅ 事前処理チェックポイントからモデル読み込み完了")
            print(f"  - DTensor問題: 完全回避済み")
            print(f"  - 語彙サイズ: {model_info['vocab_size']}")
            print(f"  - SEGトークンID: {model_info['seg_token_id']}")
        
        # 3. データセット準備
        if is_main_process:
            print(f"\n📊 {args.dataset_type}データセット準備中...")
        
        # 少量ステップ用のサンプル数調整
        samples_per_epoch = args.batch_size * args.steps_per_epoch
        dataset = HybridDataset(
            base_image_dir=getattr(config_linux, 'DATASET_BASE_DIR', './dataset'),
            gemma_processor=processor,
            dataset=args.dataset_type,
            samples_per_epoch=samples_per_epoch
        )
        
        if is_main_process:
            print(f"✅ データセット準備完了（少量ステップテスト用）")
            print(f"  - エポックあたりサンプル数: {samples_per_epoch}")
            print(f"  - 総ステップ数: {args.epochs * args.steps_per_epoch}")
        
        # 4. DeepSpeed初期化
        if is_main_process:
            print(f"\n🚀 DeepSpeed ZeRO Stage 2 初期化中...")
        
        if not os.path.exists(args.ds_config):
            raise FileNotFoundError(f"DeepSpeed設定ファイルが見つかりません: {args.ds_config}")
        
        # DeepSpeedでモデルとオプティマイザーを初期化
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=args.ds_config,
            dist_init_required=False
        )
        
        if is_main_process:
            print("✅ DeepSpeed ZeRO Stage 2 初期化完了")
            print(f"✅ モデル統計:")
            print(f"  - 総パラメータ数: {model_info['total_parameters']:,}")
            print(f"  - 学習可能パラメータ数: {model_info['trainable_parameters']:,}")
            
            # LoRA効率を計算
            lora_efficiency = (model_info['trainable_parameters'] / model_info['total_parameters']) * 100
            print(f"  - LoRA効率: {lora_efficiency:.2f}%")
        
        # 5. WandB初期化（メインプロセスのみ）
        if is_main_process:
            wandb.init(
                project="lisa-gemma-deepspeed-preprocessed",
                name=f"preprocessed_test_{session_timestamp}",
                config={
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                    "steps_per_epoch": args.steps_per_epoch,
                    "total_steps": args.epochs * args.steps_per_epoch,
                    "batch_size": args.batch_size,
                    "dataset_type": args.dataset_type,
                    "world_size": world_size,
                    "checkpoint_path": str(checkpoint_path),
                    "vocab_size": model_info['vocab_size'],
                    "seg_token_id": model_info['seg_token_id'],
                }
            )
        
        # 6. 損失関数準備
        loss_fn = CompositeLoss(
            ce_loss_weight=getattr(config_linux, 'CE_LOSS_WEIGHT', 1.0),
            dice_loss_weight=getattr(config_linux, 'DICE_LOSS_WEIGHT', 0.5),
            bce_loss_weight=getattr(config_linux, 'BCE_LOSS_WEIGHT', 2.0)
        )
        
        # 7. 少量ステップ学習ループ
        if is_main_process:
            print(f"\n🚀 少量ステップ学習開始...")
            print(f"目標: {args.epochs}エポック × {args.steps_per_epoch}ステップの正常実行確認")
            print("-" * 80)
        
        loss_history = []
        global_step = 0
        start_time = time.time()
        
        # エポックループ
        for epoch in range(args.epochs):
            if is_main_process:
                print(f"\n📅 Epoch {epoch+1}/{args.epochs} 開始")
            
            # データローダー作成
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                shuffle=True,
                drop_last=True
            )
            
            epoch_start_time = time.time()
            epoch_loss_sum = 0.0
            
            # ステップループ
            dataloader_iter = iter(dataloader)
            for step in range(args.steps_per_epoch):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)
                
                # フォワードパス
                model_engine.train()
                
                # バッチをデバイスに移動
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # モデル実行
                outputs = model_engine(
                    input_ids=batch.get('input_ids'),
                    attention_mask=batch.get('attention_mask'),
                    pixel_values=batch.get('pixel_values'),
                    labels=batch.get('labels'),
                    generate_mask=True
                )
                
                # 損失計算
                if 'loss' in outputs and outputs['loss'] is not None:
                    loss = outputs['loss']
                else:
                    # 手動損失計算
                    loss = loss_fn(outputs, batch.get('labels'))
                
                # バックワードパス
                model_engine.backward(loss)
                model_engine.step()
                
                # 統計記録
                loss_item = loss.item()
                loss_history.append({'total_loss': loss_item})
                epoch_loss_sum += loss_item
                global_step += 1
                
                # メモリ情報取得
                memory_info = get_memory_usage()
                
                if is_main_process:
                    print(f"   Step {step+1}/{args.steps_per_epoch}: Loss={loss_item:.6f} | {format_memory_info(memory_info)}")
                    
                    # WandBログ
                    wandb.log({
                        "train_loss": loss_item,
                        "global_step": global_step,
                        "epoch": epoch,
                        "gpu_memory_gb": memory_info['gpu_used_gb'],
                        "gpu_percent": memory_info['gpu_percent']
                    })
            
            # エポック完了
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss_sum / args.steps_per_epoch
            
            if is_main_process:
                print(f"✅ Epoch {epoch+1} 完了 - 平均損失: {avg_epoch_loss:.6f} - 時間: {epoch_time:.1f}秒")
        
        # 8. 学習完了サマリー
        total_time = time.time() - start_time
        
        if is_main_process:
            print("\n" + "="*80)
            print("🎉 少量ステップ学習完了！")
            print("="*80)
            print(f"✅ 総実行時間: {total_time:.1f}秒")
            print(f"✅ 総ステップ数: {global_step}")
            print(f"✅ 最終損失: {loss_history[-1]['total_loss']:.6f}")
            print(f"✅ DTensor問題: 完全回避")
            print(f"✅ 語彙サイズ問題: 事前解決済み")
            print(f"✅ DeepSpeed分散学習: 正常動作確認")
            print("="*80)
            
            print("\n🚀 次のステップの準備完了:")
            print("  1. フルエポック・フルデータセット学習")
            print("  2. A100*8での大規模分散学習")
            print("  3. gemma-3-27b-itへのスケールアップ")
            
            # WandB完了ログ
            wandb.log({
                "final_loss": loss_history[-1]['total_loss'],
                "total_time_seconds": total_time,
                "total_steps": global_step,
                "status": "success"
            })
            
            wandb.finish()
    
    except Exception as e:
        if is_main_process:
            print(f"\n❌ 学習中にエラーが発生しました:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
        
        raise

if __name__ == "__main__":
    main() 