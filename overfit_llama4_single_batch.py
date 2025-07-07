#!/usr/bin/env python3
"""
Llama4-LISA 単一バッチ過学習テスト
============================

単一バッチでの過学習テストによりLlama4-LISAモデルの
学習能力を検証するスクリプト

テスト内容:
1. 単一データサンプルを繰り返し学習
2. 損失の減少を監視
3. 過学習の成功可否を判定
4. 損失曲線をプロット保存

期待される結果:
- 損失が着実に減少し、ゼロに近づく
- 学習可能パラメータに適切に勾配が流れる
- モデルがデータを記憶する能力を確認
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib
matplotlib.use('Agg')  # バックエンドを非対話型に設定
import matplotlib.pyplot as plt
import psutil
import traceback
import gc
from pathlib import Path

print("🚀 Llama4-LISA Single Batch Overfitting Test")

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="Llama4-LISA単一バッチでの過学習テスト")
    parser.add_argument("--iterations", type=int, default=20, help="過学習テストのイテレーション数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学習率")
    parser.add_argument("--batch_size", type=int, default=1, help="バッチサイズ")
    parser.add_argument("--wait_between_iterations", type=float, default=0.0, 
                       help="各イテレーション間の待機時間（秒）")
    parser.add_argument("--output_dir", type=str, default="./outputs/llama4_overfit",
                       help="出力ディレクトリ")
    return parser.parse_args()

def setup_memory_optimizations():
    """メモリ最適化設定"""
    print("\n🧠 メモリ最適化設定を適用中...")
    
    # CUDA メモリ最適化環境変数
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    print("  ✅ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128")
    
    if torch.cuda.is_available():
        # TensorFloat-32の有効化（高速化）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  ✅ TensorFloat-32有効化")
        
        # CUDAメモリクリア
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
        print("  ✅ 全GPUメモリクリア完了")

def check_gpu_environment():
    """GPU環境の確認"""
    print("\n💻 GPU環境の確認:")
    print(f"  PyTorch バージョン: {torch.__version__}")
    print(f"  CUDA 対応: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  利用可能GPU数: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name}, メモリ: {memory_gb:.1f}GB")

def plot_loss_curve(loss_history: List[Dict[str, float]], output_path: str):
    """損失曲線をプロットして保存"""
    iterations = list(range(len(loss_history)))
    
    # 各損失成分を抽出
    total_losses = [h['total_loss'] for h in loss_history]
    
    # プロット作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 総損失のプロット（対数スケール）
    ax1.plot(iterations, total_losses, 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_yscale('log')
    ax1.set_title('Llama4-LISA単一バッチ過学習テスト: 総損失の推移（対数スケール）')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 線形スケールでの詳細プロット
    ax2.plot(iterations, total_losses, 'r-', linewidth=2, label='Total Loss (Linear)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('総損失の詳細推移（線形スケール）')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 損失曲線を保存: {output_path}")

def get_memory_usage():
    """GPU/CPUメモリ使用量を取得"""
    memory_info = {}
    
    # CPUメモリ
    cpu_memory = psutil.virtual_memory()
    memory_info['cpu_used_gb'] = cpu_memory.used / (1024**3)
    memory_info['cpu_total_gb'] = cpu_memory.total / (1024**3)
    memory_info['cpu_percent'] = cpu_memory.percent
    
    # GPUメモリ（CUDA利用可能な場合）
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_max = torch.cuda.max_memory_allocated() / (1024**3)
        gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)
        
        memory_info['gpu_used_gb'] = gpu_memory
        memory_info['gpu_max_gb'] = gpu_memory_max
        memory_info['gpu_cached_gb'] = gpu_memory_cached
        
        # GPU利用率（簡易的な計算）
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
    """メモリ情報を読みやすい形式でフォーマット"""
    cpu_info = f"CPU: {memory_info['cpu_used_gb']:.1f}/{memory_info['cpu_total_gb']:.1f}GB ({memory_info['cpu_percent']:.1f}%)"
    
    if torch.cuda.is_available():
        gpu_info = f"GPU: {memory_info['gpu_used_gb']:.1f}/{memory_info['gpu_total_gb']:.1f}GB ({memory_info['gpu_percent']:.1f}%) [Max: {memory_info['gpu_max_gb']:.1f}GB]"
    else:
        gpu_info = "GPU: N/A"
    
    return f"{cpu_info} | {gpu_info}"

def analyze_overfitting_success(loss_history: List[Dict[str, float]]) -> Dict[str, Any]:
    """過学習の成功度を分析"""
    if len(loss_history) < 2:
        return {
            "success": False, 
            "reason": "insufficient_iterations",
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "reduction_ratio": 0.0,
            "stability_ratio": 0.0,
            "iterations": len(loss_history)
        }
    
    initial_loss = loss_history[0]['total_loss']
    final_loss = loss_history[-1]['total_loss']
    
    # 損失減少率
    reduction_ratio = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0
    
    # 最後の損失の安定性チェック（少ないイテレーション対応）
    if len(loss_history) >= 3:
        stable_window = max(2, len(loss_history) // 2)
        recent_losses = [h['total_loss'] for h in loss_history[-stable_window:]]
        recent_std = torch.tensor(recent_losses).std().item()
        recent_mean = torch.tensor(recent_losses).mean().item()
        stability_ratio = recent_std / recent_mean if recent_mean > 0 else float('inf')
    else:
        stability_ratio = 0.0  # 少ないイテレーションでは安定性チェックをスキップ
    
    # 柔軟な成功基準（少ないイテレーション対応）
    if len(loss_history) < 5:
        # 少ないイテレーション（2-4回）の場合：損失減少があれば成功
        success = reduction_ratio > 0.1  # 10%以上の総損失減少
        success_reason = f"short_run_success" if success else f"insufficient_reduction_{reduction_ratio:.3f}"
    else:
        # 十分なイテレーション（5回以上）の場合：厳格な基準
        success = (
            reduction_ratio > 0.5 and  # 50%以上の損失減少
            final_loss < initial_loss * 0.1 and  # 最終損失が初期の10%未満
            stability_ratio < 0.1  # 安定した収束
        )
        
        if not success:
            if reduction_ratio <= 0.5:
                success_reason = f"insufficient_reduction_{reduction_ratio:.3f}"
            elif final_loss >= initial_loss * 0.1:
                success_reason = f"high_final_loss_{final_loss:.3f}"
            else:
                success_reason = f"unstable_convergence_{stability_ratio:.3f}"
        else:
            success_reason = "full_success"
    
    return {
        "success": success,
        "reason": success_reason,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "reduction_ratio": reduction_ratio,
        "stability_ratio": stability_ratio,
        "iterations": len(loss_history)
    }

def create_dummy_batch():
    """ダミーバッチデータの作成"""
    print("\n🎯 過学習テスト用ダミーバッチ作成中...")
    
    # テキストデータ（セグメンテーションタスク用）
    # より短いシーケンス長でメモリ効率化
    input_ids = torch.randint(1, 1000, (1, 64))  # バッチサイズ1、シーケンス長64
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    # 画像データ（Llama4のマルチモーダル入力）
    # より小さい画像サイズでメモリ効率化
    pixel_values = torch.randn(1, 3, 224, 224)  # 448→224に縮小
    
    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'labels': labels
    }
    
    print("✅ ダミーバッチ作成完了")
    print(f"  - input_ids: {input_ids.shape}")
    print(f"  - pixel_values: {pixel_values.shape}")
    
    return batch

def main():
    try:
        args = parse_args()
        
        print("🚀 Llama4-LISA Single Batch Overfitting Test")
        print("="*80)
        print("Llama4専用：単一バッチでの過学習テスト")
        print("="*80)
        
        # 出力ディレクトリの作成
        os.makedirs(args.output_dir, exist_ok=True)
        
        # メモリ最適化設定を最初に適用
        setup_memory_optimizations()
        
        # GPU環境確認
        check_gpu_environment()
        
        # プロジェクトパスを追加
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))
        
        # train_llama4_deepspeed.pyからモデルクラスをインポート
        from train_llama4_deepspeed import (
            Llama4LisaDeepSpeedConfig, 
            Llama4LisaDeepSpeedModel
        )
        
        print("✅ Llama4-LISAモジュール読み込み完了")
        
        # ===== Step 1: モデル初期化 =====
        print("\n" + "="*60)
        print("Step 1: Llama4-LISAモデル初期化")
        print("="*60)
        
        # 設定作成
        config = Llama4LisaDeepSpeedConfig()
        print("✅ モデル設定作成完了")
        
        # GPU設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ デバイス設定: {device}")
        
        # モデル初期化
        print("📥 Llama4-LISAモデル初期化中...")
        model = Llama4LisaDeepSpeedModel(config)
        model = model.to(device)
        model.train()
        
        print("✅ モデル初期化完了")
        
        # パラメータ統計
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100
        
        print(f"📊 モデル統計:")
        print(f"  - 総パラメータ数: {total_params:,}")
        print(f"  - 学習可能パラメータ数: {trainable_params:,}")
        print(f"  - 学習可能率: {trainable_percentage:.2f}%")
        
        # ===== Step 2: オプティマイザ設定 =====
        print("\n" + "="*60)
        print("Step 2: オプティマイザ設定")
        print("="*60)
        
        # 学習可能パラメータのみを対象にオプティマイザを作成
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_parameters, lr=args.learning_rate)
        
        print(f"✅ AdamWオプティマイザ設定完了")
        print(f"  - 学習率: {args.learning_rate}")
        print(f"  - 対象パラメータ数: {sum(p.numel() for p in trainable_parameters):,}")
        
        # ===== Step 3: ダミーデータ作成 =====
        print("\n" + "="*60)
        print("Step 3: 過学習用データ準備")
        print("="*60)
        
        batch = create_dummy_batch()
        
        # データをGPUに移動
        if torch.cuda.is_available():
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            print("✅ データをGPUに移動完了")
        
        # ===== Step 4: 過学習テスト実行 =====
        print("\n" + "="*60)
        print("Step 4: 過学習テスト実行")
        print("="*60)
        
        print(f"🎯 過学習テスト開始 ({args.iterations}イテレーション)")
        print(f"  - バッチサイズ: {args.batch_size}")
        print(f"  - 学習率: {args.learning_rate}")
        
        loss_history = []
        start_time = time.time()
        
        for iteration in range(args.iterations):
            print(f"\n--- イテレーション {iteration + 1}/{args.iterations} ---")
            
            # 勾配をクリア
            optimizer.zero_grad()
            
            # フォワードパス
            try:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    labels=batch['labels']
                )
                
                if not hasattr(outputs, 'loss'):
                    print("❌ モデル出力に損失が含まれていません")
                    break
                
                loss = outputs.loss
                
            except Exception as e:
                print(f"❌ フォワードパス実行エラー: {e}")
                traceback.print_exc()
                break
            
            # バックワードパス
            try:
                loss.backward()
                optimizer.step()
                
            except Exception as e:
                print(f"❌ バックワードパス実行エラー: {e}")
                traceback.print_exc()
                break
            
            # 統計記録
            loss_value = loss.item()
            loss_history.append({
                'iteration': iteration + 1,
                'total_loss': loss_value,
                'timestamp': time.time() - start_time
            })
            
            # メモリ使用量取得
            memory_info = get_memory_usage()
            
            # 進捗表示
            print(f"  損失: {loss_value:.6f}")
            print(f"  メモリ: {format_memory_info(memory_info)}")
            
            # 改善度チェック（最初の数イテレーション後）
            if len(loss_history) > 1:
                prev_loss = loss_history[-2]['total_loss']
                improvement = (prev_loss - loss_value) / prev_loss * 100
                print(f"  改善: {improvement:+.2f}%")
            
            # 待機時間
            if args.wait_between_iterations > 0:
                time.sleep(args.wait_between_iterations)
        
        total_time = time.time() - start_time
        
        # ===== Step 5: 結果分析 =====
        print("\n" + "="*60)
        print("Step 5: 過学習テスト結果分析")
        print("="*60)
        
        if len(loss_history) >= 2:
            analysis = analyze_overfitting_success(loss_history)
            
            print(f"📊 過学習テスト結果:")
            print(f"  - 成功: {'✅ YES' if analysis['success'] else '❌ NO'}")
            print(f"  - 理由: {analysis['reason']}")
            print(f"  - 初期損失: {analysis['initial_loss']:.6f}")
            print(f"  - 最終損失: {analysis['final_loss']:.6f}")
            print(f"  - 損失減少率: {analysis['reduction_ratio']:.1%}")
            print(f"  - 実行イテレーション: {analysis['iterations']}")
            print(f"  - 総実行時間: {total_time:.1f}秒")
            
            # 損失曲線のプロット
            plot_path = os.path.join(args.output_dir, 
                                   f"llama4_overfit_loss_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plot_loss_curve(loss_history, plot_path)
            
            # 結果をJSONで保存
            results = {
                "test_info": {
                    "model": "Llama4-LISA",
                    "timestamp": datetime.now().isoformat(),
                    "iterations": args.iterations,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "total_time_seconds": total_time
                },
                "model_stats": {
                    "total_parameters": int(total_params),
                    "trainable_parameters": int(trainable_params),
                    "trainable_percentage": float(trainable_percentage)
                },
                "overfitting_analysis": analysis,
                "loss_history": loss_history
            }
            
            results_path = os.path.join(args.output_dir, 
                                      f"llama4_overfit_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 結果をファイルに保存:")
            print(f"  - JSON結果: {results_path}")
            print(f"  - 損失曲線: {plot_path}")
            
            if analysis['success']:
                print("\n🎉 過学習テスト成功！")
                print("   Llama4-LISAモデルは正常に学習可能です。")
            else:
                print("\n⚠️  過学習テストで問題が検出されました。")
                print("   モデル設定や学習率を見直してください。")
        
        else:
            print("❌ 十分なイテレーションが実行されませんでした")
        
    except Exception as e:
        print(f"\n❌ 予期しないエラーが発生しました: {e}")
        traceback.print_exc()
    
    finally:
        # メモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("\n🧹 メモリクリーンアップ完了")

if __name__ == "__main__":
    main() 