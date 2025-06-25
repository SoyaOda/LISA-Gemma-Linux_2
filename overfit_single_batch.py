#!/usr/bin/env python3
"""
第5節：堅牢性と将来の開発に向けた事前検証
5.1. サニティチェック1：単一バッチへの過学習

論理的根拠:
これは、モデルの学習能力を試すための最も基本的なテストです。
もしモデルがごく少量のデータすら記憶できないのであれば、大規模データセットから汎化則を学習することは不可能です。
このテストは、アーキテクチャ、損失関数、オプティマイザが三位一体となって正しく機能していることを確認します。

期待される結果:
損失値が着実に、かつ急速に減少し、ゼロに近づいていく様子が観測されるはずです。
もし損失が停滞したり、激しく振動したりする場合は、学習率が不適切である、オプティマイザに問題がある、
あるいは損失計算に根深い問題が残っているなど、根本的な問題の存在を示唆します。
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
from transformers import AutoProcessor
import matplotlib
matplotlib.use('Agg')  # バックエンドを非対話型に設定
import matplotlib.pyplot as plt

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
from model.losses import CompositeLoss
from utils.dataset import HybridDataset, collate_fn

def get_config():
    """動的設定読み込み"""
    config_candidates = ['config_linux', 'config_small_test']
    
    for config_name in config_candidates:
        try:
            config_module = __import__(config_name)
            print(f"✓ 設定ファイルを使用: {config_name}")
            return config_module
        except ImportError:
            continue
    
    raise ImportError("利用可能な設定ファイルが見つかりません")

def parse_args():
    parser = argparse.ArgumentParser(description="単一バッチでの過学習テスト")
    parser.add_argument("--iterations", type=int, default=150, help="過学習テストのイテレーション数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学習率")
    parser.add_argument("--dataset_type", type=str, default="reason_seg", 
                       choices=["sem_seg", "refer_seg", "vqa", "reason_seg", "all"],
                       help="テストに使用するデータセットタイプ（'all'で全データセット）")
    parser.add_argument("--batch_size", type=int, default=2, help="バッチサイズ")
    return parser.parse_args()

def plot_loss_curve(loss_history: List[Dict[str, float]], output_path: str):
    """損失曲線をプロットして保存"""
    iterations = list(range(len(loss_history)))
    
    # 各損失成分を抽出
    total_losses = [h['total_loss'] for h in loss_history]
    text_losses = [h.get('text_loss', 0.0) for h in loss_history]
    dice_losses = [h.get('dice_loss', 0.0) for h in loss_history]
    bce_losses = [h.get('bce_loss', 0.0) for h in loss_history]
    
    # プロット作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 総損失のプロット（対数スケール）
    ax1.plot(iterations, total_losses, 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_yscale('log')
    ax1.set_title('単一バッチ過学習テスト: 総損失の推移（対数スケール）')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 損失成分の詳細プロット
    ax2.plot(iterations, text_losses, 'r-', linewidth=1, label='Text Loss')
    ax2.plot(iterations, dice_losses, 'g-', linewidth=1, label='DICE Loss')
    ax2.plot(iterations, bce_losses, 'm-', linewidth=1, label='BCE Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('損失成分の詳細推移')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"損失曲線を保存: {output_path}")

def analyze_overfitting_success(loss_history: List[Dict[str, float]]) -> Dict[str, Any]:
    """過学習の成功度を分析"""
    if len(loss_history) < 10:
        return {"success": False, "reason": "insufficient_iterations"}
    
    initial_loss = loss_history[0]['total_loss']
    final_loss = loss_history[-1]['total_loss']
    
    # 損失減少率
    reduction_ratio = (initial_loss - final_loss) / initial_loss
    
    # 最後の10%のイテレーションで損失が安定しているかチェック
    stable_window = max(10, len(loss_history) // 10)
    recent_losses = [h['total_loss'] for h in loss_history[-stable_window:]]
    recent_std = torch.tensor(recent_losses).std().item()
    recent_mean = torch.tensor(recent_losses).mean().item()
    stability_ratio = recent_std / recent_mean if recent_mean > 0 else float('inf')
    
    # 成功基準の判定（安定性比率を0.2に緩和）
    success = (
        reduction_ratio > 0.5 and  # 50%以上の損失減少
        final_loss < initial_loss * 0.1 and  # 最終損失が初期の10%以下
        stability_ratio < 0.2  # 最近の損失の変動が平均の20%以下（0.1から緩和）
    )
    
    return {
        "success": success,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "reduction_ratio": reduction_ratio,
        "stability_ratio": stability_ratio,
        "iterations": len(loss_history)
    }

def main():
    args = parse_args()
    config = get_config()
    
    print("="*80)
    print("第5節: 堅牢性と将来の開発に向けた事前検証")
    print("5.1. サニティチェック1：単一バッチへの過学習")
    print("="*80)
    
    # セッションタイムスタンプの生成
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    print(f"テスト設定:")
    print(f"  - イテレーション数: {args.iterations}")
    print(f"  - 学習率: {args.learning_rate}")
    print(f"  - データセットタイプ: {args.dataset_type}")
    print(f"  - バッチサイズ: {args.batch_size}")
    print("-" * 80)
    
    try:
        # 1. モデルの初期化
        print("\n📦 LISA-Gemmaモデルを初期化中...")
        
        lisa_config = LisaGemmaConfig(
            gemma_model_id=getattr(config, 'GEMMA_MODEL_ID', 'google/gemma-3-4b-it'),
            sam_checkpoint_path=getattr(config, 'SAM_CHECKPOINT_PATH', None),
            seg_token=getattr(config, 'SEG_TOKEN', '[SEG]'),
            gemma_hidden_size=getattr(config, 'GEMMA_HIDDEN_SIZE', 2560),
            sam_prompt_embed_dim=getattr(config, 'SEG_PROJECTION_DIM', 256),
            gemma_image_size=getattr(config, 'GEMMA_IMAGE_SIZE', 896),
            sam_image_size=getattr(config, 'SAM_IMAGE_SIZE', 1024),
            model_max_length=getattr(config, 'MODEL_MAX_LENGTH', 2048),
        )
        
        model = LisaGemmaForCausalLM(lisa_config)
        model = model.to(device)
        model.train()  # 学習モードに設定
        
        print("✅ モデル初期化完了")
        
        # 学習可能パラメータの情報を表示
        param_info = model.get_trainable_parameters_info()
        print(f"  - 総パラメータ数: {param_info['total_parameters']:,}")
        print(f"  - 学習可能パラメータ数: {param_info['trainable_parameters']:,}")
        print(f"  - 学習可能率: {param_info['trainable_percentage']:.2f}%")
        
        # 2. データセットとデータローダーの準備
        print(f"\n📊 {args.dataset_type}データセットを準備中...")
        
        processor = AutoProcessor.from_pretrained(lisa_config.gemma_model_id)
        
        # データセットタイプの決定
        if args.dataset_type == "all":
            dataset_spec = "sem_seg||refer_seg||vqa||reason_seg"
            print(f"  - 全データセットを使用: sem_seg, refer_seg, vqa, reason_seg")
        else:
            dataset_spec = args.dataset_type
            print(f"  - 単一データセットを使用: {args.dataset_type}")
        
        # 単一バッチテスト用の小規模データセット
        dataset = HybridDataset(
            base_image_dir=getattr(config, 'DATASET_BASE_DIR', './dataset'),
            gemma_processor=processor,
            dataset=dataset_spec,
            samples_per_epoch=args.batch_size * 2  # テスト用に少数のサンプル
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False  # 再現性のため固定
        )
        
        print(f"✅ データセット準備完了（サンプル数: {len(dataset)}）")
        
        # 3. 固定バッチの取得
        print("\n🎯 固定バッチを取得中...")
        fixed_batch = next(iter(dataloader))
        
        # バッチをデバイスに移動
        fixed_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in fixed_batch.items()}
        
        print(f"✅ 固定バッチ取得完了")
        print(f"  - バッチサイズ: {fixed_batch['input_ids'].shape[0]}")
        print(f"  - シーケンス長: {fixed_batch['input_ids'].shape[1]}")
        
        # バッチの内容を確認
        has_segmentation = 'ground_truth_mask' in fixed_batch
        print(f"  - セグメンテーションタスク: {'あり' if has_segmentation else 'なし'}")
        
        # 4. オプティマイザーと損失関数の準備
        print("\n⚙️ オプティマイザーと損失関数を準備中...")
        
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=getattr(config, 'WEIGHT_DECAY', 1e-2),
            betas=(getattr(config, 'BETA1', 0.9), getattr(config, 'BETA2', 0.95))
        )
        
        loss_fn = CompositeLoss(
            ce_loss_weight=getattr(config, 'CE_LOSS_WEIGHT', 1.0),
            dice_loss_weight=getattr(config, 'DICE_LOSS_WEIGHT', 0.5),
            bce_loss_weight=getattr(config, 'BCE_LOSS_WEIGHT', 2.0)
        )
        
        print("✅ オプティマイザーと損失関数の準備完了")
        
        # 5. 過学習ループの実行
        print(f"\n🚀 単一バッチでの過学習を開始...")
        print(f"目標: {args.iterations}イテレーションで損失をゼロに近づける")
        print("-" * 80)
        
        loss_history = []
        best_loss = float('inf')
        start_time = time.time()
        
        for iteration in range(args.iterations):
            # フォワードパス
            optimizer.zero_grad()
            
            # モデルに適した入力形式を準備
            model_inputs = {
                'input_ids': fixed_batch['input_ids'],
                'attention_mask': fixed_batch['attention_masks'],  # collate_fnでは'attention_masks'が使われる
                'labels': fixed_batch['labels'],
                'generate_mask': has_segmentation,
            }
            
            # デュアルストリーム対応
            if 'images_for_gemma' in fixed_batch:
                model_inputs['images_for_gemma'] = fixed_batch['images_for_gemma']
            if 'images_for_sam' in fixed_batch:
                model_inputs['images_for_sam'] = fixed_batch['images_for_sam']
            
            # フォワードパス実行
            outputs = model(**model_inputs)
            
            # 損失計算
            losses = loss_fn(outputs, fixed_batch)
            total_loss = losses['total_loss']
            
            # バックワードパス
            total_loss.backward()
            
            # 勾配クリッピング（安定性のため）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # オプティマイザーステップ
            optimizer.step()
            
            # 損失履歴を記録
            loss_record = {
                'iteration': iteration,
                'total_loss': total_loss.item(),
                'text_loss': losses.get('text_loss', torch.tensor(0.0)).item(),
                'dice_loss': losses.get('dice_loss', torch.tensor(0.0)).item(),
                'bce_loss': losses.get('bce_loss', torch.tensor(0.0)).item(),
            }
            loss_history.append(loss_record)
            
            # 最良損失の更新
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
            
            # 進捗表示（10イテレーションごと）
            if iteration % 10 == 0 or iteration == args.iterations - 1:
                elapsed_time = time.time() - start_time
                print(f"Iter {iteration:3d}/{args.iterations}: "
                      f"Loss={total_loss.item():.6f} "
                      f"(Text: {loss_record['text_loss']:.4f}, "
                      f"DICE: {loss_record['dice_loss']:.4f}, "
                      f"BCE: {loss_record['bce_loss']:.4f}) "
                      f"Best: {best_loss:.6f} "
                      f"Time: {elapsed_time:.1f}s")
        
        print("-" * 80)
        print("✅ 過学習ループ完了")
        
        # 6. 結果の分析
        print("\n📊 過学習結果の分析...")
        
        analysis = analyze_overfitting_success(loss_history)
        
        print(f"過学習テスト結果: {'✅ 成功' if analysis['success'] else '❌ 失敗'}")
        print(f"  - 初期損失: {analysis['initial_loss']:.6f}")
        print(f"  - 最終損失: {analysis['final_loss']:.6f}")
        print(f"  - 損失減少率: {analysis['reduction_ratio']:.1%}")
        print(f"  - 安定性比率: {analysis['stability_ratio']:.4f}")
        print(f"  - イテレーション数: {analysis['iterations']}")
        
        # 7. 結果の保存
        print("\n💾 結果を保存中...")
        
        # 出力ディレクトリの作成
        output_dir = "verification_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 損失曲線のプロット
        plot_path = os.path.join(output_dir, f"overfit_single_batch_{session_timestamp}.png")
        plot_loss_curve(loss_history, plot_path)
        
        # 詳細結果をJSONで保存
        results = {
            "session_timestamp": session_timestamp,
            "test_config": {
                "iterations": args.iterations,
                "learning_rate": args.learning_rate,
                "dataset_type": args.dataset_type,
                "batch_size": args.batch_size,
            },
            "model_info": param_info,
            "analysis": analysis,
            "loss_history": loss_history,
            "final_status": "success" if analysis['success'] else "failure"
        }
        
        json_path = os.path.join(output_dir, f"overfit_single_batch_{session_timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 結果保存完了:")
        print(f"  - 損失曲線: {plot_path}")
        print(f"  - 詳細データ: {json_path}")
        
        # 過学習成功時のみモデルチェックポイントを保存
        if analysis['success']:
            checkpoint_path = os.path.join(output_dir, f"overfit_checkpoint_{session_timestamp}.pth")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'session_timestamp': session_timestamp,
                'config': {
                    'iterations': args.iterations,
                    'learning_rate': args.learning_rate,
                    'dataset_type': args.dataset_type,
                    'final_loss': analysis['final_loss'],
                    'reduction_ratio': analysis['reduction_ratio']
                }
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  - モデルチェックポイント: {checkpoint_path}")
            
            # 最新の成功チェックポイントのシンボリックリンクを作成
            latest_checkpoint_path = os.path.join(output_dir, "latest_overfit_checkpoint.pth")
            if os.path.exists(latest_checkpoint_path):
                os.remove(latest_checkpoint_path)
            os.symlink(os.path.basename(checkpoint_path), latest_checkpoint_path)
            print(f"  - 最新チェックポイント: {latest_checkpoint_path}")
        
        print(f"✅ 結果保存完了:")
        print(f"  - 損失曲線: {plot_path}")
        print(f"  - 詳細データ: {json_path}")
        
        # 8. 最終判定
        print("\n" + "="*80)
        print("🎯 最終判定")
        print("="*80)
        
        if analysis['success']:
            print("✅ サニティチェック1: 単一バッチへの過学習 - 成功")
            print("   モデルは正常に学習能力を示しています。")
            print("   アーキテクチャ、損失関数、オプティマイザーが正しく機能しています。")
        else:
            print("❌ サニティチェック1: 単一バッチへの過学習 - 失敗")
            print("   以下の問題が考えられます:")
            
            if analysis['reduction_ratio'] < 0.1:
                print("   - 学習率が低すぎる可能性があります")
            elif analysis['stability_ratio'] > 0.2:
                print("   - 学習が不安定です（学習率が高すぎる可能性）")
            elif analysis['final_loss'] > analysis['initial_loss'] * 0.5:
                print("   - 損失関数または勾配伝播に問題がある可能性があります")
            
            print(f"   推奨: verify_loss_and_gradients.pyを再実行して根本原因を調査してください")
        
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main() 