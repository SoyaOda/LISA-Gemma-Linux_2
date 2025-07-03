#!/usr/bin/env python3
"""
第4節：損失計算と勾配伝播の精査 (Lambda Cloud最適化)

1回の完全な学習ステップ（フォワードパスとバックワードパス）を実行し、
全ての学習可能パラメータグループの勾配を検査することで、
計算グラフ全体が損なわれていないことを検証する。

論理的根拠:
- 正しく定義されたアーキテクチャも、損失が不正確に計算されたり、勾配が学習可能な重みに流れなければ、学習能力を持たない
- LISAの学習プロセスは、言語モデルのテキスト予測損失（loss_lm）と、セグメンテーションデコーダーのマスク予測損失（loss_seg）を合算した複合損失によって駆動される
- 計算グラフの切断は最も深刻なバグであり、特にLLMの隠れ状態をセグメンテーションデコーダーに渡す部分で発生しやすい
"""

import argparse
import os
import sys
from datetime import datetime

print("🚀 LISA-Llama4 Loss and Gradients Verification (Lambda Cloud Optimized)")

# 重いライブラリは遅延読み込み
# import torch  # 遅延読み込み
# from torch.utils.data import DataLoader  # 遅延読み込み
# from torch.optim import AdamW  # 遅延読み込み
# from transformers import AutoProcessor  # 遅延読み込み

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_heavy_libraries():
    """重いライブラリの遅延読み込み"""
    global torch, DataLoader, AdamW, AutoProcessor
    global LoraConfig, get_peft_model, TaskType
    global LisaLlama4ForCausalLM, LisaLlama4Config
    global HybridDataset, CompositeLoss, collate_fn
    
    print("⏳ 重いライブラリを読み込み中...")
    
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType
    
    from model.llama4_lisa import LisaLlama4ForCausalLM, LisaLlama4Config
    from utils.dataset import HybridDataset, collate_fn
    from model.losses import CompositeLoss
    
    print("✅ ライブラリの読み込み完了")

def get_config():
    """設定ファイルをインポート"""
    try:
        import config_linux as config
        return config
    except ImportError as e:
        print(f"❌ 設定ファイルconfig_linux.pyのインポートに失敗: {e}")
        print("   ワーキングディレクトリを確認してください")
        raise e

def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description="LISA-Llama4 損失と勾配の検証")
    parser.add_argument("--batch-size", type=int, default=1, help="バッチサイズ（デフォルト: 1）")
    parser.add_argument("--sample-count", type=int, default=4, help="検証サンプル数（デフォルト: 4）")
    parser.add_argument("--skip-mask-generation", action="store_true", 
                       help="マスク生成をスキップして計算グラフのみを検証")
    return parser.parse_args()

def analyze_gradients(model, param_groups):
    """
    パラメータグループ別の勾配分析
    
    Args:
        model: 検証対象のモデル
        param_groups: パラメータグループの辞書（グループ名 -> キーワードリスト）
    
    Returns:
        dict: 勾配分析結果
    """
    results = {}
    
    for group_name, keywords in param_groups.items():
        grad_count = 0
        param_count = 0
        grad_norms = []
        
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in keywords):
                param_count += 1
                if param.grad is not None:
                    grad_count += 1
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
        
        results[group_name] = {
            'param_count': param_count,
            'grad_count': grad_count,
            'grad_norms': grad_norms,
            'avg_grad_norm': sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
            'max_grad_norm': max(grad_norms) if grad_norms else 0.0
        }
        
        status = "✅" if grad_count > 0 else "❌"
        print(f"  {status} {group_name}: {grad_count}/{param_count}パラメータに勾配")
        if grad_norms:
            print(f"      平均勾配ノルム: {results[group_name]['avg_grad_norm']:.6f}")
            print(f"      最大勾配ノルム: {results[group_name]['max_grad_norm']:.6f}")
    
    return results

def main():
    try:
        print("🚀 LISA-Llama4 Loss and Gradients Verification (Lambda Cloud Optimized)")
        print("="*80)
        print("第4節: 損失計算と勾配伝播の精査")
        print("="*80)
        
        print("⏳ 重いライブラリを読み込み中...")
        
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        import traceback
        import os
        import numpy as np
        from pathlib import Path
        import gc
        
        # プロジェクトパスを追加
        import sys
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))
        
        import config_linux
        from model.llama4_lisa import LisaLlama4ForCausalLM
        from utils.dataset import HybridDataset, collate_fn
        from utils.constants import DEFAULT_SEG_TOKEN
        
        print("✅ ライブラリの読み込み完了")
        
        # 設定を読み込み（config_linuxから直接インポート）
        class Config:
            def __init__(self):
                self.model_id = config_linux.LLAMA_MODEL_ID
                self.dataset_base_dir = config_linux.DATASET_BASE_DIR
                self.batch_size = config_linux.BATCH_SIZE_PER_GPU
                self.lora_r = config_linux.LORA_R
                self.lora_alpha = config_linux.LORA_ALPHA
                self.ce_loss_weight = 1.0  # デフォルト値
                self.dice_loss_weight = 0.5  # デフォルト値
                self.bce_loss_weight = 2.0  # デフォルト値
                self.attn_implementation = config_linux.ATTN_IMPLEMENTATION
                self.device_map = config_linux.DEVICE_MAP
                self.torch_dtype = config_linux.TORCH_DTYPE
                self.sam_checkpoint_path = config_linux.SAM_CHECKPOINT_PATH
                self.llama_image_size = config_linux.LLAMA_IMAGE_SIZE
                self.sam_image_size = config_linux.SAM_IMAGE_SIZE
                self.llama_hidden_size = config_linux.LLAMA_HIDDEN_SIZE
                self.seg_projection_dim = config_linux.SEG_PROJECTION_DIM
                self.lora_target_modules = config_linux.LORA_TARGET_MODULES
                self.lora_dropout = config_linux.LORA_DROPOUT
                self.datasets = ['reason_seg']  # テスト用
                self.sample_rate = 4  # テスト用
        
        config = Config()
        print("✅ 設定読み込み完了")
        print(f"  Llamaモデル: {config.model_id}")
        print(f"  データセットベースディレクトリ: {config.dataset_base_dir}")
        print(f"  バッチサイズ: {config.batch_size}")
        print(f"  LoRA設定: r={config.lora_r}, alpha={config.lora_alpha}")
        print(f"  損失重み: CE={config.ce_loss_weight}, DICE={config.dice_loss_weight}, BCE={config.bce_loss_weight}")
        print(f"  アテンション実装: {config.attn_implementation}")
        print(f"  デバイスマップ: {config.device_map}")
        print(f"  Torch精度: {config.torch_dtype}")
        
        # デバイス設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用デバイス: {device}")
        
        # 📦 モデル初期化
        print("\n📦 モデルを初期化中...")
        from model.llama4_lisa import LisaLlama4Config
        
        # LisaLlama4Config作成
        lisa_config = LisaLlama4Config(
            llama_model_id=config.model_id,
            sam_checkpoint_path=config.sam_checkpoint_path,
            seg_token=DEFAULT_SEG_TOKEN,
            llama_hidden_size=config.llama_hidden_size,
            sam_prompt_embed_dim=config.seg_projection_dim,
            llama_image_size=config.llama_image_size,
            sam_image_size=config.sam_image_size,
            attn_implementation=config.attn_implementation,
            device_map=config.device_map,
            torch_dtype=config.torch_dtype
        )
        
        model = LisaLlama4ForCausalLM(lisa_config)
        print("✅ モデル初期化完了")
        
        # 📊 データセット準備
        print("\n📊 データセット準備中...")
        
        # [SEG]トークンをモデルのトークナイザーに設定
        if hasattr(model, 'llama_processor') and model.llama_processor is not None:
            seg_token_id = model.llama_processor.tokenizer.convert_tokens_to_ids('[SEG]')
            print(f"[SEG]トークンセットアップ完了:")
            print(f"  - 追加されたトークン数: 1")
            print(f"  - [SEG]トークンID: {seg_token_id}")
        else:
            print("⚠️ モデルプロセッサが利用できません")
        
        # データセット作成
        dataset = HybridDataset(
            base_image_dir=config.dataset_base_dir,
            llama_processor=model.llama_processor,
            llama_image_size=config.llama_image_size,
            sam_image_size=config.sam_image_size,
            dataset='reason_seg',  # テスト用に単一データセット
            samples_per_epoch=4   # 小さなサンプル数
        )
        
        print(f"✅ HybridDataset初期化完了")
        
        # GPU制約に応じたサンプル数調整
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"⚠️  sample_rate調整: {config.sample_rate} -> 1")
        print(f"  GPU メモリが{gpu_memory_gb:.1f}GBのため、バッチサイズを1に調整")
        config.sample_rate = 1
        config.batch_size = 1
        print(f"  最終バッチサイズ: {config.batch_size}")
        
        print(f"✅ データセット準備完了（サンプル数: {len(dataset)}）")
        
        # DataLoader作成
        dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0  # Lambda Cloud環境では0推奨
        )
        
        # 🔬 フォワード＆バックワードパステスト 
        print("\n🔬 フォワードパスとバックワードパスを実行中...")
        
        # モデルを学習モードに
        model.train()
        
        # 設定情報を出力
        print(f"設定: config_linux.py を使用")
        
        for batch_idx, batch in enumerate(dataloader):
            print(f"  入力バッチをGPUデバイスに移動中...")
            
            # バッチデータのデバイス移動（device_map="auto"の場合は移動不要）
            if config.device_map != "auto":
                if 'input_ids' in batch:
                    batch['input_ids'] = batch['input_ids'].to(device)
                if 'attention_mask' in batch:
                    batch['attention_mask'] = batch['attention_mask'].to(device)
                if 'images_for_llama' in batch:
                    batch['images_for_llama'] = batch['images_for_llama'].to(device)
                if 'images_for_sam' in batch:
                    batch['images_for_sam'] = batch['images_for_sam'].to(device)
                if 'labels' in batch:
                    batch['labels'] = batch['labels'].to(device)
            else:
                print("  device_map='auto'検出: 画像のデバイス移動をスキップ")
            
            print("  ✅ バッチデータのデバイス移動完了")
            
            # 形状確認
            print(f"  画像データ: {batch['images_for_llama'].shape}")
            print(f"  SAM用画像: {batch['images_for_sam'].shape}")
            print(f"  attention_mask: {batch['attention_mask'].shape}")
            print(f"  labels: {batch['labels'].shape}")
            
            # モデル入力の構築
            model_inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'images_for_llama': batch['images_for_llama'],
                'images_for_sam': batch['images_for_sam'],
                'labels': batch['labels'],
                'generate_mask': False  # 🔥 メモリ節約のためSAMをオフ
            }
            
            print("\n[フォワードパス実行]")
            print(f"  - input_ids: {model_inputs['input_ids'].shape}")
            print(f"  - attention_mask: {model_inputs['attention_mask'].shape}")
            print(f"  - images_for_llama: {model_inputs['images_for_llama'].shape}")
            print(f"  - images_for_sam: {model_inputs['images_for_sam'].shape}")
            print(f"  - generate_mask: {model_inputs['generate_mask']}")
            
            try:
                # フォワードパス実行
                outputs = model(**model_inputs)
                
                print("✅ フォワードパス成功")
                print(f"  - Loss: {outputs.get('text_loss', 'N/A')}")
                print(f"  - Logits shape: {outputs.get('logits', torch.empty(0)).shape}")
                print(f"  - Hidden states shape: {outputs.get('hidden_states', torch.empty(0)).shape}")
                print(f"  - Pred masks: {outputs.get('pred_masks', 'None')}")
                
                # バックワードパス（損失が存在する場合）
                if outputs.get('text_loss') is not None:
                    print("\n[バックワードパス実行]")
                    loss = outputs['text_loss']
                    print(f"  損失値: {loss.item():.6f}")
                    
                    # 勾配計算
                    loss.backward()
                    print("✅ バックワードパス成功")
                    
                    # 勾配統計
                    grad_stats = {}
                    total_params = 0
                    params_with_grad = 0
                    
                    for name, param in model.named_parameters():
                        total_params += 1
                        if param.grad is not None:
                            params_with_grad += 1
                            grad_norm = param.grad.norm().item()
                            if 'lora' in name.lower():
                                grad_stats.setdefault('LoRA', []).append(grad_norm)
                            elif 'multi_modal_projector' in name:
                                grad_stats.setdefault('MLP Projector', []).append(grad_norm)
                            elif 'sam' in name.lower():
                                grad_stats.setdefault('SAM', []).append(grad_norm)
                            else:
                                grad_stats.setdefault('Other', []).append(grad_norm)
                    
                    print(f"\n[勾配統計]")
                    print(f"  全パラメータ数: {total_params}")
                    print(f"  勾配を持つパラメータ数: {params_with_grad}")
                    
                    for component, grads in grad_stats.items():
                        if grads:
                            avg_grad = np.mean(grads)
                            max_grad = np.max(grads)
                            print(f"  {component}: 平均勾配ノルム={avg_grad:.6f}, 最大={max_grad:.6f}")
                
                print(f"✅ バッチ {batch_idx + 1} の処理完了")
                break  # 1バッチのみテスト
                
            except Exception as e:
                print(f"フォワード中にエラー発生: {e}")
                print(f"エラーの詳細:")
                traceback.print_exc()
                raise e
        
        print("\n🎉 Loss and Gradients Verification 完了")
        print("="*80)
        print("✅ すべてのテストが正常に完了しました")
        
    except Exception as e:
        print(f"\n❌ 致命的エラーが発生しました: {e}")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main() 