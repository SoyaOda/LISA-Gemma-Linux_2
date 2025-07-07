#!/usr/bin/env python3
"""
Llama4-LISA 損失計算と勾配伝播の精査
===============================

DeepSpeed ZeRO Stage 2環境でのLlama4-LISAモデルの
損失計算と勾配伝播を検証するスクリプト

検証項目:
1. モデルの正常な初期化
2. 学習可能パラメータの設定確認
3. フォワードパス実行
4. 損失計算の正常性
5. バックワードパス実行
6. 勾配伝播の確認
"""

import argparse
import os
import sys
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import traceback
import numpy as np
from pathlib import Path
import gc
import json

print("🚀 Llama4-LISA Loss and Gradients Verification")

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
        
        if gpu_count >= 8:
            print("  ✅ A100*8環境確認完了")
        else:
            print(f"  ⚠️  予期したGPU数(8)より少ない: {gpu_count}")
    else:
        print("  ❌ CUDA環境が利用できません")

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
        
        # FlashAttention有効化（PyTorch 2.0+）
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            print("  ✅ FlashAttention有効化")

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

def get_memory_usage():
    """メモリ使用量の取得"""
    memory_info = {}
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            memory_info[f'gpu_{i}_allocated'] = allocated
            memory_info[f'gpu_{i}_reserved'] = reserved
    
    return memory_info

def print_memory_usage(prefix=""):
    """メモリ使用量を表示"""
    if torch.cuda.is_available():
        print(f"\n{prefix}📊 メモリ使用量:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {allocated:.2f}GB 使用中, {reserved:.2f}GB 予約済み / {total:.2f}GB")

def create_dummy_batch():
    """ダミーバッチデータの作成"""
    print("\n🎯 ダミーバッチデータ作成中...")
    
    # テキストデータ（セグメンテーションタスク用）
    input_ids = torch.randint(1, 1000, (1, 128))  # バッチサイズ1、シーケンス長128
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    # 画像データ（Llama4のマルチモーダル入力）
    # Llama4は448x448の画像を期待
    pixel_values = torch.randn(1, 3, 448, 448)
    
    # SAM用の画像データ（1024x1024）
    sam_images = torch.randn(1, 3, 1024, 1024)
    
    # セグメンテーションマスク
    sam_masks = torch.randint(0, 2, (1, 1, 256, 256)).float()
    
    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'labels': labels,
        'sam_images': sam_images,
        'sam_masks': sam_masks
    }
    
    print("✅ ダミーバッチデータ作成完了")
    print(f"  - input_ids: {input_ids.shape}")
    print(f"  - pixel_values: {pixel_values.shape}")
    print(f"  - sam_images: {sam_images.shape}")
    print(f"  - sam_masks: {sam_masks.shape}")
    
    return batch

def main():
    try:
        print("🚀 Llama4-LISA Loss and Gradients Verification")
        print("="*80)
        print("Llama4専用：損失計算と勾配伝播の精査")
        print("="*80)
        
        # メモリ最適化設定を最初に適用
        setup_memory_optimizations()
        
        # GPU環境確認
        check_gpu_environment()
        
        # プロジェクトパスを追加
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))
        
        print_memory_usage("初期")
        
        # train_llama4_deepspeed.pyからモデルクラスをインポート
        from train_llama4_deepspeed import (
            Llama4LisaDeepSpeedConfig, 
            Llama4LisaDeepSpeedModel,
            SAMProjector
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
        
        print_memory_usage("モデル読み込み後")
        
        # ===== Step 2: 学習可能パラメータの確認 =====
        print("\n" + "="*60)
        print("Step 2: 学習可能パラメータの確認")
        print("="*60)
        
        # パラメータグループの定義
        param_groups = {
            "LoRA_adapters": ["lora_A", "lora_B"],
            "SAM_projector": ["sam_projector"],
            "SAM_mask_decoder": ["mask_decoder"],
            "Llama4_base": ["language_model", "vision"],
            "SAM_encoders": ["image_encoder", "prompt_encoder"]
        }
        
        # パラメータ統計
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100
        
        print(f"📊 パラメータ統計:")
        print(f"  - 総パラメータ数: {total_params:,}")
        print(f"  - 学習可能パラメータ数: {trainable_params:,}")
        print(f"  - 学習可能率: {trainable_percentage:.2f}%")
        
        # パラメータグループ別確認
        print(f"\n📋 パラメータグループ別確認:")
        for group_name, keywords in param_groups.items():
            group_params = 0
            trainable_group_params = 0
            
            for name, param in model.named_parameters():
                if any(keyword in name for keyword in keywords):
                    group_params += param.numel()
                    if param.requires_grad:
                        trainable_group_params += param.numel()
            
            if group_params > 0:
                trainable_ratio = (trainable_group_params / group_params) * 100
                status = "✅" if trainable_group_params > 0 else "❌"
                print(f"  {status} {group_name}: {trainable_group_params:,}/{group_params:,} ({trainable_ratio:.1f}%)")
        
        # ===== Step 3: ダミーデータ作成 =====
        print("\n" + "="*60)
        print("Step 3: ダミーデータ作成")
        print("="*60)
        
        batch = create_dummy_batch()
        
        # データをGPUに移動
        if torch.cuda.is_available():
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            print("✅ データをGPUに移動完了")
        
        print_memory_usage("データ準備後")
        
        # ===== Step 4: フォワードパス実行 =====
        print("\n" + "="*60)
        print("Step 4: フォワードパス実行")
        print("="*60)
        
        print("🔄 フォワードパス実行中...")
        model.train()  # 学習モードに設定
        
        try:
            # 勾配をクリア
            model.zero_grad()
            
            # フォワードパス実行
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
                labels=batch['labels']
            )
            
            print("✅ フォワードパス実行完了")
            print(f"  - 出力形状: {outputs.logits.shape if hasattr(outputs, 'logits') else 'N/A'}")
            print(f"  - 損失: {outputs.loss.item() if hasattr(outputs, 'loss') else 'N/A'}")
            
        except Exception as e:
            print(f"❌ フォワードパス実行エラー: {e}")
            traceback.print_exc()
            return
        
        print_memory_usage("フォワードパス後")
        
        # ===== Step 5: バックワードパス実行 =====
        print("\n" + "="*60)
        print("Step 5: バックワードパス実行")
        print("="*60)
        
        print("🔄 バックワードパス実行中...")
        
        try:
            # 損失から勾配計算
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
                loss.backward()
                print("✅ バックワードパス実行完了")
                print(f"  - 損失値: {loss.item():.6f}")
            else:
                print("❌ 損失が出力に含まれていません")
                return
            
        except Exception as e:
            print(f"❌ バックワードパス実行エラー: {e}")
            traceback.print_exc()
            return
        
        print_memory_usage("バックワードパス後")
        
        # ===== Step 6: 勾配確認 =====
        print("\n" + "="*60)
        print("Step 6: 勾配伝播の確認")
        print("="*60)
        
        print("📊 パラメータグループ別勾配分析:")
        gradient_results = analyze_gradients(model, param_groups)
        
        # 勾配統計サマリー
        print(f"\n📈 勾配統計サマリー:")
        total_grads = sum(result['grad_count'] for result in gradient_results.values())
        total_params_with_grad_requirement = sum(result['param_count'] for result in gradient_results.values() 
                                               if any(keyword in ["lora", "sam_projector", "mask_decoder"] 
                                                     for keyword in param_groups[group_name]) 
                                               for group_name, result in gradient_results.items())
        
        print(f"  - 勾配を持つパラメータ数: {total_grads}")
        print(f"  - 学習可能パラメータ数: {trainable_params:,}")
        
        # ===== Step 7: 結果サマリー =====
        print("\n" + "="*60)
        print("Step 7: 検証結果サマリー")
        print("="*60)
        
        # 成功条件チェック
        success_criteria = {
            "model_initialization": True,
            "forward_pass": hasattr(outputs, 'logits'),
            "loss_calculation": hasattr(outputs, 'loss'),
            "backward_pass": hasattr(outputs, 'loss'),
            "gradient_propagation": total_grads > 0,
            "lora_gradients": gradient_results.get("LoRA_adapters", {}).get('grad_count', 0) > 0,
            "sam_projector_gradients": gradient_results.get("SAM_projector", {}).get('grad_count', 0) > 0
        }
        
        print("🎯 検証結果:")
        for criterion, success in success_criteria.items():
            status = "✅" if success else "❌"
            print(f"  {status} {criterion}")
        
        overall_success = all(success_criteria.values())
        
        if overall_success:
            print("\n🎉 全ての検証項目が成功しました！")
            print("   Llama4-LISAモデルは学習準備完了です。")
        else:
            print("\n⚠️  一部の検証項目で問題が検出されました。")
            print("   問題を解決してから学習を開始してください。")
        
        # 結果をJSONで保存
        results = {
            "timestamp": datetime.now().isoformat(),
            "success_criteria": success_criteria,
            "overall_success": overall_success,
            "parameter_stats": {
                "total_parameters": int(total_params),
                "trainable_parameters": int(trainable_params),
                "trainable_percentage": float(trainable_percentage)
            },
            "gradient_results": {
                k: {
                    "param_count": v["param_count"],
                    "grad_count": v["grad_count"],
                    "avg_grad_norm": v["avg_grad_norm"],
                    "max_grad_norm": v["max_grad_norm"]
                } for k, v in gradient_results.items()
            }
        }
        
        output_file = f"llama4_verification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 検証結果を保存: {output_file}")
        
        print_memory_usage("最終")
        
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