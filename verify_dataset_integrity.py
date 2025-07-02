#!/usr/bin/env python3
"""
LISA-Llama4 Dataset Integrity Verification
Gemma版成功パターン準拠 - 個別データセットクラス使用

個別データセットクラスを使用してデータの整合性を検証します。
HybridDatasetに依存せず、プロセッサー非依存の検証を行います。
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import json

print("🚀 LISA-Llama4 Dataset Integrity Verification (Individual Dataset Pattern)")
print("📦 基本ライブラリ読み込み完了")

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_config():
    """
    config_llama4.pyを読み込む
    """
    try:
        from config_llama4 import create_config
        config = create_config("default")
        print(f"✅ 設定ファイルを読み込み: config_llama4.py")
        return config
    except ImportError as e:
        print(f"❌ ERROR: config_llama4.pyが見つかりません")
        print(f"   詳細: {e}")
        print(f"   現在のディレクトリ: {os.getcwd()}")
        raise SystemExit("config_llama4.pyが必須です。")

def parse_args():
    parser = argparse.ArgumentParser(description="Verify Individual Dataset Integrity")
    parser.add_argument(
        "--datasets", 
        nargs='+', 
        default=["sem_seg", "refer_seg", "vqa", "reason_seg"],
        help="List of sub-dataset names to inspect"
    )
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to inspect per dataset")
    return parser.parse_args()

def load_heavy_libraries():
    """重いライブラリを必要時に読み込む"""
    print("📦 重いライブラリを読み込み中...")
    global torch, np, plt, Image, ToPILImage
    
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # バックエンドを非対話型に設定
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision.transforms import ToPILImage
    
    print("✅ PyTorch, NumPy, Matplotlib読み込み完了")

def save_comparison_image(image_tensor, mask_tensor, dataset_name, sample_idx, 
                         image_path, conversations, questions, class_names, session_timestamp):
    """比較画像とメタデータを保存"""
    try:
        item_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_filename = f"{dataset_name}_{sample_idx}_{item_timestamp}"
        
        # セッション用のサブディレクトリ作成
        output_dir = os.path.join("verification_output", f"llama4_individual_{session_timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # メタデータ保存
        metadata = {
            "dataset_name": dataset_name,
            "sample_idx": sample_idx,
            "session_timestamp": session_timestamp,
            "item_timestamp": item_timestamp,
            "original_image_path": image_path,
            "conversations": conversations,
            "questions": questions,
            "class_names": class_names,
            "processing_note": "Llama4 version - individual dataset verification",
        }
        
        metadata_filename = os.path.join(output_dir, f"{base_filename}_metadata.json")
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"💾 メタデータ保存: {metadata_filename}")
        return None, metadata_filename
        
    except Exception as e:
        print(f"保存エラー: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    args = parse_args()
    config = get_config()
    
    # セッションタイムスタンプの生成
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n🔍 LISA-Llama4 Individual Dataset Integrity Verification")
    print("=" * 70)
    print(f"🚀 検証セッション開始: {session_timestamp}")
    print(f"📁 出力ディレクトリ: verification_output/llama4_individual_{session_timestamp}/")
    print("=" * 70)
    
    # 重いライブラリを必要時に読み込み
    load_heavy_libraries()
    
    # 個別データセットクラスを使用した検証（Gemma版パターン）
    try:
        print("📦 データセットクラスを読み込み中...")
        from utils.sem_seg_dataset import SemSegDataset
        from utils.refer_seg_dataset import ReferSegDataset
        from utils.vqa_dataset import VQADataset
        from utils.reason_seg_dataset import ReasonSegDataset
        
        print("✅ データセットクラス読み込み完了")
        print(f"📦 個別データセット初期化中...")
        
        # データセット設定（config_llama4.pyから統一管理）
        print(f"📋 Llama4設定情報:")
        print(f"   データセットベースディレクトリ: {config.DATASET_BASE_DIR}")
        print(f"   SAMチェックポイント: {config.SAM_CHECKPOINT_PATH}")
        print(f"   バッチサイズ: {config.BATCH_SIZE}")
        print(f"   勾配蓄積ステップ: {config.GRADIENT_ACCUMULATION_STEPS}")
        
        dataset_configs = []
        if "sem_seg" in args.datasets:
            dataset_configs.append(("sem_seg", SemSegDataset, {
                "base_image_dir": config.DATASET_BASE_DIR,
                "tokenizer": None,  # プロセッサー非依存
                "samples_per_epoch": config.SAMPLES_PER_EPOCH,
                "sem_seg_data": "ade20k||cocostuff||mapillary||pascal_part||paco_lvis"
            }))
        
        if "refer_seg" in args.datasets:
            dataset_configs.append(("refer_seg", ReferSegDataset, {
                "base_image_dir": config.DATASET_BASE_DIR,
                "tokenizer": None,
                "samples_per_epoch": config.SAMPLES_PER_EPOCH,
                "refer_seg_data": "refclef||refcoco||refcoco+||refcocog"
            }))
        
        if "vqa" in args.datasets:
            dataset_configs.append(("vqa", VQADataset, {
                "base_image_dir": config.DATASET_BASE_DIR,
                "tokenizer": None,
                "samples_per_epoch": config.SAMPLES_PER_EPOCH,
                "vqa_data": "llava_instruct_150k"
            }))
        
        if "reason_seg" in args.datasets:
            dataset_configs.append(("reason_seg", ReasonSegDataset, {
                "base_image_dir": config.DATASET_BASE_DIR,
                "tokenizer": None,
                "samples_per_epoch": config.SAMPLES_PER_EPOCH,
                "reason_seg_data": "ReasonSeg|train"
            }))
        
        # 各データセットを初期化して検証
        print("\n🔬 個別データセット検証開始")
        successful_datasets = 0
        total_samples_processed = 0
        
        for dataset_name, dataset_class, dataset_kwargs in dataset_configs:
            print(f"\n===== {dataset_name.upper()} データセット検証 =====")
            
            try:
                # データセット初期化
                dataset = dataset_class(**dataset_kwargs)
                print(f"✅ {dataset_name} 初期化完了 (サンプル数: {len(dataset)})")
                
                if len(dataset) == 0:
                    print(f"⚠️ {dataset_name} データセットが空です")
                    continue
                
                successful_datasets += 1
                
                # サンプル検証
                for i in range(min(args.num_samples, len(dataset))):
                    print(f"\n--- Sample {i+1}/{args.num_samples} ---")
                    
                    try:
                        sample = dataset[i]
                        total_samples_processed += 1
                        
                        # データ構造確認
                        if isinstance(sample, dict):
                            print(f"📦 辞書形式: {list(sample.keys())}")
                            
                            # 会話データ
                            if 'conversations' in sample:
                                conversations = sample['conversations']
                                print(f"💬 会話ターン数: {len(conversations)}")
                                for j, turn in enumerate(conversations[:2]):
                                    if isinstance(turn, dict):
                                        print(f"  Turn {j+1}: {turn.get('from')} -> {turn.get('value', '')[:100]}...")
                            
                            # 画像データ
                            image_tensor = sample.get('image') or sample.get('images')
                            if image_tensor is not None:
                                print(f"🖼️ 画像: {image_tensor.shape}, {image_tensor.dtype}")
                            
                            # マスクデータ
                            mask_tensor = sample.get('masks') or sample.get('mask')
                            if mask_tensor is not None:
                                if isinstance(mask_tensor, list):
                                    print(f"🎯 マスク: リスト({len(mask_tensor)}個)")
                                    if len(mask_tensor) > 0:
                                        print(f"    最初のマスク: {mask_tensor[0].shape if hasattr(mask_tensor[0], 'shape') else type(mask_tensor[0])}")
                                else:
                                    print(f"🎯 マスク: {mask_tensor.shape}, {mask_tensor.dtype}")
                            
                            # メタデータ保存
                            if image_tensor is not None:
                                save_comparison_image(
                                    image_tensor, mask_tensor,
                                    dataset_name, i,
                                    image_path=sample.get('image_path', ''),
                                    conversations=sample.get('conversations'),
                                    questions=sample.get('questions'),
                                    class_names=sample.get('sampled_classes'),
                                    session_timestamp=session_timestamp
                                )
                        
                        elif isinstance(sample, (list, tuple)):
                            print(f"📦 tuple形式: {len(sample)}要素")
                            for j, element in enumerate(sample):
                                if isinstance(element, torch.Tensor):
                                    print(f"  [{j}]: Tensor {element.shape}")
                                elif isinstance(element, str):
                                    print(f"  [{j}]: String ('{element[:50]}...')")
                                elif isinstance(element, list):
                                    print(f"  [{j}]: List({len(element)}個)")
                                else:
                                    print(f"  [{j}]: {type(element)}")
                            
                            # メタデータ保存（タプル形式）
                            try:
                                image_path = sample[0] if len(sample) > 0 and isinstance(sample[0], str) else ""
                                sam_image = sample[1] if len(sample) > 1 and isinstance(sample[1], torch.Tensor) else None
                                conversations = sample[3] if len(sample) > 3 and isinstance(sample[3], list) else None
                                masks = sample[4] if len(sample) > 4 and isinstance(sample[4], torch.Tensor) else None
                                questions = sample[7] if len(sample) > 7 and isinstance(sample[7], list) else None
                                sampled_classes = sample[8] if len(sample) > 8 and isinstance(sample[8], list) else None
                                
                                if sam_image is not None:
                                    save_comparison_image(
                                        sam_image, masks, 
                                        dataset_name, i,
                                        image_path=image_path,
                                        conversations=conversations,
                                        questions=questions,
                                        class_names=sampled_classes,
                                        session_timestamp=session_timestamp
                                    )
                                
                            except Exception as e:
                                print(f"⚠️ メタデータ保存エラー: {e}")
                        
                        print(f"✅ {dataset_name} Sample {i+1} 検証完了")
                        
                    except Exception as e:
                        print(f"❌ {dataset_name} Sample {i+1} エラー: {e}")
                        import traceback
                        traceback.print_exc()
                
                print(f"✅ {dataset_name} データセット検証完了")
                
            except Exception as e:
                print(f"❌ {dataset_name} データセット初期化エラー: {e}")
                import traceback
                traceback.print_exc()
        
        # 最終結果
        print(f"\n🎉 Llama4個別データセット検証完了")
        print("=" * 70)
        print(f"📊 検証結果:")
        print(f"   成功したデータセット: {successful_datasets}/{len(dataset_configs)}")
        print(f"   処理したサンプル数: {total_samples_processed}")
        print(f"📁 結果は verification_output/llama4_individual_{session_timestamp}/ に保存されました")
        print("=" * 70)
        
        if successful_datasets > 0:
            print("✅ 検証成功: 少なくとも1つのデータセットが正常に動作しました")
            return 0
        else:
            print("❌ 検証失敗: 有効なデータセットが見つかりませんでした")
            return 1
        
    except Exception as e:
        print(f"❌ 検証エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 