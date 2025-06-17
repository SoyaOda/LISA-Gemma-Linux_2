#!/usr/bin/env python3
"""
実データでのLISA-Gemma3統合テスト
実際のデータセットファイルを使用してエンドツーエンドのテストを実行
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import json
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_linux import *
from utils.dataset import LisaGemma3Dataset, collate_fn_gemma3
from utils.reason_seg_dataset import ReasonSegDataset
from utils.vqa_dataset import VQADataset
from utils.refer_seg_dataset import ReferSegDataset
from utils.sem_seg_dataset import SemSegDataset
from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
from model.losses import CompositeLoss
from torch.utils.data import DataLoader

class RealDataTester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用デバイス: {self.device}")
        
        # 設定の確認
        self.check_config()
        
    def check_config(self):
        """設定とデータセットパスの確認"""
        print("=== 設定確認 ===")
        print(f"Dataset base dir: {DATASET_BASE_DIR}")
        print(f"SAM checkpoint: {SAM_CHECKPOINT_PATH}")
        print(f"Gemma model ID: {GEMMA_MODEL_ID}")
        
        # config_linux.pyのcheck_paths関数を呼び出して必須パスを検証
        # 存在しない場合はFileNotFoundErrorで停止
        try:
            from config_linux import check_paths
            print("\n必須パスの検証中...")
            check_paths()
            print("✓ すべての必須パスが確認されました")
        except FileNotFoundError as e:
            print(f"\n✗ 必須パスの検証に失敗: {e}")
            raise
        
    def test_individual_datasets(self):
        """各データセットの個別テスト"""
        print("\n=== 個別データセットテスト ===")
        
        # ReasonSegデータセット
        print("\n1. ReasonSegデータセットテスト")
        try:
            dataset = ReasonSegDataset(
                base_image_dir=DATASET_BASE_DIR,
                tokenizer=None,  # 簡易テスト用
                samples_per_epoch=10,
                reason_seg_data="ReasonSeg|train"
            )
            print(f"✓ ReasonSegデータセット作成成功 (サンプル数: {len(dataset)})")
            
            # 最初のサンプルを取得
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  - サンプル構造: {len(sample)} 要素のタプル")
        except Exception as e:
            print(f"✗ ReasonSegデータセットテスト失敗: {e}")
        
        # VQAデータセット
        print("\n2. VQAデータセットテスト")
        try:
            dataset = VQADataset(
                base_image_dir=DATASET_BASE_DIR,
                tokenizer=None,
                samples_per_epoch=10,
                vqa_data="llava_instruct_150k"
            )
            print(f"✓ VQAデータセット作成成功 (サンプル数: {len(dataset)})")
        except Exception as e:
            print(f"✗ VQAデータセットテスト失敗: {e}")
        
        # ReferSegデータセット
        print("\n3. ReferSegデータセットテスト")
        try:
            dataset = ReferSegDataset(
                base_image_dir=DATASET_BASE_DIR,
                tokenizer=None,
                samples_per_epoch=10,
                refer_seg_data="refcoco"
            )
            print(f"✓ ReferSegデータセット作成成功 (サンプル数: {len(dataset)})")
        except Exception as e:
            print(f"✗ ReferSegデータセットテスト失敗: {e}")
        
        # SemSegデータセット
        print("\n4. SemSegデータセットテスト")
        try:
            dataset = SemSegDataset(
                base_image_dir=DATASET_BASE_DIR,
                tokenizer=None,
                samples_per_epoch=10,
                sem_seg_data="ade20k"
            )
            print(f"✓ SemSegデータセット作成成功 (サンプル数: {len(dataset)})")
        except Exception as e:
            print(f"✗ SemSegデータセットテスト失敗: {e}")
    
    def test_unified_dataset(self):
        """統合データセットのテスト"""
        print("\n=== 統合データセットテスト ===")
        
        try:
            # Gemma-3プロセッサーの初期化
            from transformers import AutoProcessor
            gemma_processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
            
            # 統合データセットの作成
            dataset = LisaGemma3Dataset(
                base_image_dir=DATASET_BASE_DIR,
                gemma_processor=gemma_processor,
                samples_per_epoch=20,
                dataset="reason_seg||vqa",  # 利用可能なデータセットのみ
                sample_rate=[1, 1],
                reason_seg_data="ReasonSeg|train",
                vqa_data="llava_instruct_150k",
                refer_seg_data="refcoco",
                sem_seg_data="ade20k"
            )
            
            print(f"✓ 統合データセット作成成功 (サンプル数: {len(dataset)})")
            
            # データローダーのテスト
            from functools import partial
            collate_fn = partial(collate_fn_gemma3, gemma_processor=gemma_processor)
            
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0  # デバッグ用
            )
            
            print("✓ データローダー作成成功")
            
            # バッチの取得テスト
            for i, batch in enumerate(dataloader):
                print(f"  バッチ {i+1}:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"    {key}: {value.shape} ({value.dtype})")
                    elif isinstance(value, list):
                        print(f"    {key}: list[{len(value)}]")
                    else:
                        print(f"    {key}: {type(value)}")
                
                if i >= 2:  # 最初の3バッチのみテスト
                    break
                    
        except Exception as e:
            print(f"✗ 統合データセットテスト失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def test_model_initialization(self):
        """モデル初期化のテスト"""
        print("\n=== モデル初期化テスト ===")
        
        try:
            # SAMチェックポイントなしでの初期化テスト
            config = LisaGemmaConfig(
                gemma_model_id=GEMMA_MODEL_ID,
                sam_checkpoint_path="",  # SAMなしでテスト
                gemma_hidden_size=2560,
                sam_prompt_embed_dim=256
            )
            
            print("Gemma-3モデル（SAMなし）の初期化中...")
            model = LisaGemmaForCausalLM(config)
            print("✓ モデル初期化成功")
            
            # パラメータ情報の表示
            param_info = model.get_trainable_parameters_info()
            print(f"  総パラメータ数: {param_info['total_parameters']:,}")
            print(f"  訓練可能パラメータ数: {param_info['trainable_parameters']:,}")
            print(f"  訓練可能割合: {param_info['trainable_percentage']:.2f}%")
            
            # デバイスに移動
            model.to(self.device)
            print(f"✓ モデルを{self.device}に移動")
            
            return model
            
        except Exception as e:
            print(f"✗ モデル初期化失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_model_forward(self, model):
        """モデルのフォワードパステスト"""
        print("\n=== モデルフォワードパステスト ===")
        
        if model is None:
            print("✗ モデルが初期化されていません")
            return
        
        try:
            # ダミー画像とテキストでテスト
            dummy_image = Image.new('RGB', (512, 512), color='red')
            test_prompt = "この画像の赤い部分をセグメント化してください。 <SEG>"
            
            print("フォワードパス実行中...")
            with torch.no_grad():
                outputs = model(
                    image=dummy_image,
                    text_prompt=test_prompt,
                    generate_mask=False  # SAMなしなのでマスク生成は無効
                )
            
            print("✓ フォワードパス成功")
            print(f"  出力キー: {list(outputs.keys())}")
            
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"    {key}: {type(value)}")
                    
        except Exception as e:
            print(f"✗ フォワードパス失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def test_loss_function(self):
        """損失関数のテスト"""
        print("\n=== 損失関数テスト ===")
        
        try:
            loss_fn = CompositeLoss(
                ce_loss_weight=1.0,
                dice_loss_weight=0.5,
                bce_loss_weight=2.0
            )
            print("✓ 損失関数初期化成功")
            
            # ダミーデータで損失計算テスト
            batch_size = 2
            seq_len = 100
            vocab_size = 32000
            mask_size = 64
            
            # ダミー出力
            outputs = {
                "gemma_logits": torch.randn(batch_size, seq_len, vocab_size),
                "predicted_masks": torch.randn(batch_size, 1, mask_size, mask_size)
            }
            
            # ダミーバッチ
            batch = {
                "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
                "ground_truth_mask": torch.randint(0, 2, (batch_size, 1, mask_size, mask_size)).float()
            }
            
            # 損失計算
            losses = loss_fn(outputs, batch)
            print("✓ 損失計算成功")
            
            for key, value in losses.items():
                print(f"    {key}: {value.item():.4f}")
                
        except Exception as e:
            print(f"✗ 損失関数テスト失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def run_all_tests(self):
        """全テストの実行"""
        print("LISA-Gemma3 実データテスト開始")
        print("=" * 50)
        
        # 個別データセットテスト
        self.test_individual_datasets()
        
        # 統合データセットテスト
        self.test_unified_dataset()
        
        # モデル初期化テスト
        model = self.test_model_initialization()
        
        # モデルフォワードパステスト
        self.test_model_forward(model)
        
        # 損失関数テスト
        self.test_loss_function()
        
        print("\n" + "=" * 50)
        print("テスト完了")


if __name__ == "__main__":
    tester = RealDataTester()
    tester.run_all_tests() 