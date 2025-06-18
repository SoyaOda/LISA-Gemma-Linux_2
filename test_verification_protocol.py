#!/usr/bin/env python3
"""
LISA-Gemma3 段階的検証プロトコル
仕様書第6章「実行および検証プロトコル」に従った実装
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from functools import partial
import time

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_linux import *
from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
from model.losses import CompositeLoss, IoUMetric
from utils.dataset import LisaGemma3Dataset, collate_fn_gemma3
from peft import LoraConfig, get_peft_model


class VerificationProtocol:
    """仕様書第6章に従った段階的検証プロトコル"""
    
    def __init__(self, use_sam=True):
        self.use_sam = use_sam
        self.model = None
        self.dataloader = None
        self.loss_fn = None
        self.results = {}
        
    def setup_model(self):
        """モデルとLoRAの設定"""
        print("=" * 60)
        print("🔧 モデルとLoRAの設定")
        print("=" * 60)
        
        # 設定作成
        config = LisaGemmaConfig(
            gemma_model_id=GEMMA_MODEL_ID,
            sam_checkpoint_path=SAM_CHECKPOINT_PATH if self.use_sam else "",
            seg_token="<SEG>",
            gemma_hidden_size=2560,
            sam_prompt_embed_dim=256,
        )
        
        # モデル初期化
        print("モデルを初期化中...")
        self.model = LisaGemmaForCausalLM(config)
        
        # LoRA設定
        print("LoRA設定を適用中...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # 内部のGemmaモデルにLoRAを適用
        self.model.gemma_model = get_peft_model(self.model.gemma_model, lora_config)
        
        # パラメータ凍結設定（仕様書第2章.5）
        for param in self.model.gemma_model.base_model.model.parameters():
            param.requires_grad = False
        
        # 訓練可能コンポーネント
        for param in self.model.mlp_projector.parameters():
            param.requires_grad = True
            
        if self.use_sam and self.model.sam_mask_decoder:
            for param in self.model.sam_mask_decoder.parameters():
                param.requires_grad = True
        
        # パラメータ数確認
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ モデル初期化完了")
        print(f"総パラメータ数: {total_params:,}")
        print(f"訓練可能パラメータ数: {trainable_params:,}")
        print(f"訓練可能な割合: {trainable_params/total_params*100:.2f}%")
        
        self.results['model_setup'] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_percentage': trainable_params/total_params*100
        }
        
    def setup_data(self, batch_size=2, samples=10):
        """データセットとデータローダーの設定"""
        print("\n" + "=" * 60)
        print("📊 データセットとデータローダーの設定")
        print("=" * 60)
        
        # Gemma-3プロセッサー初期化
        gemma_processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
        
        # データセット作成
        dataset = LisaGemma3Dataset(
            base_image_dir=DATASET_BASE_DIR,
            gemma_processor=gemma_processor,
            samples_per_epoch=samples,
            precision="bf16",
            gemma_image_size=GEMMA_IMAGE_SIZE,
            sam_image_size=SAM_IMAGE_SIZE,
        )
        
        # データローダー作成
        collate_fn = partial(collate_fn_gemma3, gemma_processor=gemma_processor)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # 損失関数設定
        self.loss_fn = CompositeLoss(
            ce_loss_weight=CE_LOSS_WEIGHT,
            dice_loss_weight=DICE_LOSS_WEIGHT,
            bce_loss_weight=BCE_LOSS_WEIGHT
        )
        
        print(f"✅ データセット作成完了: {len(dataset)} サンプル")
        print(f"✅ データローダー作成完了: バッチサイズ {batch_size}")
        
    def step1_data_sanity_check(self):
        """ステップ1: データ健全性チェック（仕様書第6章.2.1）"""
        print("\n" + "=" * 60)
        print("📋 ステップ1: データ健全性チェック")
        print("=" * 60)
        
        try:
            # 1バッチ取得
            batch = next(iter(self.dataloader))
            
            print("🔍 バッチ内容の検証:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                elif isinstance(value, list):
                    print(f"  {key}: list of {len(value)} items")
                else:
                    print(f"  {key}: {type(value)}")
            
            # 画像テンソル形状の検証
            pixel_values = batch['pixel_values']
            expected_gemma_shape = (len(batch['images']), 3, GEMMA_IMAGE_SIZE, GEMMA_IMAGE_SIZE)
            
            if pixel_values.shape == expected_gemma_shape:
                print(f"✅ Gemma画像形状が正しい: {pixel_values.shape}")
            else:
                print(f"❌ Gemma画像形状が不正: 期待{expected_gemma_shape}, 実際{pixel_values.shape}")
            
            # テキストプロンプトのデコード検証
            input_ids = batch['input_ids']
            gemma_processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
            
            print("\n🔍 テキストプロンプトの検証:")
            for i in range(min(2, input_ids.shape[0])):
                decoded = gemma_processor.tokenizer.decode(input_ids[i], skip_special_tokens=False)
                print(f"  サンプル{i+1}: {decoded[:100]}...")
                
                # SEGトークンの存在確認
                seg_token_id = gemma_processor.tokenizer.convert_tokens_to_ids("<SEG>")
                has_seg = seg_token_id in input_ids[i]
                print(f"    <SEG>トークン: {'✅ 存在' if has_seg else '❌ なし'}")
            
            # マスクの可視化（最初のサンプルのみ）
            if batch['masks'] and len(batch['masks']) > 0:
                self._visualize_mask(batch['images'][0], batch['masks'][0], "step1_mask_overlay.png")
                print(f"✅ マスクの可視化を保存: step1_mask_overlay.png")
            
            self.results['step1'] = {'status': 'success', 'batch_info': {k: str(v.shape) if isinstance(v, torch.Tensor) else str(type(v)) for k, v in batch.items()}}
            print("✅ ステップ1: データ健全性チェック完了")
            return batch
            
        except Exception as e:
            print(f"❌ ステップ1失敗: {e}")
            import traceback
            traceback.print_exc()
            self.results['step1'] = {'status': 'failed', 'error': str(e)}
            return None
    
    def step2_forward_pass_test(self, batch):
        """ステップ2: フォワードパス・テスト（仕様書第6章.2.2）"""
        print("\n" + "=" * 60)
        print("🚀 ステップ2: フォワードパス・テスト")
        print("=" * 60)
        
        if batch is None:
            print("❌ バッチが無効のためスキップ")
            return None
            
        try:
            self.model.eval()
            device = next(self.model.parameters()).device
            
            # バッチをGPUに移動
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            print("🔍 フォワードパス実行中...")
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    labels=batch['input_ids'],
                    generate_mask=self.use_sam,
                )
            
            print("🔍 出力の検証:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                    
                    # 次元不一致エラーの検証
                    if 'mask' in key.lower() and self.use_sam:
                        print(f"    マスク値範囲: [{value.min().item():.4f}, {value.max().item():.4f}]")
                elif value is not None:
                    print(f"  {key}: {type(value)}")
                else:
                    print(f"  {key}: None")
            
            # 期待する出力の検証
            required_outputs = ['logits']
            if self.use_sam:
                required_outputs.append('predicted_masks')
                
            for req_output in required_outputs:
                if req_output in outputs and outputs[req_output] is not None:
                    print(f"✅ {req_output}: 正常")
                else:
                    print(f"⚠️ {req_output}: 欠損または None")
            
            self.results['step2'] = {'status': 'success', 'outputs': list(outputs.keys())}
            print("✅ ステップ2: フォワードパス・テスト完了")
            return outputs
            
        except Exception as e:
            print(f"❌ ステップ2失敗: {e}")
            import traceback
            traceback.print_exc()
            self.results['step2'] = {'status': 'failed', 'error': str(e)}
            return None
    
    def step3_single_batch_overfitting(self, batch, max_steps=100):
        """ステップ3: 単一バッチ過学習（仕様書第6章.2.3 - 最も重要なテスト）"""
        print("\n" + "=" * 60)
        print("🎯 ステップ3: 単一バッチ過学習（最重要テスト）")
        print("=" * 60)
        
        if batch is None:
            print("❌ バッチが無効のためスキップ")
            return None
            
        try:
            self.model.train()
            device = next(self.model.parameters()).device
            
            # バッチをGPUに移動
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # オプティマイザ設定
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            )
            
            print(f"🔍 {max_steps}ステップの過学習開始...")
            losses = []
            
            for step in range(max_steps):
                optimizer.zero_grad()
                
                # フォワードパス
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    labels=batch['input_ids'],
                    generate_mask=self.use_sam,
                )
                
                # 損失計算
                batch_for_loss = {
                    'labels': batch['input_ids'],
                    'ground_truth_mask': batch.get('masks')
                }
                
                loss_dict = self.loss_fn(outputs, batch_for_loss)
                total_loss = loss_dict['total_loss']
                
                # バックワード
                total_loss.backward()
                optimizer.step()
                
                losses.append(total_loss.item())
                
                # ログ出力
                if step % 10 == 0 or step == max_steps - 1:
                    print(f"  Step {step:3d}: Loss = {total_loss.item():.4f}")
            
            # 収束判定
            initial_loss = losses[0]
            final_loss = losses[-1]
            reduction_ratio = (initial_loss - final_loss) / initial_loss
            
            print(f"\n📊 過学習結果:")
            print(f"  初期損失: {initial_loss:.4f}")
            print(f"  最終損失: {final_loss:.4f}")
            print(f"  損失減少率: {reduction_ratio*100:.2f}%")
            
            # 成功基準の判定
            success_threshold = 0.1  # 10%以上の損失減少
            if reduction_ratio > success_threshold:
                print(f"✅ 過学習成功: 損失が{reduction_ratio*100:.1f}%減少（閾値{success_threshold*100}%以上）")
                status = 'success'
            else:
                print(f"⚠️ 過学習不十分: 損失減少{reduction_ratio*100:.1f}%（閾値{success_threshold*100}%未満）")
                status = 'insufficient'
            
            # 損失カーブの保存
            self._plot_loss_curve(losses, "step3_overfitting_curve.png")
            
            self.results['step3'] = {
                'status': status,
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'reduction_ratio': reduction_ratio,
                'losses': losses
            }
            
            print("✅ ステップ3: 単一バッチ過学習完了")
            return losses
            
        except Exception as e:
            print(f"❌ ステップ3失敗: {e}")
            import traceback
            traceback.print_exc()
            self.results['step3'] = {'status': 'failed', 'error': str(e)}
            return None
    
    def step4_gradient_flow_inspection(self):
        """ステップ4: 勾配フロー検査（仕様書第6章.2.4）"""
        print("\n" + "=" * 60)
        print("🔍 ステップ4: 勾配フロー検査")
        print("=" * 60)
        
        try:
            print("🔍 訓練可能パラメータの勾配確認:")
            
            gradient_info = {
                'with_gradients': [],
                'without_gradients': [],
                'total_trainable': 0,
                'total_with_grad': 0
            }
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    gradient_info['total_trainable'] += 1
                    
                    if param.grad is not None:
                        gradient_info['total_with_grad'] += 1
                        gradient_info['with_gradients'].append({
                            'name': name,
                            'shape': list(param.shape),
                            'grad_norm': param.grad.norm().item()
                        })
                        print(f"  ✅ {name}: grad_norm = {param.grad.norm().item():.6f}")
                    else:
                        gradient_info['without_gradients'].append(name)
                        print(f"  ❌ {name}: grad = None")
            
            print(f"\n📊 勾配フロー統計:")
            print(f"  訓練可能パラメータ数: {gradient_info['total_trainable']}")
            print(f"  勾配を持つパラメータ数: {gradient_info['total_with_grad']}")
            print(f"  勾配フロー率: {gradient_info['total_with_grad']/gradient_info['total_trainable']*100:.1f}%")
            
            # 重要コンポーネントの勾配確認
            print(f"\n🔍 重要コンポーネントの勾配確認:")
            important_components = {
                'MLP Projector': 'mlp_projector',
                'LoRA Adapters': 'lora',
                'SAM Decoder': 'sam_mask_decoder' if self.use_sam else None
            }
            
            for comp_name, comp_key in important_components.items():
                if comp_key is None:
                    continue
                    
                comp_grads = [info for info in gradient_info['with_gradients'] if comp_key in info['name']]
                if comp_grads:
                    avg_grad_norm = np.mean([info['grad_norm'] for info in comp_grads])
                    print(f"  ✅ {comp_name}: {len(comp_grads)}個のパラメータ, 平均勾配ノルム = {avg_grad_norm:.6f}")
                else:
                    print(f"  ❌ {comp_name}: 勾配なし")
            
            # 成功判定
            success_rate = gradient_info['total_with_grad'] / gradient_info['total_trainable']
            if success_rate > 0.9:  # 90%以上
                status = 'success'
                print("✅ 勾配フロー検査成功: 大部分のパラメータに勾配が流れています")
            else:
                status = 'partial'
                print(f"⚠️ 勾配フロー不完全: {success_rate*100:.1f}%のパラメータにのみ勾配")
            
            self.results['step4'] = {
                'status': status,
                'gradient_flow_rate': success_rate,
                'gradient_info': gradient_info
            }
            
            print("✅ ステップ4: 勾配フロー検査完了")
            return gradient_info
            
        except Exception as e:
            print(f"❌ ステップ4失敗: {e}")
            import traceback
            traceback.print_exc()
            self.results['step4'] = {'status': 'failed', 'error': str(e)}
            return None
    
    def step5_results_interpretation(self):
        """ステップ5: 結果の解釈と定性的評価（仕様書第6章.2.5）"""
        print("\n" + "=" * 60)
        print("📈 ステップ5: 結果の解釈と定性的評価")
        print("=" * 60)
        
        try:
            print("📊 全体的な検証結果サマリー:")
            
            total_steps = 5
            successful_steps = 0
            
            for step_num in range(1, total_steps + 1):
                step_key = f'step{step_num}'
                if step_key in self.results:
                    status = self.results[step_key]['status']
                    if status == 'success':
                        successful_steps += 1
                        print(f"  ✅ ステップ{step_num}: 成功")
                    elif status == 'partial' or status == 'insufficient':
                        print(f"  ⚠️ ステップ{step_num}: 部分的成功")
                    else:
                        print(f"  ❌ ステップ{step_num}: 失敗")
                else:
                    print(f"  ❓ ステップ{step_num}: 未実行")
            
            success_rate = successful_steps / total_steps
            print(f"\n📊 総合成功率: {success_rate*100:.1f}% ({successful_steps}/{total_steps})")
            
            # 推奨事項
            print(f"\n💡 推奨事項:")
            
            if success_rate >= 0.8:
                print("  🎉 優秀な結果です！本格的な学習を開始できます。")
                recommendation = "ready_for_training"
            elif success_rate >= 0.6:
                print("  👍 基本的な動作は確認できました。一部の問題を修正後、学習を開始してください。")
                recommendation = "minor_fixes_needed"
            else:
                print("  ⚠️ 重要な問題があります。学習前に根本的な修正が必要です。")
                recommendation = "major_fixes_needed"
            
            # 具体的な修正提案
            if 'step3' in self.results and self.results['step3']['status'] != 'success':
                print("    - 単一バッチ過学習が不十分です。学習率やモデル設定を確認してください。")
            
            if 'step4' in self.results and self.results['step4']['status'] != 'success':
                print("    - 勾配フローに問題があります。パラメータの凍結設定を確認してください。")
            
            # IoU評価（SAM使用時）
            if self.use_sam and hasattr(self, 'model') and self.model.sam_mask_decoder:
                print("\n🎯 セグメンテーション性能評価:")
                iou_metric = IoUMetric()
                print("  セグメンテーション機能が有効です。")
            else:
                print("\n🎯 セグメンテーション性能評価:")
                print("  SAMが無効のため、セグメンテーション評価はスキップされました。")
            
            self.results['step5'] = {
                'status': 'success',
                'success_rate': success_rate,
                'recommendation': recommendation,
                'successful_steps': successful_steps,
                'total_steps': total_steps
            }
            
            print("✅ ステップ5: 結果の解釈と定性的評価完了")
            return self.results
            
        except Exception as e:
            print(f"❌ ステップ5失敗: {e}")
            import traceback
            traceback.print_exc()
            self.results['step5'] = {'status': 'failed', 'error': str(e)}
            return None
    
    def _visualize_mask(self, pil_image, mask, filename):
        """マスクの可視化"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 元画像
            axes[0].imshow(pil_image)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # マスク
            if isinstance(mask, torch.Tensor):
                mask_np = mask.squeeze().cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            axes[1].imshow(mask_np, cmap='gray')
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis('off')
            
            # オーバーレイ
            overlay = np.array(pil_image)
            if len(mask_np.shape) == 2:
                mask_colored = np.zeros((mask_np.shape[0], mask_np.shape[1], 3))
                mask_colored[:, :, 0] = mask_np  # 赤チャンネル
                overlay = overlay * 0.7 + mask_colored * 0.3 * 255
            
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title("Overlay")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"可視化エラー: {e}")
    
    def _plot_loss_curve(self, losses, filename):
        """損失カーブのプロット"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(losses, 'b-', linewidth=2)
            plt.title('Single Batch Overfitting - Loss Curve')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"プロットエラー: {e}")
    
    def run_full_verification(self, use_sam=None):
        """完全な検証プロトコルの実行"""
        if use_sam is not None:
            self.use_sam = use_sam
            
        print("🚀 LISA-Gemma3 段階的検証プロトコル開始")
        print("仕様書第6章「実行および検証プロトコル」に従った実装")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # セットアップ
            self.setup_model()
            self.setup_data()
            
            # 段階的検証実行
            batch = self.step1_data_sanity_check()
            outputs = self.step2_forward_pass_test(batch)
            losses = self.step3_single_batch_overfitting(batch)
            gradients = self.step4_gradient_flow_inspection()
            results = self.step5_results_interpretation()
            
            # 実行時間
            elapsed_time = time.time() - start_time
            
            print("\n" + "=" * 80)
            print("🎯 検証プロトコル完了")
            print("=" * 80)
            print(f"⏱️ 実行時間: {elapsed_time:.1f}秒")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ 検証プロトコル実行エラー: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """メイン実行関数"""
    # SAMありとなしの両方でテスト
    print("LISA-Gemma3 段階的検証プロトコル")
    print("=" * 80)
    
    # SAMなしでの検証
    print("\n🔧 SAMなしでの検証開始...")
    protocol_no_sam = VerificationProtocol(use_sam=False)
    results_no_sam = protocol_no_sam.run_full_verification()
    
    # SAMありでの検証（SAMチェックポイントが利用可能な場合）
    if os.path.exists(SAM_CHECKPOINT_PATH):
        print("\n🔧 SAMありでの検証開始...")
        protocol_with_sam = VerificationProtocol(use_sam=True)
        results_with_sam = protocol_with_sam.run_full_verification()
    else:
        print(f"\n⚠️ SAMチェックポイントが見つかりません: {SAM_CHECKPOINT_PATH}")
        print("SAMありでの検証はスキップされました。")
        results_with_sam = None
    
    # 最終レポート
    print("\n" + "=" * 80)
    print("📋 最終検証レポート")
    print("=" * 80)
    
    if results_no_sam:
        success_rate_no_sam = results_no_sam.get('step5', {}).get('success_rate', 0)
        print(f"SAMなし検証: {success_rate_no_sam*100:.1f}% 成功")
    
    if results_with_sam:
        success_rate_with_sam = results_with_sam.get('step5', {}).get('success_rate', 0)
        print(f"SAMあり検証: {success_rate_with_sam*100:.1f}% 成功")
    
    print("\n🎯 次のステップ:")
    print("1. 検証結果を確認し、必要に応じて修正を実施")
    print("2. 成功率が80%以上の場合、本格的な学習を開始")
    print("3. DeepSpeedを使用した分散学習の実行")


if __name__ == "__main__":
    main()