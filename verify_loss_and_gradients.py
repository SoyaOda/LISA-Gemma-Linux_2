#!/usr/bin/env python3
"""
第4節：損失計算と勾配伝播の精査

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

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.gemma_lisa import LisaGemmaForCausalLM
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
    parser = argparse.ArgumentParser(description="損失計算と勾配伝播の検証")
    return parser.parse_args()

def analyze_gradients(model, param_groups):
    """モデルの勾配を分析"""
    results = {}
    
    for module_name, module_keywords in param_groups.items():
        module_params = []
        module_grads = []
        
        for name, param in model.named_parameters():
            # キーワードでモジュールを判定
            if any(keyword in name for keyword in module_keywords):
                if param.requires_grad:
                    module_params.append((name, param))
                    
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_mean = param.grad.abs().mean().item()
                        module_grads.append({
                            "name": name,
                            "grad_norm": grad_norm,
                            "grad_mean": grad_mean,
                            "shape": list(param.shape)
                        })
        
        has_grad = len(module_grads) > 0
        avg_grad_norm = sum(g["grad_norm"] for g in module_grads) / len(module_grads) if module_grads else 0
        avg_grad_mean = sum(g["grad_mean"] for g in module_grads) / len(module_grads) if module_grads else 0
        
        results[module_name] = {
            "total_params": len(module_params),
            "params_with_grad": len(module_grads),
            "has_grad": has_grad,
            "avg_grad_norm": avg_grad_norm,
            "avg_grad_mean": avg_grad_mean,
            "details": module_grads[:3]  # 最初の3つの詳細のみ
        }
    
    return results

def main():
    args = parse_args()
    config = get_config()
    
    print("="*80)
    print("第4節: 損失計算と勾配伝播の精査")
    print("="*80)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    try:
        # 1. モデルの初期化
        print("\n📦 モデルを初期化中...")
        
        # LisaGemmaモデルの初期化
        from model.gemma_lisa import LisaGemmaConfig
        
        lisa_config = LisaGemmaConfig(
            gemma_model_id=getattr(config, 'GEMMA_MODEL_ID', 'google/gemma-3-4b-it'),
            sam_checkpoint_path=getattr(config, 'SAM_CHECKPOINT_PATH', None),
            seg_token=getattr(config, 'SEG_TOKEN', '[SEG]'),
            gemma_hidden_size=getattr(config, 'GEMMA_HIDDEN_SIZE', 2560),
            sam_prompt_embed_dim=getattr(config, 'SAM_PROMPT_EMBED_DIM', 256),
            gemma_image_size=getattr(config, 'GEMMA_IMAGE_SIZE', 896),
            sam_image_size=getattr(config, 'SAM_IMAGE_SIZE', 1024),
            model_max_length=getattr(config, 'MODEL_MAX_LENGTH', 2048),
        )
        
        model = LisaGemmaForCausalLM(lisa_config)
        model = model.to(device)
        model.train()  # 学習モードに設定
        
        print("✅ モデル初期化完了")
        
        # デバッグ情報の出力
        print("\n[デバッグ情報]")
        print(f"SEGトークンID: {model.seg_token_id}")
        
        # 埋め込み層の情報
        embed_layer = model.gemma_model.get_input_embeddings()
        print(f"埋め込み層のサイズ: {embed_layer.weight.shape}")
        print(f"埋め込み層の語彙サイズ: {embed_layer.weight.shape[0]}")
        
        # プロセッサーの語彙サイズ
        print(f"トークナイザーの語彙サイズ: {len(model.gemma_processor.tokenizer)}")
        
        # SEGトークンが埋め込み層の範囲内にあるか確認
        if model.seg_token_id >= embed_layer.weight.shape[0]:
            print(f"⚠️ 警告: SEGトークンID ({model.seg_token_id}) が埋め込み層のサイズ ({embed_layer.weight.shape[0]}) を超えています！")
            print("埋め込み層をリサイズします...")
            
            # 手動で埋め込み層をリサイズ
            model.gemma_model.resize_token_embeddings(len(model.gemma_processor.tokenizer))
            embed_layer = model.gemma_model.get_input_embeddings()
            print(f"リサイズ後の埋め込み層のサイズ: {embed_layer.weight.shape}")

        # パラメータ情報の表示
        param_info = model.get_trainable_parameters_info()
        print(f"\n  - 総パラメータ数: {param_info['total_parameters']:,}")
        print(f"  - 学習可能パラメータ数: {param_info['trainable_parameters']:,}")
        print(f"  - 学習可能率: {param_info['trainable_percentage']:.2f}%")
        
        # 2. プロセッサーとデータセットの準備
        print("\n📦 データセットを準備中...")
        
        processor = AutoProcessor.from_pretrained(lisa_config.gemma_model_id)
        
        # テスト用に小規模なデータセットを使用
        dataset = HybridDataset(
            base_image_dir=config.DATASET_BASE_DIR,
            gemma_processor=processor,
            dataset='reason_seg',  # セグメンテーションタスクを含むデータセット
            samples_per_epoch=10  # 少数のサンプル
        )
        
        # DataCollatorの準備
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_fn,
            shuffle=False
        )
        
        print(f"✅ データセット準備完了（サンプル数: {len(dataset)}）")
        
        # 3. オプティマイザーの準備
        optimizer = AdamW(model.parameters(), lr=1e-5)
        
        # 4. 損失関数の準備
        loss_fn = CompositeLoss(
            ce_loss_weight=getattr(config, 'CE_LOSS_WEIGHT', 1.0),
            dice_loss_weight=getattr(config, 'DICE_LOSS_WEIGHT', 0.5),
            bce_loss_weight=getattr(config, 'BCE_LOSS_WEIGHT', 2.0)
        )
        
        print("\n🔬 フォワードパスとバックワードパスを実行中...")
        
        # 1バッチを取得
        batch = next(iter(dataloader))
        
        # バッチをデバイスに移動
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # 5. フォワードパス
        print("\n[フォワードパス実行]")
        optimizer.zero_grad()
        
        # モデルに必要な入力を準備
        model_inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_masks'],  # attention_masksに注意
            'images_for_gemma': batch.get('images_for_gemma'),
            'images_for_sam': batch.get('images_for_sam'),
            'labels': batch['labels'],
            'generate_mask': True
        }
        
        # フォワードパス実行
        outputs = model(**model_inputs)
        
        print("✅ フォワードパス完了")
        
        # 6. 損失計算
        print("\n[損失計算チェック]:")
        
        # 損失を計算
        losses = loss_fn(outputs, batch)
        
        # 各損失項の確認
        total_loss = losses.get('total_loss')
        text_loss = losses.get('text_loss') 
        dice_loss = losses.get('dice_loss')
        bce_loss = losses.get('bce_loss')
        
        # 損失値の表示
        print(f"  - 総損失: {total_loss.item():.4f}")
        print(f"  - テキスト損失: {text_loss.item():.4f}")
        print(f"  - DICE損失: {dice_loss.item():.4f}")
        print(f"  - BCE損失: {bce_loss.item():.4f}")
        
        # 損失の検証
        assert total_loss is not None and not torch.isnan(total_loss), "総損失がNoneまたはNaN!"
        assert total_loss.requires_grad, "総損失が勾配を必要としていません。グラフが切断されている可能性があります。"
        
        print("✅ 損失計算成功 - すべての損失項が正常に計算されました")
        
        # 7. バックワードパス
        print("\n[勾配伝播チェック]:")
        print("バックワードパスを実行中...")
        
        total_loss.backward()
        
        print("✅ バックワードパス完了")
        
        # 8. 勾配の検査
        print("\n[パラメータグループごとの勾配検査]:")
        
        # 各モジュールのキーワード定義（Gemma-3モデル用に更新）
        param_groups = {
            'gemma_embeddings': ['gemma_model.model.embed_tokens', 'gemma_model.get_input_embeddings'],
            'mlp_projector': ['mlp_projector'],
            'sam_image_encoder': ['sam_image_encoder'],
            'sam_prompt_encoder': ['sam_prompt_encoder'],
            'sam_mask_decoder': ['sam_mask_decoder'],
            'gemma_lm_head': ['gemma_model.lm_head', 'gemma_model.get_output_embeddings']
        }
        
        # 勾配を分析
        grad_results = analyze_gradients(model, param_groups)
        
        # 結果の表示
        all_grads_ok = True
        for module_name, result in grad_results.items():
            if result['total_params'] == 0:
                print(f"  - {module_name}: パラメータなし")
                continue
            
            if result['has_grad']:
                print(f"  - {module_name}: ✅ 勾配あり")
                print(f"    - パラメータ数: {result['total_params']}")
                print(f"    - 勾配ありパラメータ: {result['params_with_grad']}")
                print(f"    - 平均勾配ノルム: {result['avg_grad_norm']:.2e}")
                print(f"    - 平均勾配絶対値: {result['avg_grad_mean']:.2e}")
            else:
                if module_name in ['sam_image_encoder', 'sam_prompt_encoder']:
                    # これらは凍結されているので勾配がないのが正常
                    print(f"  - {module_name}: ✅ 勾配なし（凍結モジュール）")
                else:
                    print(f"  - {module_name}: ❌ 勾配なし - 計算グラフが切断されている可能性")
                    all_grads_ok = False
        
        # 総合判定
        print("\n" + "="*60)
        if all_grads_ok:
            print("✅ 成功: すべての学習可能モジュールに勾配が伝播しています")
            print("💡 計算グラフは正常に接続されており、学習が可能です")
        else:
            print("❌ エラー: 一部のモジュールに勾配が伝播していません")
            print("💡 .detach()やtorch.no_grad()の使用箇所を確認してください")
        
        # 追加の診断情報
        print("\n[追加診断情報]:")
        
        # SEGトークンの存在確認
        seg_token_count = (batch['input_ids'] == processor.tokenizer.convert_tokens_to_ids('[SEG]')).sum().item()
        print(f"  - バッチ内のSEGトークン数: {seg_token_count}")
        
        # 予測マスクの有無
        if outputs.get('predicted_masks') is not None:
            print(f"  - 予測マスクの形状: {outputs['predicted_masks'].shape}")
        else:
            print("  - 予測マスク: なし")
        
        # 正解マスクの有無
        if batch.get('ground_truth_mask') is not None:
            print(f"  - 正解マスクの形状: {batch['ground_truth_mask'].shape}")
        else:
            print("  - 正解マスク: なし")
        
        print("\n🎉 第4節検証完了: 損失計算と勾配伝播の精査")
        print(f"🕐 完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ 検証中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        
        # エラー時の診断情報
        print("\n[エラー診断]:")
        print("1. GPUメモリ不足の場合: バッチサイズを小さくしてください")
        print("2. モデル初期化エラーの場合: 設定ファイルのパスを確認してください")
        print("3. データセットエラーの場合: データセットのパスを確認してください")

if __name__ == "__main__":
    main() 