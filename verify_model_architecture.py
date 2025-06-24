#!/usr/bin/env python3
"""
第3節：視覚-言語アライメントのためのアーキテクチャ監査

完全なLISA-Gemmaモデルをインスタンス化し、その構成要素（Vision Tower, Projector, LLM, Seg Decoder）を検査し、
次元の互換性を検証し、各モジュールの学習可能パラメータの状態を報告する。

論理的根拠:
- デュアルエンコーダー問題: Vision EncoderとLanguage Modelを効果的に協調させる
- プロジェクション層の構造が健全でなければ、マルチモーダル学習は完全に失敗する
- 次元の整合性とパラメータの学習設定が仕様書通りであることを確認
"""

import argparse
import yaml
import os
import sys
import traceback
from datetime import datetime

import torch
import torch.nn as nn

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    parser = argparse.ArgumentParser(description="Verify model architecture and parameters")
    return parser.parse_args()

def count_parameters(module):
    """モジュールの総パラメータ数と学習可能パラメータ数をカウント"""
    total_params = 0
    trainable_params = 0
    
    for param in module.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return total_params, trainable_params

def main():
    args = parse_args()
    config = get_config()
    
    print("="*80)
    print("第3節: 視覚-言語アライメントのためのアーキテクチャ監査")
    print("="*80)
    
    try:
        # デバイス設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用デバイス: {device}")
        
        # 1. モデルの初期化
        print("\n📝 LISA-Gemmaモデルを初期化中...")
        
        from model.gemma_lisa import LisaGemmaForCausalLM, LisaGemmaConfig
        
        # LisaGemmaConfigの作成
        lisa_config = LisaGemmaConfig(
            gemma_model_id=config.GEMMA_MODEL_ID,
            sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
            seg_token=config.SEG_TOKEN,
            gemma_hidden_size=config.GEMMA_HIDDEN_SIZE,
            sam_prompt_embed_dim=config.SEG_PROJECTION_DIM,
            gemma_image_size=config.GEMMA_IMAGE_SIZE,
            sam_image_size=config.SAM_IMAGE_SIZE,
            model_max_length=config.MODEL_MAX_LENGTH,
        )
        
        # モデルの初期化
        model = LisaGemmaForCausalLM(lisa_config)
        model = model.to(device)
        
        print("✅ モデル初期化完了")
        print("-" * 80)
        
        # 2. モデル全体の構造を出力
        print("\n[モデルアーキテクチャ]:")
        print(f"モデルタイプ: {model.__class__.__name__}")
        
        # 主要コンポーネントの存在確認
        has_gemma = hasattr(model, 'gemma_model') and model.gemma_model is not None
        has_sam_encoder = hasattr(model, 'sam_image_encoder') and model.sam_image_encoder is not None
        has_sam_prompt = hasattr(model, 'sam_prompt_encoder') and model.sam_prompt_encoder is not None
        has_sam_decoder = hasattr(model, 'sam_mask_decoder') and model.sam_mask_decoder is not None
        has_projector = hasattr(model, 'mlp_projector') and model.mlp_projector is not None
        
        print("\n主要コンポーネント:")
        print(f"  ✓ Gemma-3 Model: {'存在' if has_gemma else '欠如'}")
        print(f"  ✓ SAM Image Encoder: {'存在' if has_sam_encoder else '欠如'}")
        print(f"  ✓ SAM Prompt Encoder: {'存在' if has_sam_prompt else '欠如'}")
        print(f"  ✓ SAM Mask Decoder: {'存在' if has_sam_decoder else '欠如'}")
        print(f"  ✓ MLP Projector: {'存在' if has_projector else '欠如'}")
        print("-" * 80)
        
        # 3. 主要モジュールの次元整合性チェック
        print("\n[次元整合性チェック]:")
        
        if has_projector:
            # MLPプロジェクタの構造を確認
            print("\nMLPプロジェクタ構造:")
            for i, layer in enumerate(model.mlp_projector):
                print(f"  Layer {i}: {layer}")
                if isinstance(layer, nn.Linear):
                    print(f"    - 入力次元: {layer.in_features}")
                    print(f"    - 出力次元: {layer.out_features}")
            
            # ダミー入力でテスト
            dummy_hidden_state = torch.randn(1, config.GEMMA_HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
            print(f"\nダミー入力形状: {dummy_hidden_state.shape}")
            
            with torch.no_grad():
                projected_output = model.mlp_projector(dummy_hidden_state)
            
            print(f"プロジェクタ出力形状: {projected_output.shape}")
            print(f"期待される出力次元: {config.SEG_PROJECTION_DIM}")
            
            # 次元の一致をアサート
            projector_output_dim = projected_output.shape[-1]
            expected_dim = config.SEG_PROJECTION_DIM
            
            if projector_output_dim == expected_dim:
                print(f"\n✅ SUCCESS: プロジェクタ出力次元({projector_output_dim})が")
                print(f"   SAMプロンプト埋め込み次元({expected_dim})と一致しています")
            else:
                print(f"\n❌ ERROR: 次元不一致!")
                print(f"   プロジェクタ出力: {projector_output_dim}")
                print(f"   期待される次元: {expected_dim}")
        
        # SAMコンポーネントのテスト（存在する場合）
        if has_sam_encoder and has_sam_prompt and has_sam_decoder:
            print("\n\nSAMコンポーネントの検証:")
            
            # SAM画像エンコーダのテスト
            dummy_sam_image = torch.randn(1, 3, config.SAM_IMAGE_SIZE, config.SAM_IMAGE_SIZE, 
                                        device=device, dtype=torch.float32)
            print(f"ダミーSAM画像形状: {dummy_sam_image.shape}")
            
            with torch.no_grad():
                sam_features = model.sam_image_encoder(dummy_sam_image)
            
            print(f"SAM画像特徴量形状: {sam_features.shape}")
            
            # プロンプトエンコーダのテスト
            if has_projector and projector_output_dim == expected_dim:
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=projected_output.unsqueeze(1).to(torch.float32)
                    )
                
                print(f"SAMスパース埋め込み形状: {sparse_embeddings.shape}")
                print(f"SAMデンス埋め込み形状: {dense_embeddings.shape}")
                print("\n✅ SUCCESS: SAMコンポーネントの次元整合性が確認されました")
        
        print("-" * 80)
        
        # 4. 学習可能パラメータのチェック
        print("\n[学習可能パラメータの分析]:")
        
        total_params = 0
        trainable_params = 0
        
        trainable_modules = {
            'gemma_embeddings': [],
            'gemma_lm_head': [],
            'sam_image_encoder': [],
            'sam_prompt_encoder': [],
            'sam_mask_decoder': [],
            'mlp_projector': [],
            'others': []
        }
        
        # パラメータの分類
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
                # モジュールごとに分類
                if 'gemma_model' in name and ('embed_tokens' in name or 'input_embeddings' in name):
                    trainable_modules['gemma_embeddings'].append(name)
                elif 'gemma_model' in name and ('lm_head' in name or 'output_embeddings' in name):
                    trainable_modules['gemma_lm_head'].append(name)
                elif 'sam_image_encoder' in name:
                    trainable_modules['sam_image_encoder'].append(name)
                elif 'sam_prompt_encoder' in name:
                    trainable_modules['sam_prompt_encoder'].append(name)
                elif 'sam_mask_decoder' in name:
                    trainable_modules['sam_mask_decoder'].append(name)
                elif 'mlp_projector' in name:
                    trainable_modules['mlp_projector'].append(name)
                else:
                    trainable_modules['others'].append(name)
        
        # 統計情報の表示
        print(f"\n総パラメータ数: {total_params / 1e6:.2f}M")
        print(f"学習可能パラメータ数: {trainable_params / 1e6:.2f}M")
        print(f"学習可能率: {100 * trainable_params / total_params:.2f}%")
        
        print("\n\n学習可能パラメータグループ:")
        for module_name, params in trainable_modules.items():
            if params:
                print(f"\n[{module_name}]: {len(params)}個の学習可能パラメータ")
                # 最初の5個だけ表示
                for i, p_name in enumerate(params[:5]):
                    print(f"  - {p_name}")
                if len(params) > 5:
                    print(f"  ... 他 {len(params) - 5} 個")
        
        # 期待される学習設定の確認
        print("\n\n[学習設定の検証]:")
        
        # SAM Image Encoderの凍結確認
        sam_encoder_frozen = True
        if has_sam_encoder:
            for param in model.sam_image_encoder.parameters():
                if param.requires_grad:
                    sam_encoder_frozen = False
                    break
        
        # SAM Prompt Encoderの凍結確認
        sam_prompt_frozen = True
        if has_sam_prompt:
            for param in model.sam_prompt_encoder.parameters():
                if param.requires_grad:
                    sam_prompt_frozen = False
                    break
        
        # SAM Mask Decoderの学習可能確認
        sam_decoder_trainable = False
        if has_sam_decoder:
            for param in model.sam_mask_decoder.parameters():
                if param.requires_grad:
                    sam_decoder_trainable = True
                    break
        
        # MLP Projectorの学習可能確認
        projector_trainable = False
        if has_projector:
            for param in model.mlp_projector.parameters():
                if param.requires_grad:
                    projector_trainable = True
                    break
        
        # Gemmaモデルの状態確認
        gemma_main_frozen = True
        gemma_embeddings_trainable = False
        if has_gemma:
            # 埋め込み層以外のパラメータをチェック
            for name, param in model.gemma_model.named_parameters():
                if param.requires_grad:
                    if 'embed_tokens' in name or 'lm_head' in name:
                        gemma_embeddings_trainable = True
                    else:
                        gemma_main_frozen = False
        
        print("\n期待される設定との比較:")
        print(f"  SAM Image Encoder: {'✅ 凍結' if sam_encoder_frozen else '❌ 学習可能（期待: 凍結）'}")
        print(f"  SAM Prompt Encoder: {'✅ 凍結' if sam_prompt_frozen else '❌ 学習可能（期待: 凍結）'}")
        print(f"  SAM Mask Decoder: {'✅ 学習可能' if sam_decoder_trainable else '❌ 凍結（期待: 学習可能）'}")
        print(f"  MLP Projector: {'✅ 学習可能' if projector_trainable else '❌ 凍結（期待: 学習可能）'}")
        print(f"  Gemma本体: {'✅ 凍結' if gemma_main_frozen else '❌ 学習可能（期待: 凍結）'}")
        print(f"  Gemma埋め込み層: {'✅ 学習可能' if gemma_embeddings_trainable else '❌ 凍結（期待: 学習可能）'}")
        
        # 5. 追加の診断情報
        print("\n\n[追加診断情報]:")
        
        # デバイスとデータ型の確認
        print("\nモジュールのデバイスとデータ型:")
        if has_gemma:
            gemma_param = next(model.gemma_model.parameters())
            print(f"  Gemma Model: device={gemma_param.device}, dtype={gemma_param.dtype}")
        if has_sam_encoder:
            sam_param = next(model.sam_image_encoder.parameters())
            print(f"  SAM Encoder: device={sam_param.device}, dtype={sam_param.dtype}")
        if has_projector:
            proj_param = next(model.mlp_projector.parameters())
            print(f"  MLP Projector: device={proj_param.device}, dtype={proj_param.dtype}")
        
        # メモリ使用量の推定
        if torch.cuda.is_available():
            print(f"\nGPUメモリ使用量: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        
        print("\n" + "="*80)
        print("🎉 第3節検証完了: 視覚-言語アライメントのためのアーキテクチャ監査")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 検証中にエラーが発生しました: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 