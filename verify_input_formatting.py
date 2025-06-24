#!/usr/bin/env python3
"""
第2節：Gemma向けマルチモーダル入力フォーマットの検証
完全版: ラベルマスキング検証強化

DataLoaderから完全に処理済みのバッチを取得し、トークン化、特殊トークンの配置、
そして損失計算のためのラベルマスキングが正しく行われているかを検証する。

論理的根拠:
- LlavaからGemmaへのVLM変更に伴う入力フォーマットの変換が正しいことを確認
- 特殊トークン（<image>, <seg>）の配置とラベルマスキング（-100）の整合性検証
- 学習が静かに失敗することを防ぐため、最終的な入力形式を詳細に検査
"""

import argparse
import yaml
import json
import os
import sys
from datetime import datetime
import traceback
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append('.')

# utils から必要な関数をインポート
from utils.dataset import HybridDataset, collate_fn

def load_config_file(config_path: str) -> Dict[str, Any]:
    """設定ファイルを読み込む"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Python設定ファイルの場合
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # 設定をディクショナリ形式に変換
        config_dict = {
            'model_args': {
                'model_path': getattr(config_module, 'MODEL_PATH', 'google/gemma-3-4b-it'),
                'model_max_length': getattr(config_module, 'MODEL_MAX_LENGTH', 2048),
                'gemma_image_size': getattr(config_module, 'GEMMA_IMAGE_SIZE', 896),
                'sam_image_size': getattr(config_module, 'SAM_IMAGE_SIZE', 1024),
            },
            'data_args': {
                'base_image_dir': getattr(config_module, 'DATASET_BASE_DIR', './dataset'),
                'train_datasets': 'reason_seg',  # テスト用に1つのデータセットを指定
            }
        }
        return config_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Gemma3入力フォーマット検証")
    parser.add_argument("--config", type=str, default="config_small_test.py", 
                       help="設定ファイルのパス")
    parser.add_argument("--num_samples_to_check", type=int, default=2, 
                       help="検証するサンプル数")
    parser.add_argument("--output_dir", type=str, default="verification_output",
                       help="出力ディレクトリ")
    return parser.parse_args()

def inspect_special_tokens(tokenizer, dataset_tokenizer=None):
    """トークナイザーの特殊トークンを検査"""
    print("\n[特殊トークン情報]:")
    print(f"  - BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  - EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  - PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  - UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    
    # Gemma-3特有のトークン
    try:
        start_turn_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
        end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>") 
        print(f"  - <start_of_turn>: ID {start_turn_id}")
        print(f"  - <end_of_turn>: ID {end_turn_id}")
    except:
        print("  - Gemma-3ターントークンが見つかりません")
    
    # SEGトークン
    seg_token_id = None
    # まずデータセットのトークナイザーから確認
    if dataset_tokenizer and "[SEG]" in dataset_tokenizer.get_vocab():
        seg_token_id = dataset_tokenizer.convert_tokens_to_ids("[SEG]")
        print(f"  - [SEG] token: ID {seg_token_id} (データセットで追加)")
    elif "[SEG]" in tokenizer.get_vocab():
        seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
        print(f"  - [SEG] token: ID {seg_token_id}")
    else:
        print("  - [SEG]トークンが見つかりません（モデル初期化時に追加されます）")
        # HybridDatasetで一般的に使用されるID
        seg_token_id = 262145  # 既知のデフォルト値
    
    # 画像トークン（固定256個）
    print(f"  - 画像トークン数: 256固定")
    
    return {
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'unk_token_id': tokenizer.unk_token_id,
        'seg_token_id': seg_token_id
    }

def analyze_token_sequence(input_ids, labels, tokenizer, special_tokens, sample_idx=0, show_full=False):
    """トークンシーケンスの詳細分析"""
    print(f"\n===== Sample {sample_idx + 1} トークン分析 =====")
    
    # 基本統計
    seq_len = len(input_ids)
    print(f"\n[シーケンス統計]:")
    print(f"  - シーケンス長: {seq_len}")
    print(f"  - 非パディングトークン数: {(input_ids != special_tokens['pad_token_id']).sum().item()}")
    print(f"  - ラベル付きトークン数: {(labels != -100).sum().item()}")
    
    # 特殊トークンの位置を検出
    print("\n[特殊トークン位置]:")
    
    # BOSトークン
    bos_positions = (input_ids == special_tokens['bos_token_id']).nonzero(as_tuple=True)[0]
    if len(bos_positions) > 0:
        print(f"  - BOSトークン: 位置 {bos_positions.tolist()}")
    
    # EOSトークン  
    eos_positions = (input_ids == special_tokens['eos_token_id']).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        print(f"  - EOSトークン: 位置 {eos_positions.tolist()}")
    
    # SEGトークン
    if special_tokens['seg_token_id'] is not None:
        seg_positions = (input_ids == special_tokens['seg_token_id']).nonzero(as_tuple=True)[0]
        if len(seg_positions) > 0:
            print(f"  - SEGトークン: 位置 {seg_positions.tolist()}")
    
    # Gemma-3ターントークン
    try:
        start_turn_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
        end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        
        start_positions = (input_ids == start_turn_id).nonzero(as_tuple=True)[0]
        end_positions = (input_ids == end_turn_id).nonzero(as_tuple=True)[0]
        
        if len(start_positions) > 0:
            print(f"  - <start_of_turn>: 位置 {start_positions.tolist()}")
        if len(end_positions) > 0:
            print(f"  - <end_of_turn>: 位置 {end_positions.tolist()}")
    except:
        pass
    
    # デコードされた入力の表示
    print("\n[デコードされた入力]:")
    try:
        # 負のトークンIDを無視してデコード
        valid_input_ids = input_ids.clone()
        # 負の値を適切な値に置換
        negative_mask = valid_input_ids < 0
        if negative_mask.any():
            print(f"  ⚠️ 負のトークンID検出: {negative_mask.sum().item()}個（画像トークンと推定）")
            # 負の値をUNKトークンIDに置換
            valid_input_ids[negative_mask] = special_tokens.get('unk_token_id', 3)
        
        decoded_input = tokenizer.decode(valid_input_ids, skip_special_tokens=False)
        # 長すぎる場合は省略
        if len(decoded_input) > 500 and not show_full:
            print(f"{decoded_input[:250]}...（中略）...{decoded_input[-250:]}")
        else:
            print(decoded_input)
    except Exception as e:
        print(f"デコードエラー: {e}")
        # フォールバック: 一部のみをデコード
        try:
            non_neg_mask = input_ids >= 0
            if non_neg_mask.any():
                valid_tokens = input_ids[non_neg_mask][:50]  # 最初の50個の有効トークンのみ
                fallback_decode = tokenizer.decode(valid_tokens, skip_special_tokens=False)
                print(f"部分デコード（最初の50個の有効トークン）: {fallback_decode}")
        except:
            print("デコード完全失敗 - トークンIDのみ表示")
    
    # トークンとラベルの詳細表示
    print("\n[トークン・ラベル対応表]:")
    print(f"{'位置':<6} | {'トークン':<20} | {'Token ID':<10} | {'Label':<10} | {'説明':<30}")
    print("-" * 80)
    
    # 表示するトークン範囲を決定
    if show_full:
        # 完全表示モード：全トークンを表示
        display_indices = list(range(seq_len))
    else:
        # 重要部分のみ表示
        important_indices = []
        
        # 非パディング部分をすべて含める
        non_pad_end = seq_len
        for i in range(seq_len - 1, -1, -1):
            if input_ids[i] != special_tokens['pad_token_id']:
                non_pad_end = i + 1
                break
        
        # 非パディング部分が50トークン以下なら全部表示
        if non_pad_end <= 50:
            important_indices = list(range(non_pad_end))
        else:
            # 最初の15トークン
            important_indices.extend(range(15))
            
            # SEGトークン周辺（前後3トークン）
            if special_tokens['seg_token_id'] is not None:
                seg_positions = (input_ids == special_tokens['seg_token_id']).nonzero(as_tuple=True)[0]
                for pos in seg_positions:
                    pos = pos.item()
                    important_indices.extend(range(max(0, pos-3), min(seq_len, pos+4)))
            
            # modelターン開始周辺
            try:
                model_id = tokenizer.convert_tokens_to_ids("model")
                model_positions = (input_ids == model_id).nonzero(as_tuple=True)[0]
                for pos in model_positions:
                    pos = pos.item()
                    important_indices.extend(range(max(0, pos-2), min(seq_len, pos+10)))
            except:
                pass
            
            # 最後の10トークン（パディング前）
            important_indices.extend(range(max(0, non_pad_end-10), non_pad_end))
        
        # 重複を削除してソート
        display_indices = sorted(set(important_indices))
    
    # トークン表示
    prev_idx = -2
    for idx in display_indices:
        if idx >= seq_len:
            continue
        
        # 省略表示（3トークン以上離れていたら）
        if not show_full and idx > prev_idx + 3:
            print(f"... ({prev_idx+1}～{idx-1}番目のトークンを省略) ...")
        
        token_id = input_ids[idx].item()
        label_id = labels[idx].item()
        
        # トークンテキストを取得
        try:
            if token_id == special_tokens['pad_token_id']:
                token_text = "[PAD]"
            elif token_id == special_tokens['bos_token_id']:
                token_text = "[BOS]"
            elif token_id == special_tokens['eos_token_id']:
                token_text = "[EOS]"
            elif special_tokens['seg_token_id'] and token_id == special_tokens['seg_token_id']:
                token_text = "[SEG]"
            elif token_id < 0:
                token_text = f"[IMAGE_TOKEN:{token_id}]"  # 負のトークンIDは画像トークン
            else:
                try:
                    token_text = tokenizer.decode([token_id])
                    # 改行やタブを可視化
                    token_text = token_text.replace('\n', '\\n').replace('\t', '\\t')
                    if len(token_text) > 20:
                        token_text = token_text[:17] + "..."
                except:
                    token_text = f"[ID:{token_id}]"
        except:
            token_text = f"[ID:{token_id}]"
        
        # ラベルの説明
        if label_id == -100:
            label_desc = "無視（損失計算外）"
        else:
            label_desc = "予測対象"
        
        print(f"{idx:<6} | {token_text:<20} | {token_id:<10} | {label_id:<10} | {label_desc:<30}")
        
        prev_idx = idx
    
    # ラベルマスキングの検証
    print("\n[ラベルマスキング検証]:")
    
    # ターン境界を正確に検出
    try:
        # トークンIDベースでターンを検出
        start_turn_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
        end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        user_id = tokenizer.convert_tokens_to_ids("user")
        model_id = tokenizer.convert_tokens_to_ids("model")
        
        # userターンとmodelターンの境界を探す
        user_turn_start = None
        model_turn_start = None
        
        for i in range(len(input_ids) - 2):
            if (input_ids[i] == start_turn_id and 
                i + 1 < len(input_ids) and input_ids[i + 1] == user_id):
                user_turn_start = i
            elif (input_ids[i] == start_turn_id and 
                  i + 1 < len(input_ids) and input_ids[i + 1] == model_id):
                model_turn_start = i
                break
        
        if user_turn_start is not None and model_turn_start is not None:
            print("  ✓ userターンとmodelターンが検出されました")
            
            # userターンのマスキング検証
            user_turn_end = model_turn_start
            user_labels = labels[user_turn_start:user_turn_end]
            user_masked_count = (user_labels == -100).sum().item()
            user_total = len(user_labels)
            print(f"  - userターン: {user_masked_count}/{user_total} トークンがマスク済み")
            
            # modelターンの予測対象検証
            model_labels = labels[model_turn_start:]
            # パディングを除外
            non_pad_mask = input_ids[model_turn_start:] != special_tokens['pad_token_id']
            model_labels_non_pad = model_labels[non_pad_mask]
            
            model_predicted_count = (model_labels_non_pad != -100).sum().item()
            model_total = len(model_labels_non_pad)
            print(f"  - modelターン: {model_predicted_count}/{model_total} トークンが予測対象")
            
            # 詳細な問題検出
            if user_masked_count < user_total:
                print(f"  ⚠️ userターンに{user_total - user_masked_count}個の予測対象トークンが存在")
        else:
            print("  ⚠️ ターン境界の検出に失敗しました")
            
    except Exception as e:
        print(f"  ⚠️ ターン検出エラー: {e}")
    
    return {
        'seq_len': seq_len,
        'non_pad_tokens': (input_ids != special_tokens['pad_token_id']).sum().item(),
        'labeled_tokens': (labels != -100).sum().item(),
        'has_seg_token': special_tokens['seg_token_id'] is not None and (input_ids == special_tokens['seg_token_id']).any().item()
    }

def main():
    args = parse_args()
    
    print("="*80)
    print("第2節: Gemma向けマルチモーダル入力フォーマットの検証")
    print("="*80)
    
    try:
        # 設定ファイルの読み込み
        print(f"設定ファイル読み込み: {args.config}")
        config = load_config_file(args.config)
        
        model_args = config.get('model_args', {})
        data_args = config.get('data_args', {})
        
        print(f"✅ 設定読み込み完了")
        print(f"  モデル: {model_args.get('model_path', 'N/A')}")
        print(f"  最大長: {model_args.get('model_max_length', 'N/A')}")
        print(f"  データセット: {data_args.get('train_datasets', 'N/A')}")
        
        # セッションタイムスタンプの生成
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("🔍 LISA-Gemma Input Formatting Verification")
        print("=" * 60)
        print(f"🚀 検証セッション開始: {session_timestamp}")
        print("=" * 60)
        
        # 1. Tokenizerとプロセッサーの初期化
        print("\n📝 Tokenizer/Processorを初期化中...")
        model_id = model_args.get('model_path', 'google/gemma-3-4b-it')
        
        # AutoProcessorのインポートと初期化
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = processor.tokenizer
        
        # パディングトークンの設定（Gemmaはデフォルトでpadトークンを持たない場合がある）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"  ⚠️ パディングトークンをEOSトークンに設定: {tokenizer.pad_token}")
        
        # SEGトークンの追加確認
        if "[SEG]" not in tokenizer.get_vocab():
            print("  ⚠️ [SEG]トークンが見つかりません。追加をスキップします。")
        else:
            print(f"  ✓ [SEG]トークンが検出されました: ID {tokenizer.convert_tokens_to_ids('[SEG]')}")
        
        # 特殊トークンの検査（まずはプロセッサのトークナイザーのみで）
        special_tokens = inspect_special_tokens(tokenizer)
        
        # 2. データセットとDataLoaderの準備
        print("\n📦 データセットを準備中...")
        
        # HybridDatasetを使用（トークン化とDataCollatorとの互換性のため）
        dataset = HybridDataset(
            base_image_dir=data_args.get('base_image_dir', './dataset'),
            gemma_processor=processor,  # Gemmaプロセッサーを使用
            dataset=data_args.get('train_datasets', 'sem_seg'),  # テスト用に1つのデータセットタイプのみ使用
            samples_per_epoch=50  # テスト用に少数のサンプル
        )
        
        print(f"  ✓ データセット初期化完了（サンプル数: {len(dataset)}）")
        
        # DataLoaderの作成
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_fn,  # オリジナルのcollate_fnを使用
            shuffle=False  # 再現性のため
        )
        
        print(f"  ✓ DataLoader準備完了（バッチサイズ: 4）")
        
        # 3. バッチの取得と検査
        print("\n🔬 バッチデータを取得中...")
        batch = next(iter(dataloader))
        print("  ✓ バッチ取得成功")
        
        # バッチの基本情報
        print("\n[バッチ情報]:")
        # 重要なキーを優先して表示
        important_keys = ['input_ids', 'labels', 'attention_masks', 'images_for_gemma', 'images_for_sam', 
                          'ground_truth_mask', 'seg_token_mask', 'text_prompts']
        other_keys = []
        
        for key in important_keys:
            if key in batch:
                value = batch[key]
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  - {key}: {type(value)}")
        
        # その他のキーも表示
        for key, value in batch.items():
            if key not in important_keys:
                other_keys.append(key)
        
        if other_keys:
            print(f"  - その他のキー: {', '.join(other_keys)}")
        
        # 必須フィールドの確認（HybridDatasetの返り値形式に合わせて更新）
        required_fields = ['input_ids', 'labels', 'attention_masks']  # attention_masksに修正
        missing_fields = [f for f in required_fields if f not in batch]
        if missing_fields:
            print(f"\n❌ 必須フィールドが不足: {missing_fields}")
            print(f"利用可能なキー: {list(batch.keys())}")
            return
        
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch.get('attention_masks')  # attention_masksに修正
        
        # 4. 各サンプルの詳細検査
        num_to_check = min(args.num_samples_to_check, input_ids.shape[0])
        
        analysis_results = []
        for i in range(num_to_check):
            result = analyze_token_sequence(
                input_ids[i], 
                labels[i], 
                tokenizer, 
                special_tokens,
                sample_idx=i,
                show_full=False
            )
            analysis_results.append(result)
        
        # 5. 統計サマリー
        print("\n" + "=" * 60)
        print("📊 検証サマリー")
        print("=" * 60)
        
        print("\n[バッチ全体の統計]:")
        total_samples = input_ids.shape[0]
        avg_seq_len = sum(r['seq_len'] for r in analysis_results) / len(analysis_results)
        avg_labeled = sum(r['labeled_tokens'] for r in analysis_results) / len(analysis_results)
        
        # SEGトークンを再確認（直接バッチデータから）
        samples_with_seg = 0
        # special_tokensから取得したIDを使用
        seg_token_id = special_tokens.get('seg_token_id', 262145)  # デフォルト値も提供
        for i in range(num_to_check):
            if seg_token_id is not None and (input_ids[i] == seg_token_id).any().item():
                samples_with_seg += 1
        
        print(f"  - 総サンプル数: {total_samples}")
        print(f"  - 平均シーケンス長: {avg_seq_len:.1f}")
        print(f"  - 平均ラベル付きトークン数: {avg_labeled:.1f}")
        print(f"  - SEGトークンを含むサンプル数: {samples_with_seg}/{num_to_check}")
        
        # 期待される形式との整合性チェック
        print("\n[期待される形式との整合性]:")
        
        # Gemma-3チャットテンプレートの確認
        all_have_turns = True
        for i in range(num_to_check):
            try:
                # 負のトークンIDを除去してからデコード
                sample_ids = input_ids[i].clone()
                negative_mask = sample_ids < 0
                if negative_mask.any():
                    # 負の値をUNKトークンに置換
                    sample_ids[negative_mask] = special_tokens.get('unk_token_id', 3)
                
                decoded = tokenizer.decode(sample_ids, skip_special_tokens=False)
                if "<start_of_turn>" not in decoded or "<end_of_turn>" not in decoded:
                    all_have_turns = False
                    break
            except Exception as e:
                print(f"    サンプル{i+1}のデコードエラー: {e}")
                all_have_turns = False
                break
        
        if all_have_turns:
            print("  ✅ Gemma-3チャットテンプレート形式: 適合")
        else:
            print("  ❌ Gemma-3チャットテンプレート形式: 不適合")
        
        # ラベルマスキングの詳細確認
        masking_issues = []
        masking_details = []
        for i in range(num_to_check):
            # 各サンプルでuserターンとmodelターンを検出
            try:
                start_turn_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
                end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
                user_id = tokenizer.convert_tokens_to_ids("user")
                model_id = tokenizer.convert_tokens_to_ids("model")
                
                # userターンの正確な範囲を検出
                user_turn_start = None
                user_turn_end = None
                model_turn_start = None
                
                for j in range(len(input_ids[i]) - 1):
                    if input_ids[i][j] == start_turn_id and input_ids[i][j+1] == user_id:
                        user_turn_start = j
                    elif user_turn_start is not None and input_ids[i][j] == end_turn_id:
                        # userターンの<end_of_turn>を見つけた
                        user_turn_end = j + 1  # <end_of_turn>の次まで含める
                        # 次の改行も含める場合
                        if j + 1 < len(input_ids[i]) and tokenizer.decode([input_ids[i][j+1].item()]) == '\n':
                            user_turn_end = j + 2
                    elif input_ids[i][j] == start_turn_id and input_ids[i][j+1] == model_id:
                        model_turn_start = j
                        break
                
                if user_turn_start is not None and user_turn_end is not None:
                    # userターン範囲でラベルをチェック
                    user_turn_labels = labels[i][user_turn_start:user_turn_end]
                    unmasked_indices = (user_turn_labels != -100).nonzero(as_tuple=True)[0]
                    
                    if len(unmasked_indices) > 0:
                        # 問題のあるトークンを特定
                        problem_tokens = []
                        for idx in unmasked_indices:
                            abs_idx = user_turn_start + idx.item()
                            token_text = tokenizer.decode([input_ids[i][abs_idx].item()])
                            problem_tokens.append(f"'{token_text}' (pos {abs_idx})")
                        
                        masking_issues.append(f"サンプル{i+1}: userターンに{len(unmasked_indices)}個の予測対象トークン")
                        masking_details.append(f"   問題のトークン: {', '.join(problem_tokens)}")
                        
            except Exception as e:
                masking_issues.append(f"サンプル{i+1}: ラベルマスキング検証エラー: {e}")
        
        if not masking_issues:
            print("  ✅ ラベルマスキング: 完璧（全userターンが-100でマスク済み）")
        else:
            print("  ⚠️ ラベルマスキング: 問題検出")
            print("     [期待される挙動]: userターン全体（BOSから<end_of_turn>\\nまで）は-100でマスクされるべき")
            for issue, detail in zip(masking_issues, masking_details):
                print(f"     - {issue}")
                if detail:
                    print(f"       {detail}")
        
        # 画像トークンの確認（256個の連続した特殊トークン）
        print("  ℹ️ 画像トークン: Gemma-3では256個の固定トークンとして処理")
        
        # 推奨事項
        if masking_issues:
            print("\n[推奨される修正]:")
            print("  1. utils/conversation.pyまたはdata_processing.pyでラベルマスキングロジックを確認")
            print("  2. userターンの<end_of_turn>トークンとその後の改行も-100でマスクするよう修正")
            print("  3. modelターンの開始位置から予測対象とするよう境界条件を調整")
        
        # 6. 結果の保存
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        result_file = os.path.join(output_dir, f"input_formatting_verification_{session_timestamp}.json")
        
        verification_result = {
            "session_info": {
                "timestamp": session_timestamp,
                "model_id": model_id,
                "batch_size": 4,
                "num_samples_checked": num_to_check
            },
            "tokenizer_info": {
                "vocab_size": tokenizer.vocab_size,
                "special_tokens": {
                    "bos": str(tokenizer.bos_token),
                    "eos": str(tokenizer.eos_token),
                    "pad": str(tokenizer.pad_token),
                    "has_seg_token": "[SEG]" in tokenizer.get_vocab()
                }
            },
            "batch_statistics": {
                "total_samples": total_samples,
                "avg_sequence_length": avg_seq_len,
                "avg_labeled_tokens": avg_labeled,
                "samples_with_seg_token": samples_with_seg
            },
            "format_compliance": {
                "gemma3_chat_template": all_have_turns,
                "proper_label_masking": len(masking_issues) == 0,
                "masking_issues": masking_issues
            }
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(verification_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 検証結果を保存: {result_file}")
        
        print(f"\n🎉 第2節検証完了: Gemma向けマルチモーダル入力フォーマットの検証")
        print(f"🕐 セッション終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n" + "="*60)
        print("3. 詳細なラベルマスキング検証（Gemma3準拠）")
        print("="*60)
        
        # 特殊トークンIDを動的に取得
        start_of_turn_id = None
        end_of_turn_id = None
        user_id = None
        model_id = None
        
        try:
            start_of_turn_id = processor.tokenizer.convert_tokens_to_ids("<start_of_turn>")
            end_of_turn_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            user_id = processor.tokenizer.convert_tokens_to_ids("user")
            model_id = processor.tokenizer.convert_tokens_to_ids("model")
        except:
            # フォールバック値
            start_of_turn_id = 106
            end_of_turn_id = 107
            user_id = 2188
            model_id = 2091
        
        print(f"特殊トークンID:")
        print(f"  <start_of_turn>: {start_of_turn_id}")
        print(f"  <end_of_turn>: {end_of_turn_id}")
        print(f"  user: {user_id}")
        print(f"  model: {model_id}")
        
        total_correct_samples = 0
        total_samples = min(args.num_samples_to_check, input_ids.shape[0])
        
        for i in range(total_samples):
            print(f"\n--- サンプル {i+1}/{total_samples} の詳細分析 ---")
            
            sample_input_ids = input_ids[i].tolist()
            sample_labels = labels[i].tolist()
            
            # パディングを除外
            non_pad_length = 0
            for j, token_id in enumerate(sample_input_ids):
                if token_id != processor.tokenizer.pad_token_id:
                    non_pad_length = j + 1
            
            sample_input_ids = sample_input_ids[:non_pad_length]
            sample_labels = sample_labels[:non_pad_length]
            
            print(f"有効トークン数: {non_pad_length}")
            
            # ターン境界を検出
            turn_analysis = []
            i_tok = 0
            current_turn = None
            turn_start = 0
            
            while i_tok < len(sample_input_ids):
                if sample_input_ids[i_tok] == start_of_turn_id:
                    # 前のターンを終了
                    if current_turn is not None:
                        turn_analysis.append({
                            'type': current_turn,
                            'start': turn_start,
                            'end': i_tok - 1,
                            'tokens': sample_input_ids[turn_start:i_tok],
                            'labels': sample_labels[turn_start:i_tok]
                        })
                    
                    # 新しいターンを開始
                    if i_tok + 1 < len(sample_input_ids):
                        next_token = sample_input_ids[i_tok + 1]
                        if next_token == user_id:
                            current_turn = 'user'
                            turn_start = i_tok
                        elif next_token == model_id:
                            current_turn = 'model'
                            turn_start = i_tok
                        else:
                            current_turn = 'unknown'
                            turn_start = i_tok
                    else:
                        current_turn = 'incomplete'
                        turn_start = i_tok
                i_tok += 1
            
            # 最後のターンを追加
            if current_turn is not None:
                turn_analysis.append({
                    'type': current_turn,
                    'start': turn_start,
                    'end': len(sample_input_ids) - 1,
                    'tokens': sample_input_ids[turn_start:],
                    'labels': sample_labels[turn_start:]
                })
            
            # ターン別検証
            sample_correct = True
            for turn in turn_analysis:
                turn_type = turn['type']
                turn_tokens = turn['tokens']
                turn_labels = turn['labels']
                
                print(f"\n{turn_type.upper()}ターン (位置 {turn['start']}-{turn['end']}):")
                
                if turn_type == 'user':
                    # userターンは全て-100であるべき
                    masked_count = sum(1 for label in turn_labels if label == -100)
                    total_count = len(turn_labels)
                    
                    if masked_count == total_count:
                        print(f"  ✅ 正しくマスク済み: {masked_count}/{total_count} トークン")
                    else:
                        print(f"  ❌ マスクエラー: {masked_count}/{total_count} トークンがマスクされている")
                        sample_correct = False
                        
                        # エラー詳細を表示
                        for j, (token_id, label) in enumerate(zip(turn_tokens, turn_labels)):
                            if label != -100:
                                token_text = processor.tokenizer.decode([token_id])
                                print(f"    位置{turn['start']+j}: '{token_text}' (ID:{token_id}) -> label:{label} (期待:-100)")
                
                elif turn_type == 'model':
                    # modelターンの構造を分析
                    # <start_of_turn>model の部分は-100、応答部分は予測対象
                    
                    prediction_start = -1
                    prediction_end = -1
                    
                    # <start_of_turn>model の直後から予測開始を探す
                    for j, token_id in enumerate(turn_tokens):
                        if j >= 2:  # <start_of_turn>model の後
                            if turn_tokens[j] != end_of_turn_id:  # <end_of_turn>でない
                                if prediction_start == -1:
                                    prediction_start = j
                                prediction_end = j
                            else:
                                break
                    
                    mask_errors = []
                    pred_errors = []
                    
                    for j, (token_id, label) in enumerate(zip(turn_tokens, turn_labels)):
                        token_text = processor.tokenizer.decode([token_id])
                        
                        if j < 2:  # <start_of_turn>model 部分
                            if label != -100:
                                mask_errors.append(f"位置{turn['start']+j}: '{token_text}' -> label:{label} (期待:-100)")
                        elif prediction_start <= j <= prediction_end:  # 応答部分
                            if label != token_id:
                                pred_errors.append(f"位置{turn['start']+j}: '{token_text}' -> label:{label} (期待:{token_id})")
                        else:  # <end_of_turn>など
                            if label != -100:
                                mask_errors.append(f"位置{turn['start']+j}: '{token_text}' -> label:{label} (期待:-100)")
                    
                    if not mask_errors and not pred_errors:
                        print(f"  ✅ 正しいラベル設定")
                        if prediction_start != -1:
                            print(f"    予測対象: 位置{turn['start']+prediction_start}-{turn['start']+prediction_end}")
                    else:
                        sample_correct = False
                        if mask_errors:
                            print(f"  ❌ マスクエラー:")
                            for error in mask_errors[:3]:  # 最初の3つまで表示
                                print(f"    {error}")
                        if pred_errors:
                            print(f"  ❌ 予測ラベルエラー:")
                            for error in pred_errors[:3]:  # 最初の3つまで表示
                                print(f"    {error}")
            
            if sample_correct:
                total_correct_samples += 1
                print(f"\n✅ サンプル{i+1}: ラベルマスキング正常")
            else:
                print(f"\n❌ サンプル{i+1}: ラベルマスキングに問題あり")
        
        print(f"\n" + "="*60)
        print(f"最終結果: {total_correct_samples}/{total_samples} サンプルが正しいラベルマスキング")
        if total_correct_samples == total_samples:
            print("🎉 全サンプルでラベルマスキングが正常です！")
        else:
            print("⚠️  一部サンプルでラベルマスキングに問題があります")
        print("="*60)
        
        # 出力ディレクトリの作成
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 検証結果をJSONファイルに保存
        output_file = os.path.join(args.output_dir, f"input_formatting_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        verification_results = {
            'timestamp': datetime.now().isoformat(),
            'config_file': args.config,
            'model_path': model_args.get('model_path'),
            'num_samples_checked': total_samples,
            'correct_samples': total_correct_samples,
            'success_rate': total_correct_samples / total_samples if total_samples > 0 else 0,
            'special_tokens': {
                'start_of_turn': start_of_turn_id,
                'end_of_turn': end_of_turn_id,
                'user': user_id,
                'model': model_id,
                'seg_token': seg_token_id if 'seg_token_id' in locals() else None,
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(verification_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 検証結果を保存: {output_file}")
        
        # 最終サマリー
        print(f"\n" + "🎯 "*20)
        print(f"第2節検証完了！")
        print(f"  正常サンプル: {total_correct_samples}/{total_samples}")
        print(f"  成功率: {total_correct_samples/total_samples*100:.1f}%")
        if total_correct_samples == total_samples:
            print(f"  🎉 Gemma3入力フォーマット: 完全対応")
        else:
            print(f"  ⚠️  一部改善が必要")
        print(f"🎯 "*20)
        
    except Exception as e:
        print(f"\n❌ 検証中にエラーが発生しました: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 