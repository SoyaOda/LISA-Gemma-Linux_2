# model/gemma_lisa.py
"""
LISA-Gemma3アーキテクチャ
Gemma-3の公式API仕様に準拠した実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

from transformers import AutoProcessor, Gemma3ForConditionalGeneration, PreTrainedModel, PretrainedConfig
from model.segment_anything import sam_model_registry
from model.segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from utils.constants import IMAGE_TOKEN_INDEX, GEMMA_IMAGE_TOKEN_NUM

# LISA-Gemmaモデルのカスタム設定クラス
class LisaGemmaConfig(PretrainedConfig):
    model_type = "lisa_gemma"

    def __init__(
        self,
        gemma_model_id: str = "google/gemma-3-4b-it",
        sam_checkpoint_path: Optional[str] = None,
        seg_token: str = "[SEG]",
        gemma_hidden_size: int = 2560,  # Gemma 3 4B の hidden_size
        sam_prompt_embed_dim: int = 256,
        gemma_image_size: int = 896,
        sam_image_size: int = 1024,
        model_max_length: int = 2048,
        **kwargs,
    ):
        self.gemma_model_id = gemma_model_id
        self.sam_checkpoint_path = sam_checkpoint_path
        self.seg_token = seg_token
        self.gemma_hidden_size = gemma_hidden_size
        self.sam_prompt_embed_dim = sam_prompt_embed_dim
        self.gemma_image_size = gemma_image_size
        self.sam_image_size = sam_image_size
        self.model_max_length = model_max_length
        super().__init__(**kwargs)

class LisaGemmaForCausalLM(PreTrainedModel):
    config_class = LisaGemmaConfig

    def __init__(self, config: LisaGemmaConfig):
        super().__init__(config)

        # 1. Gemma-3 multimodal model の初期化
        print(f"Gemma-3マルチモーダルモデルをロード中... ({config.gemma_model_id})")
        self.gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            config.gemma_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 1.1 仕様書第2章: Gemmaモデル本体のパラメータを凍結
        print("Gemmaモデルのパラメータを仕様書に従って凍結中...")
        for param in self.gemma_model.parameters():
            param.requires_grad = False
        
        # 1.2 例外: 埋め込み層とLMヘッドは訓練可能に（新しいSEGトークン対応）
        if hasattr(self.gemma_model, 'get_input_embeddings'):
            for param in self.gemma_model.get_input_embeddings().parameters():
                param.requires_grad = True
        if hasattr(self.gemma_model, 'get_output_embeddings'):
            for param in self.gemma_model.get_output_embeddings().parameters():
                param.requires_grad = True
        
        print("✅ Gemmaパラメータ凍結が完了（埋め込み層とLMヘッドは訓練可能）")
        
        # 2. Gemma-3用プロセッサーの初期化
        print(f"Gemma-3プロセッサーをロード中... ({config.gemma_model_id})")
        self.gemma_processor = AutoProcessor.from_pretrained(config.gemma_model_id)
        
        # 3. SAMコンポーネントのロードと凍結（SAMチェックポイントが存在する場合のみ）
        if config.sam_checkpoint_path and config.sam_checkpoint_path != "":
            print(f"SAMモデルをロード中... ({config.sam_checkpoint_path})")
            try:
                sam = sam_model_registry["vit_h"](checkpoint=config.sam_checkpoint_path)
                
                # SAMの画像エンコーダを抽出し、凍結する
                self.sam_image_encoder = sam.image_encoder
                for param in self.sam_image_encoder.parameters():
                    param.requires_grad = False
                
                # SAMのプロンプトエンコーダを抽出し、凍結する
                self.sam_prompt_encoder = sam.prompt_encoder
                for param in self.sam_prompt_encoder.parameters():
                    param.requires_grad = False
                
                # SAMのマスクデコーダを抽出し、訓練可能にする
                self.sam_mask_decoder = sam.mask_decoder
                for param in self.sam_mask_decoder.parameters():
                    param.requires_grad = True
                    
                # SAMコンポーネントをGemmaと同じデバイスに移動
                device = next(self.gemma_model.parameters()).device
                self.sam_image_encoder = self.sam_image_encoder.to(device)
                self.sam_prompt_encoder = self.sam_prompt_encoder.to(device)
                self.sam_mask_decoder = self.sam_mask_decoder.to(device)
                
                print("✅ SAMコンポーネントの初期化が完了しました")
                
            except Exception as e:
                print(f"❌ SAMの初期化に失敗しました: {e}")
                print("SAMコンポーネントなしで続行します（セグメンテーション機能は利用できません）")
                self.sam_image_encoder = None
                self.sam_prompt_encoder = None
                self.sam_mask_decoder = None
        else:
            print("⚠️  SAMチェックポイントが指定されていません。SAMコンポーネントは初期化されません。")
            self.sam_image_encoder = None
            self.sam_prompt_encoder = None
            self.sam_mask_decoder = None

        # 4. MLPプロジェクタの定義 (GemmaとSAMを繋ぐ橋)
        print("MLPプロジェクタを初期化中...")
        device = next(self.gemma_model.parameters()).device
        self.mlp_projector = nn.Sequential(
            nn.Linear(config.gemma_hidden_size, config.gemma_hidden_size),
            nn.GELU(),
            nn.Linear(config.gemma_hidden_size, config.sam_prompt_embed_dim),
        ).to(device).to(torch.bfloat16)

        # 5. 特別なセグメンテーショントークンを語彙に追加
        print("セグメンテーショントークンを追加中...")
        self.seg_token = config.seg_token
        
        # トークナイザーにSEGトークンを追加
        if self.seg_token not in self.gemma_processor.tokenizer.get_vocab():
            # トークンを追加
            num_added_tokens = self.gemma_processor.tokenizer.add_tokens([self.seg_token], special_tokens=True)
            print(f"✅ {num_added_tokens}個のトークンが追加されました")
        else:
            print(f"✅ {self.seg_token}は既に語彙に存在します")
        
        # SEGトークンのIDを取得
        self.seg_token_id = self.gemma_processor.tokenizer.convert_tokens_to_ids(self.seg_token)
        print(f"SEGトークンID: {self.seg_token_id}")
        
        # 埋め込み層のリサイズ（画像トークン範囲を含む）
        # 必要な語彙サイズを計算
        # 現在の語彙サイズ + 画像トークン範囲（256個）
        current_vocab_size = len(self.gemma_processor.tokenizer)
        required_vocab_size = max(current_vocab_size, IMAGE_TOKEN_INDEX + GEMMA_IMAGE_TOKEN_NUM)
        
        # 埋め込み層の現在のサイズを確認
        actual_embed_size = self.gemma_model.get_input_embeddings().weight.shape[0]
        print(f"現在の埋め込み層サイズ: {actual_embed_size}")
        print(f"現在の語彙サイズ: {current_vocab_size}")
        print(f"必要な語彙サイズ: {required_vocab_size}")
        print(f"画像トークン範囲: {IMAGE_TOKEN_INDEX} - {IMAGE_TOKEN_INDEX + GEMMA_IMAGE_TOKEN_NUM - 1}")
        
        # 埋め込み層のリサイズが必要かチェック
        if actual_embed_size < required_vocab_size:
            print(f"埋め込み層をリサイズ中: {actual_embed_size} -> {required_vocab_size}")
            
            try:
                self.gemma_model.resize_token_embeddings(required_vocab_size)
                print(f"✅ 埋め込み層が正常にリサイズされました（新サイズ: {required_vocab_size}）")
            except RuntimeError as e:
                if "DTensor" in str(e):
                    print(f"⚠️ DeepSpeed環境での実行を検出。埋め込み層のリサイズを延期します")
                else:
                    print(f"❌ 埋め込み層のリサイズに失敗: {e}")
                    raise e
            
            # リサイズ後のサイズを確認
            new_embed_size = self.gemma_model.get_input_embeddings().weight.shape[0]
            print(f"リサイズ後の埋め込み層サイズ: {new_embed_size}")
            
            if new_embed_size < required_vocab_size:
                raise RuntimeError(f"埋め込み層のリサイズに失敗: {new_embed_size} < {required_vocab_size}")
        else:
            print(f"✅ 埋め込み層サイズは十分です: {actual_embed_size} >= {required_vocab_size}")
        
        # 設定情報を保存
        self.gemma_image_size = config.gemma_image_size
        self.sam_image_size = config.sam_image_size
        self.model_max_length = config.model_max_length
        
        print("✅ LISA-Gemmaモデルの初期化が完了しました")

    @classmethod
    def from_config_file(cls, config_path: str, **kwargs):
        """設定ファイルからモデルを初期化するクラスメソッド"""
        import importlib.util
        import sys
        
        # 設定ファイルをモジュールとして読み込み
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["config"] = config_module
        spec.loader.exec_module(config_module)
        
        # 設定からLisaGemmaConfigを作成
        lisa_config = LisaGemmaConfig(
            gemma_model_id=getattr(config_module, 'GEMMA_MODEL_ID', "google/gemma-3-4b-it"),
            sam_checkpoint_path=getattr(config_module, 'SAM_CHECKPOINT_PATH', None),
            seg_token=getattr(config_module, 'SEG_TOKEN', "[SEG]"),
            gemma_hidden_size=getattr(config_module, 'GEMMA_HIDDEN_SIZE', 2560),
            sam_prompt_embed_dim=getattr(config_module, 'SEG_PROJECTION_DIM', 256),
            gemma_image_size=getattr(config_module, 'GEMMA_IMAGE_SIZE', 896),
            sam_image_size=getattr(config_module, 'SAM_IMAGE_SIZE', 1024),
            model_max_length=getattr(config_module, 'MODEL_MAX_LENGTH', 2048),
            **kwargs
        )
        
        return cls(lisa_config)

    def has_sam_capability(self) -> bool:
        """SAMセグメンテーション機能が利用可能かチェック"""
        return all([
            self.sam_image_encoder is not None,
            self.sam_prompt_encoder is not None,
            self.sam_mask_decoder is not None
        ])

    def prepare_multimodal_input(self, image, text_prompt):
        """
        Gemma-3の公式チャットテンプレートに従って入力を準備
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        
        # チャットテンプレートを適用
        inputs = self.gemma_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        return inputs

    def get_trainable_parameters_info(self):
        """訓練可能なパラメータの情報を取得"""
        total_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100 if total_params > 0 else 0
        }
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """PEFT対応のため必要なメソッド"""
        return self.gemma_model.prepare_inputs_for_generation(input_ids, **kwargs)
    
    def get_input_embeddings(self):
        """PEFT対応のため必要なメソッド"""
        return self.gemma_model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """PEFT対応のため必要なメソッド"""
        self.gemma_model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        """PEFT対応のため必要なメソッド"""
        return self.gemma_model.get_output_embeddings()
    
    def set_output_embeddings(self, value):
        """PEFT対応のため必要なメソッド"""
        self.gemma_model.set_output_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images_for_gemma: Optional[torch.FloatTensor] = None,  # デュアルストリーム対応
        images_for_sam: Optional[torch.FloatTensor] = None,    # デュアルストリーム対応
        image=None,  # PIL Image (単一画像用)
        text_prompt: str = None,  # 単一テキスト用
        generate_mask: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        LISA-Gemmaのフォワードパス
        デュアルストリーム・データパイプライン対応版
        """
        try:
            device = next(self.gemma_model.parameters()).device
            
            # 単一画像+テキストの場合（推論時）
            if image is not None and text_prompt is not None:
                return self._forward_single(image, text_prompt, generate_mask, device)
            
            # デュアルストリーム・バッチ処理の場合（学習時）
            if input_ids is not None and (images_for_gemma is not None or pixel_values is not None):
                # デュアルストリーム対応の新しいフォワードパス
                if images_for_gemma is not None and images_for_sam is not None:
                    return self._forward_dual_stream_batch(
                        input_ids, attention_mask, images_for_gemma, images_for_sam, 
                        labels, generate_mask, device
                    )
                # 従来のpixel_values形式との後方互換性
                elif pixel_values is not None:
                    return self._forward_batch(input_ids, attention_mask, pixel_values, labels, generate_mask, device)
            
            raise ValueError("Either (image, text_prompt) or (input_ids, images_for_gemma, images_for_sam) must be provided")
            
        except Exception as e:
            print(f"フォワードパス中にエラーが発生: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _forward_single(self, image, text_prompt, generate_mask, device):
        """単一画像・テキストのフォワードパス（推論用）"""
        # 1. Gemma-3の公式方式で入力を準備
        gemma_inputs = self.prepare_multimodal_input(image, text_prompt)
        
        # デバイスに移動
        gemma_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in gemma_inputs.items()}
        
        # 2. Gemmaモデルでテキスト生成とセグメンテーション判定
        with torch.no_grad():
            gemma_outputs = self.gemma_model(
                **gemma_inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        results = {
            "gemma_logits": gemma_outputs.logits,
            "hidden_states": gemma_outputs.hidden_states,
        }
        
        # 3. SEGトークンが含まれている場合のマスク生成
        if generate_mask and self.sam_image_encoder is not None:
            input_ids = gemma_inputs["input_ids"]
            seg_positions = (input_ids == self.seg_token_id).nonzero(as_tuple=True)
            
            if len(seg_positions[0]) > 0:
                print(f"入力テキストでSEGトークンが{len(seg_positions[0])}個検出されました")
                
                # SAM用の画像前処理（1024x1024にリサイズ）
                sam_image = image.resize((1024, 1024))
                sam_image_tensor = torch.tensor(np.array(sam_image)).permute(2, 0, 1).float()
                sam_image_tensor = sam_image_tensor.unsqueeze(0).to(device)
                
                # SAMの画像エンコーディング
                with torch.no_grad():
                    sam_features = self.sam_image_encoder(sam_image_tensor)
                
                # SEGトークンの隠れ状態を抽出してマスク生成
                masks = self._generate_masks_from_seg_tokens(
                    gemma_outputs.hidden_states[-1], seg_positions, sam_features, device
                )
                results["predicted_masks"] = masks
            else:
                results["predicted_masks"] = None
        
        return results
    
    def _forward_batch(self, input_ids, attention_mask, pixel_values, labels, generate_mask, device):
        """バッチ処理のフォワードパス（学習用）"""
        
        # pixel_valuesが空の場合の処理
        if pixel_values.numel() == 0:
            raise ValueError("pixel_valuesが空です。マルチモーダル処理には画像が必要です。")
        
        # 1. Gemmaモデルでのフォワードパス（画像あり）
        print(f"🔍 Gemmaモデル入力情報:")
        print(f"  - input_ids: {input_ids.shape}")
        print(f"  - attention_mask: {attention_mask.shape}")
        print(f"  - pixel_values: {pixel_values.shape}")
        print(f"  - labels: {labels.shape if labels is not None else 'None'}")
        
        gemma_outputs = self.gemma_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        
        results = {
            "text_loss": gemma_outputs.loss,  # Gemmaのlanguage modeling loss
            "logits": gemma_outputs.logits,
            "hidden_states": gemma_outputs.hidden_states,
        }
        
        # 2. セグメンテーション処理
        if generate_mask and self.sam_image_encoder is not None and pixel_values.numel() > 0:
            # SEGトークンの位置を検出
            seg_positions = (input_ids == self.seg_token_id).nonzero(as_tuple=True)
            
            if len(seg_positions[0]) > 0:
                print(f"バッチ内でSEGトークンが{len(seg_positions[0])}個検出されました")
                
                # バッチ内の画像をSAM用に前処理（pixel_valuesから変換）
                batch_size = pixel_values.shape[0]
                sam_features_list = []
                
                for i in range(batch_size):
                    # Gemma用の画像をSAM用に変換（896x896 -> 1024x1024）
                    gemma_img = pixel_values[i]  # (C, H, W)
                    
                    # SAM用にリサイズ（bilinear補間を使用）
                    sam_img = F.interpolate(
                        gemma_img.unsqueeze(0), 
                        size=(1024, 1024), 
                        mode='bilinear', 
                        align_corners=False
                    )  # (1, C, H, W)
                    
                    # SAM用の正規化（RGB値0-1を0-255に変換）
                    sam_img = sam_img * 255.0
                    
                    # SAMの画像エンコーディング
                    with torch.no_grad():
                        sam_features = self.sam_image_encoder(sam_img)
                    sam_features_list.append(sam_features)
                
                # SEGトークンからマスクを生成
                masks = self._generate_masks_from_seg_tokens_batch(
                    gemma_outputs.hidden_states[-1], seg_positions, sam_features_list, device
                )
                
                # バッチサイズに合わせてマスクを調整
                if masks is not None:
                    # SEGトークンの数がバッチサイズと一致しない場合の処理
                    if masks.shape[0] != batch_size:
                        # 各画像に対してマスクを生成（SEGトークンがない画像にはダミーマスクを作成）
                        batch_masks = []
                        seg_count = 0
                        for i in range(batch_size):
                            # この画像にSEGトークンがあるかチェック
                            has_seg = any(seg_positions[0] == i)
                            if has_seg:
                                batch_masks.append(masks[seg_count])
                                seg_count += 1
                            else:
                                # ダミーマスクを作成
                                dummy_mask = torch.zeros_like(masks[0])
                                batch_masks.append(dummy_mask)
                        masks = torch.stack(batch_masks, dim=0)
                    
                results["predicted_masks"] = masks
            else:
                results["predicted_masks"] = None
        else:
            # SEGトークンが存在する場合、MLPプロジェクタを通して勾配フローを確保
            seg_positions = (input_ids == self.seg_token_id).nonzero(as_tuple=True)
            
            if len(seg_positions[0]) > 0:
                print(f"SEGトークン{len(seg_positions[0])}個でMLPプロジェクタの勾配フローを確保")
                
                # MLPプロジェクタの勾配フローを確保するため
                mlp_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                for batch_idx, token_idx in zip(seg_positions[0], seg_positions[1]):
                    seg_hidden = gemma_outputs.hidden_states[-1][batch_idx, token_idx]
                    seg_embedding = self.mlp_projector(seg_hidden)
                    # 小さなダミー損失を追加（MLPプロジェクタに勾配を流すため）
                    mlp_loss = mlp_loss + seg_embedding.sum() * 1e-6
                
                # テキスト損失にMLP損失を追加
                if results["text_loss"] is not None:
                    results["text_loss"] = results["text_loss"] + mlp_loss
                else:
                    results["text_loss"] = mlp_loss
                    
            results["predicted_masks"] = None
        
        return results
    
    def _forward_dual_stream_batch(self, input_ids, attention_mask, images_for_gemma, images_for_sam, labels, generate_mask, device):
        """
        デュアルストリーム・バッチ処理のフォワードパス（仕様書第2章対応）
        
        Args:
            input_ids: トークン化されたテキスト (B, seq_len)
            attention_mask: アテンションマスク (B, seq_len)
            images_for_gemma: Gemma用前処理済み画像 (B, 3, 896, 896)
            images_for_sam: SAM用前処理済み画像 (B, 3, 1024, 1024)
            labels: ラベル
            generate_mask: マスク生成フラグ
            device: デバイス
        """
        # ======================================================================
        # パスウェイ 1: SAMの画像エンコーディング (セグメンテーション用)
        # ======================================================================
        sam_features_list = []
        if generate_mask and self.sam_image_encoder is not None and images_for_sam is not None:
            print(f"SAM画像エンコーディング開始: {images_for_sam.shape}")
            
            # SAMの画像エンコーダは凍結されているため、勾配計算は不要
            with torch.no_grad():
                batch_size = images_for_sam.shape[0]
                for i in range(batch_size):
                    sam_img = images_for_sam[i:i+1]  # (1, 3, 1024, 1024)
                    sam_features = self.sam_image_encoder(sam_img)
                    sam_features_list.append(sam_features)
        
        # ======================================================================
        # パスウェイ 2: Gemmaの推論 (意図理解用)
        # ======================================================================
        print(f"🔍 Gemmaモデル入力情報 (デュアルストリーム):")
        print(f"  - input_ids: {input_ids.shape}")
        print(f"  - attention_mask: {attention_mask.shape}")
        print(f"  - images_for_gemma: {images_for_gemma.shape}")
        print(f"  - labels: {labels.shape if labels is not None else 'None'}")
        
        # Gemmaモデルに画像とテキストを入力し、出力を得る
        gemma_outputs = self.gemma_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=images_for_gemma,  # Gemma用前処理済み画像
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        
        # テキスト生成の損失（VQAタスクなどで使用）
        text_loss = gemma_outputs.loss
        
        results = {
            "text_loss": text_loss,
            "logits": gemma_outputs.logits,
            "hidden_states": gemma_outputs.hidden_states,
        }
        
        # ======================================================================
        # 橋渡し: MLPプロジェクタによる特徴量変換とSAMマスクデコーダ
        # ======================================================================
        if generate_mask and sam_features_list:
            # SEGトークンの位置を検出
            seg_positions = (input_ids == self.seg_token_id).nonzero(as_tuple=True)
            
            if len(seg_positions[0]) > 0:
                print(f"バッチ内でSEGトークンが{len(seg_positions[0])}個検出されました")
                
                # SEGトークンからマスクを生成
                masks = self._generate_masks_from_seg_tokens_dual_stream(
                    gemma_outputs.hidden_states[-1], seg_positions, sam_features_list, device
                )
                
                results["predicted_masks"] = masks
            else:
                results["predicted_masks"] = None
        else:
            # SEGトークンが存在する場合、MLPプロジェクタを通して勾配フローを確保
            seg_positions = (input_ids == self.seg_token_id).nonzero(as_tuple=True)
            
            if len(seg_positions[0]) > 0:
                print(f"SEGトークン{len(seg_positions[0])}個でMLPプロジェクタの勾配フローを確保")
                
                # MLPプロジェクタの勾配フローを確保するため
                mlp_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                for batch_idx, token_idx in zip(seg_positions[0], seg_positions[1]):
                    seg_hidden = gemma_outputs.hidden_states[-1][batch_idx, token_idx]
                    seg_embedding = self.mlp_projector(seg_hidden)
                    # 小さなダミー損失を追加（MLPプロジェクタに勾配を流すため）
                    mlp_loss = mlp_loss + seg_embedding.sum() * 1e-6
                
                # テキスト損失にMLP損失を追加
                if results["text_loss"] is not None:
                    results["text_loss"] = results["text_loss"] + mlp_loss
                else:
                    results["text_loss"] = mlp_loss
                    
            results["predicted_masks"] = None
        
        return results

    def _generate_masks_from_seg_tokens_dual_stream(self, hidden_states, seg_positions, sam_features_list, device):
        """
        デュアルストリーム対応のSEGトークンからマスク生成
        各SEGトークンに対応するSAM特徴量を使用してマスクを生成
        """
        pred_masks = []
        
        for batch_idx, token_idx in zip(seg_positions[0], seg_positions[1]):
            # Gemmaの隠れ状態からSEGトークンの埋め込みを取得
            seg_hidden = hidden_states[batch_idx, token_idx]
            
            # MLPプロジェクタを通してSAMが理解できる埋め込みに変換
            seg_embedding = self.mlp_projector(seg_hidden.unsqueeze(0))  # (1, 256)
            
            # 対応するSAM特徴量を取得
            sam_features = sam_features_list[batch_idx.item()]  # (1, 256, 64, 64)
            
            # SAMデコーダでマスク生成
            sparse_embeddings = seg_embedding.unsqueeze(1)  # (1, 1, 256)
            dense_embeddings = torch.zeros(
                (sam_features.shape[0], sam_features.shape[2], sam_features.shape[3]),
                device=device,
                dtype=sam_features.dtype
            )
            
            # SAMプロンプトエンコーダからデンスPEを取得
            dense_pe = self.sam_prompt_encoder.get_dense_pe()
            
            try:
                mask, iou_pred = self.sam_mask_decoder(
                    image_embeddings=sam_features,
                    image_pe=dense_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                pred_masks.append(mask)
            except Exception as e:
                print(f"SAMデコーダでエラー: {e}")
                # ダミーマスクを作成
                dummy_mask = torch.zeros(
                    (1, 1, 256, 256), 
                    device=device, 
                    dtype=sam_features.dtype
                )
                pred_masks.append(dummy_mask)
        
        if len(pred_masks) == 0:
            return None
        elif len(pred_masks) == 1:
            return pred_masks[0]
        else:
            return torch.cat(pred_masks, dim=0)

    def generate_with_segmentation(self, image, text_prompt, max_new_tokens=100):
        """
        テキスト生成とセグメンテーションを同時に実行
        """
        # まずテキスト生成
        gemma_inputs = self.prepare_multimodal_input(image, text_prompt)
        
        device = next(self.gemma_model.parameters()).device
        gemma_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in gemma_inputs.items()}
        
        # テキスト生成
        with torch.inference_mode():
            generated_ids = self.gemma_model.generate(
                **gemma_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.gemma_processor.tokenizer.eos_token_id
            )
        
        # 生成されたテキストをデコード
        input_len = gemma_inputs["input_ids"].shape[-1]
        new_tokens = generated_ids[0][input_len:]
        generated_text = self.gemma_processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # SEGトークンが含まれているかチェック
        if self.seg_token in generated_text:
            print(f"生成されたテキストに{self.seg_token}が含まれています")
            # セグメンテーション実行
            results = self.forward(image, text_prompt, generate_mask=True)
            results["generated_text"] = generated_text
        else:
            results = {"generated_text": generated_text, "predicted_masks": None}
        
        return results 