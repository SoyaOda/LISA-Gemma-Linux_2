# model/llama4_lisa.py
"""
LISA-Llama4アーキテクチャ (Llama-4-Scout 17B + SAM)
Llama4-Scoutモデルの公式API仕様に準拠した実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

from transformers import AutoProcessor, Llama4ForConditionalGeneration, PreTrainedModel, PretrainedConfig, BitsAndBytesConfig
from model.segment_anything import sam_model_registry
from model.segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from model.losses import CompositeLoss  # 統一された損失関数
from utils.constants import DEFAULT_SEG_TOKEN
from torchvision import transforms
from PIL import Image

class MultiModalProjector(nn.Module):
    """マルチモーダルプロジェクタ: Llama4隠れ状態 → SAM埋め込み変換"""
    def __init__(self, llama_hidden_size: int, sam_prompt_embed_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(llama_hidden_size, llama_hidden_size),
            nn.GELU(),
            nn.Linear(llama_hidden_size, sam_prompt_embed_dim),
        )
    
    def forward(self, hidden_states):
        return self.projector(hidden_states)

# LISA-Llama4モデルのカスタム設定クラス
class LisaLlama4Config(PretrainedConfig):
    model_type = "lisa_llama4"

    def __init__(
        self,
        llama_model_id: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        sam_checkpoint_path: Optional[str] = None,
        seg_token: str = "[SEG]",
        llama_hidden_size: int = 5120,    # Llama4 Scoutのhidden_size（テキスト隠れ次元）
        sam_prompt_embed_dim: int = 256,  # SAMのプロンプト埋め込み次元 (ViT-Hは256)
        llama_image_size: int = 448,      # Llama4 Visionモデルの基本タイル画像サイズ
        sam_image_size: int = 1024,       # SAMエンコーダ入力サイズ
        model_max_length: int = 131072,   # Llama4の最大シーケンス長（128K）
        # Llama-4-Scout-17B-16E-Instruct特有の設定
        attn_implementation: str = "eager",          # 安定したアテンション実装（flex_attentionはバグあり）
        device_map: str = "auto",                     # GPU自動分散
        torch_dtype: str = "bfloat16",               # 推奨精度
        **kwargs,
    ):
        self.llama_model_id = llama_model_id
        self.sam_checkpoint_path = sam_checkpoint_path
        self.seg_token = seg_token
        self.llama_hidden_size = llama_hidden_size
        self.sam_prompt_embed_dim = sam_prompt_embed_dim
        self.llama_image_size = llama_image_size
        self.sam_image_size = sam_image_size
        self.model_max_length = model_max_length
        self.attn_implementation = attn_implementation
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        super().__init__(**kwargs)

class LisaLlama4ForCausalLM(PreTrainedModel):
    config_class = LisaLlama4Config

    def __init__(self, config: LisaLlama4Config):
        super().__init__(config)

        # 動的コンパイルを無効化してGPU分散エラーを回避（成功した単独モデルと同じ設定）
        torch.compiler.disable()
        print("動的コンパイル無効化: GPU分散エラー回避のため")

        # 1. Llama-4マルチモーダルモデルの初期化（成功した単独モデル準拠設定）
        print(f"Llama-4モデルをロード中... ({config.llama_model_id})")
        print(f"  - アテンション実装: {config.attn_implementation}")
        print(f"  - デバイスマップ: {config.device_map}")
        print(f"  - Torch精度: {config.torch_dtype}")
        
        # torch_dtypeの変換
        if config.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif config.torch_dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16  # デフォルト
        
        # 2. 量子化設定（成功した単独モデルと同じ設定）
        quantization_config = None
        use_4bit = True  # 4bit量子化を使用（成功した設定）
        if use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("4bit量子化設定を適用")
        
        # 3. Llama4モデル初期化（成功した単独モデルと同じ設定）
        print("Llama4モデル初期化開始...")
        try:
            self.llama_model = Llama4ForConditionalGeneration.from_pretrained(
                config.llama_model_id,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                attn_implementation=config.attn_implementation,  # eager設定を使用
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("✅ Llama-4モデルの初期化完了")
            print(f"  - パラメータ数: {sum(p.numel() for p in self.llama_model.parameters()):,}")
            print(f"  - デバイス分散: {self.llama_model.hf_device_map}")
            
        except Exception as e:
            print(f"❌ Llama4モデル初期化エラー: {e}")
            raise
        
        # 1.1 モデル本体のパラメータを完全凍結（LoRA微調整の下準備）
        print("Llama4モデルのパラメータを全て凍結中...")
        for param in self.llama_model.parameters():
            param.requires_grad = False
        # Note: Embedding層とLMヘッドも凍結（LoRAで効率的に学習するため）
        # この時点ではSEGトークン追加に備えて凍結解除はしない
        
        print("✅ Llama4モデルのパラメータ凍結が完了しました")
        
        # 2. Llama4用プロセッサーの初期化
        print(f"Llama4 Processorをロード中... ({config.llama_model_id})")
        self.llama_processor = AutoProcessor.from_pretrained(config.llama_model_id)
        
        # 4. SAMコンポーネントのロードと凍結（指定がある場合）
        if config.sam_checkpoint_path:
            print(f"SAMモデルをロード中... ({config.sam_checkpoint_path})")
            try:
                self.sam_model = sam_model_registry["vit_h"](checkpoint=config.sam_checkpoint_path)
                self.sam_model.eval()
                
                # SAMモデルをGPUに移動（統一管理）
                self._move_sam_model_to_device()
                
                # SAMパラメータを凍結
                for param in self.sam_model.parameters():
                    param.requires_grad = False
                print("✅ SAMコンポーネントの初期化と凍結が完了しました")
            except Exception as e:
                print(f"❌ SAMのロードに失敗: {e}")
                print("SAMなしで続行します（セグメンテーション機能は無効）")
                self.sam_model = None
        else:
            print("⚠️ SAMチェックポイントが指定されていません。SAM機能はオフになります。")
            self.sam_model = None

        # 5. MLPプロジェクタの初期化
        print("MLPプロジェクタを構築中...")
        self.multi_modal_projector = MultiModalProjector(
            llama_hidden_size=config.llama_hidden_size,
            sam_prompt_embed_dim=config.sam_prompt_embed_dim
        )
        print("✅ MLPプロジェクタ初期化完了")

        # 6. セグメンテーショントークンの語彙追加
        print("セグメンテーショントークンを追加中...")
        self.seg_token = DEFAULT_SEG_TOKEN  # 既定の[SEG]トークン文字列
        tokenizer = self.llama_processor.tokenizer
        if self.seg_token not in tokenizer.get_vocab():
            num_added = tokenizer.add_tokens([self.seg_token], special_tokens=True)
            print(f"✅ 語彙に{num_added}個のトークンを追加しました: {self.seg_token}")
        else:
            print(f"✅ {self.seg_token}は既にトークナイザーに存在します")
        # SEGトークンIDを取得
        self.seg_token_id = tokenizer.convert_tokens_to_ids(self.seg_token)
        print(f"SEGトークンID: {self.seg_token_id}")
        
        # 6.1 埋め込み層のリサイズ（追加トークンに対応）
        current_vocab_size = len(tokenizer)
        embed_size = self.llama_model.get_input_embeddings().weight.shape[0]
        print(f"現在の埋め込みボキャブラリサイズ: {embed_size}, トークナイザー語彙数: {current_vocab_size}")
        if embed_size < current_vocab_size:
            print(f"埋め込み層をリサイズします: {embed_size} -> {current_vocab_size}")
            try:
                self.llama_model.resize_token_embeddings(current_vocab_size)
                print(f"✅ 埋め込み層を{current_vocab_size}次元にリサイズしました")
            except RuntimeError as e:
                if "DTensor" in str(e):
                    print("⚠️ DeepSpeed環境検出: 埋め込み層のリサイズは実行時に再試行します")
                else:
                    print(f"❌ 埋め込み層のリサイズに失敗: {e}")
                    raise e
            new_embed_size = self.llama_model.get_input_embeddings().weight.shape[0]
            print(f"リサイズ後の埋め込みサイズ: {new_embed_size}")
            if new_embed_size < current_vocab_size:
                raise RuntimeError("埋め込み層のリサイズが未完了です")
        else:
            print("✅ 埋め込み層サイズは既に十分対応しています")

        # 7. 統一された損失関数の初期化
        print("CompositeLoss損失関数を初期化中...")
        self.loss_fn = CompositeLoss(
            ce_loss_weight=1.0,    # テキスト生成損失の重み
            dice_loss_weight=0.5,  # DICE損失の重み
            bce_loss_weight=2.0    # BCE損失の重み
        )
        print("✅ CompositeLoss損失関数の初期化が完了しました")

        # 8. 便利のため設定値を保存
        self.llama_image_size = config.llama_image_size
        self.sam_image_size = config.sam_image_size
        self.model_max_length = config.model_max_length

        print("✅ LISA-Llama4モデルの初期化が完了しました（成功した単独モデル準拠設定）")

    @classmethod
    def from_config_file(cls, config_path: str, **kwargs):
        """設定ファイルからLisaLlama4モデルを初期化するクラスメソッド"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["config"] = config_module
        spec.loader.exec_module(config_module)
        # 設定モジュールからLisaLlama4Configを構築
        lisa_config = LisaLlama4Config(
            llama_model_id=getattr(config_module, 'LLAMA_MODEL_ID', "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
            sam_checkpoint_path=getattr(config_module, 'SAM_CHECKPOINT_PATH', None),
            seg_token=getattr(config_module, 'SEG_TOKEN', "[SEG]"),
            llama_hidden_size=getattr(config_module, 'LLAMA_HIDDEN_SIZE', 5120),
            sam_prompt_embed_dim=getattr(config_module, 'SEG_PROJECTION_DIM', 256),
            llama_image_size=getattr(config_module, 'LLAMA_IMAGE_SIZE', 448),
            sam_image_size=getattr(config_module, 'SAM_IMAGE_SIZE', 1024),
            model_max_length=getattr(config_module, 'MODEL_MAX_LENGTH', 131072),
            # Llama-4-Scout特有の設定
            attn_implementation=getattr(config_module, 'ATTN_IMPLEMENTATION', "eager"),
            device_map=getattr(config_module, 'DEVICE_MAP', "auto"),
            torch_dtype=getattr(config_module, 'TORCH_DTYPE', "bfloat16"),
            **kwargs
        )
        return cls(lisa_config)

    def has_sam_capability(self) -> bool:
        """SAMによるマスク生成機能が利用可能か確認"""
        return self.sam_model is not None

    def prepare_multimodal_input(self, image, text_prompt, for_training=False):
        """
        Llama4のマルチモーダル入力を正しく準備
        
        Args:
            image: PIL Image or torch.Tensor
            text_prompt: str
            for_training: bool - Training時とInference時で処理を分ける
        """
        if for_training:
            # Training時: 生のtensorを使用（apply_chat_templateは使わない）
            # これは_forward_dual_stream_batchで既に正しく実装済み
            raise ValueError("Training時はこのメソッドを使わず、直接tensorを渡してください")
        else:
            # Inference時: apply_chat_templateを使用
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
            
            try:
                # Llama4の公式Processorでチャットテンプレート適用
                inputs = self.llama_processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                print(f"🔍 apply_chat_template結果: type={type(inputs)}, keys={list(inputs.keys())}")
                
                # BatchFeatureまたは辞書を辞書として扱う（両方ともdict-likeインターフェース）
                if hasattr(inputs, 'keys') and hasattr(inputs, '__getitem__'):
                    # 辞書形式のデータに変換（必要に応じて）
                    result_dict = {}
                    for key in inputs.keys():
                        result_dict[key] = inputs[key]
                    print(f"✅ BatchFeature/dict変換成功: keys={list(result_dict.keys())}")
                    return result_dict
                else:
                    raise ValueError(f"apply_chat_templateが期待通りのdict-likeオブジェクトを返しませんでした: {type(inputs)}")
            except Exception as e:
                print(f"⚠️ apply_chat_template エラー: {e}")
                print("🔄 フォールバック: 基本的なtokenization")
                
                # フォールバック: 基本的なtokenization
                if isinstance(image, torch.Tensor):
                    pixel_values = image.unsqueeze(0) if image.dim() == 3 else image
                else:
                    # PIL to tensor conversion
                    import torchvision.transforms as transforms
                    transform = transforms.Compose([
                        transforms.Resize((448, 448)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    pixel_values = transform(image).unsqueeze(0)
                
                input_ids = self.llama_processor.tokenizer.encode(
                    text_prompt, 
                    return_tensors="pt",
                    add_special_tokens=True
                )
                
                return {
                    "input_ids": input_ids,
                    "pixel_values": pixel_values,
                    "attention_mask": torch.ones_like(input_ids)
                }

    def get_trainable_parameters_info(self):
        """学習可能なパラメータ数等の情報を取得"""
        total_params = 0
        trainable_params = 0
        for _, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params * 100) if total_params > 0 else 0
        }

    # PEFT（LoRA）対応のため、基底モデルと同じインターフェース関数を用意
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.llama_model.prepare_inputs_for_generation(input_ids, **kwargs)
    def get_input_embeddings(self):
        return self.llama_model.get_input_embeddings()
    def set_input_embeddings(self, value):
        self.llama_model.set_input_embeddings(value)
    def get_output_embeddings(self):
        return self.llama_model.get_output_embeddings()
    def set_output_embeddings(self, value):
        self.llama_model.set_output_embeddings(value)

    def _get_model_device(self) -> torch.device:
        """
        モデルのメインデバイスを取得
        
        Returns:
            torch.device: Llamaモデルのデバイス
        """
        return next(self.llama_model.parameters()).device
    
    def _move_to_device(self, tensor_or_tensors, device=None, dtype=None, description=""):
        """
        テンソルまたはテンソルリストを指定デバイス・データ型に移動
        
        Args:
            tensor_or_tensors: 単一テンソルまたはテンソルのリスト/辞書
            device: 移動先デバイス (None=モデルデバイス使用)
            dtype: 変換先データ型 (None=変換なし)
            description: ログ用説明
            
        Returns:
            移動後のテンソル(群)
        """
        if device is None:
            device = self._get_model_device()
            
        def _move_single_tensor(tensor):
            if tensor is None:
                return tensor
            result = tensor
            if dtype is not None:
                result = result.to(dtype)
            result = result.to(device)
            return result
        
        # 単一テンソル処理
        if isinstance(tensor_or_tensors, torch.Tensor):
            result = _move_single_tensor(tensor_or_tensors)
            if description:
                print(f"🔄 {description}: {result.shape} → {device}")
            return result
        
        # リスト処理
        elif isinstance(tensor_or_tensors, (list, tuple)):
            results = [_move_single_tensor(t) for t in tensor_or_tensors]
            if description:
                print(f"🔄 {description}: {len(results)}個のテンソル → {device}")
            return type(tensor_or_tensors)(results)
        
        # 辞書処理
        elif isinstance(tensor_or_tensors, dict):
            results = {k: _move_single_tensor(v) for k, v in tensor_or_tensors.items()}
            if description:
                print(f"🔄 {description}: 辞書 {list(results.keys())} → {device}")
            return results
        
        else:
            return tensor_or_tensors
    
    def _prepare_sam_inputs_for_device(self, *tensors, description="SAM入力"):
        """
        SAM用テンソルをFloat32 + GPU移動の統一処理
        
        Args:
            *tensors: SAM用テンソル群
            description: ログ用説明
            
        Returns:
            tuple: デバイス・データ型変換後のテンソル群
        """
        # SAMはFloat32が必要
        converted = []
        for tensor in tensors:
            if tensor is not None:
                # BFloat16 → Float32 変換 + GPU移動
                converted_tensor = self._move_to_device(
                    tensor, dtype=torch.float32, description=f"{description}変換"
                )
                converted.append(converted_tensor)
            else:
                converted.append(tensor)
        
        return tuple(converted) if len(converted) > 1 else converted[0]
    
    def _detect_seg_tokens(self, input_ids):
        """
        SEGトークン検出の統一処理
        
        Args:
            input_ids: テキストトークンID
            
        Returns:
            tuple: SEGトークン位置のタプル (batch_indices, token_indices)
        """
        seg_positions = (input_ids == self.seg_token_id).nonzero(as_tuple=True)
        if len(seg_positions[0]) > 0:
            print(f"SEGトークン検出: {len(seg_positions[0])}個")
        return seg_positions
    
    def _prepare_sam_image_features(self, pixel_values, device):
        """
        pixel_valuesからSAM用画像特徴量を生成（統一処理）
        
        Args:
            pixel_values: Llama用画像テンソル [B, C, H, W]
            device: 処理デバイス
            
        Returns:
            list: SAM画像特徴量リスト
        """
        batch_size = pixel_values.shape[0]
        sam_features_list = []
        
        for i in range(batch_size):
            # 各画像をSAM用に変換
            img = pixel_values[i:i+1]  # (1,C,H,W)
            sam_img = F.interpolate(
                img, 
                size=(self.sam_image_size, self.sam_image_size), 
                mode='bilinear', 
                align_corners=False
            )
            sam_img = sam_img * 255.0  # 正規化: 0-1 -> 0-255
            sam_img = self._prepare_sam_inputs_for_device(
                sam_img, description=f"SAMバッチ画像{i}"
            )
            
            # SAM画像エンコーディング
            with torch.no_grad():
                sam_feat = self.sam_model.image_encoder(sam_img)
            sam_features_list.append(sam_feat)
            
        return sam_features_list
    
    def _compute_composite_loss(self, outputs, labels, predicted_masks, ground_truth_mask=None, description="", **kwargs):
        """
        CompositeLoss計算の統一処理
        
        Args:
            outputs: モデル出力
            labels: ラベル
            predicted_masks: 予測マスク
            ground_truth_mask: 正解マスク (optional)
            description: ログ用説明
            **kwargs: 追加パラメータ（ground_truth_mask取得等）
            
        Returns:
            dict: 統一損失結果
        """
        print(f"🔍 CompositeLoss使用による統一損失計算 {description}")
        
        # kwargsからground_truth_maskを取得（dual_stream用）
        if ground_truth_mask is None and 'ground_truth_mask' in kwargs:
            ground_truth_mask = kwargs['ground_truth_mask']
            print(f"🔍 ground_truth_mask取得: {ground_truth_mask.shape if ground_truth_mask is not None else 'None'}")
        
        # model_outputs準備
        model_outputs = {
            "text_loss": outputs.loss,
            "logits": outputs.logits,
            "predicted_masks": predicted_masks,
        }
        
        # batch_data準備
        batch_data = {
            "labels": labels,
            "ground_truth_mask": ground_truth_mask,
        }
        
        # CompositeLoss計算
        losses = self.loss_fn(model_outputs, batch_data)
        
        # 損失情報の詳細表示（dual_stream用）
        if "dual" in description:
            total_loss = losses.get("total_loss")
            text_loss = losses.get("text_loss")
            dice_loss = losses.get("dice_loss")
            bce_loss = losses.get("bce_loss")
            
            print(f"📊 損失結果:")
            print(f"  - 総損失: {total_loss.item() if total_loss is not None else 'None'}")
            print(f"  - テキスト損失: {text_loss.item() if text_loss is not None else 'None'}")
            print(f"  - DICE損失: {dice_loss.item() if dice_loss is not None else 'None'}")
            print(f"  - BCE損失: {bce_loss.item() if bce_loss is not None else 'None'}")
        
        return {
            "text_loss": losses.get("total_loss"),
            "losses": losses,
            "model_outputs": model_outputs
        }
    
    def _handle_model_error(self, error: Exception, context: str, critical: bool = True):
        """
        モデル関連エラーの統一ハンドリング
        
        Args:
            error: 発生した例外
            context: エラー発生文脈
            critical: クリティカルエラーかどうか
            
        Raises:
            RuntimeError: クリティカルエラーの場合
        """
        error_msg = f"❌ {context}でエラー発生: {str(error)}"
        print(error_msg)
        
        if critical:
            raise RuntimeError(f"{context}の処理に失敗しました: {error}") from error
        else:
            print(f"⚠️ {context}: 非クリティカルエラーのため処理を継続")
    

    
    def _safe_model_forward(self, model_func, inputs, context="モデル推論", **kwargs):
        """
        モデル推論の安全実行ラッパー
        
        Args:
            model_func: 実行するモデル関数
            inputs: 入力データ
            context: エラー文脈
            **kwargs: 追加引数
            
        Returns:
            モデル出力
        """
        try:
            return model_func(**inputs, **kwargs)
        except Exception as e:
            self._handle_model_error(e, context, critical=True)
    
    def _safe_sam_decode(self, sam_features, sparse_embeddings, dense_embeddings, dense_pe, context="SAMデコーダ"):
        """
        SAMデコーダの安全実行（エラー時は例外を投げる）
        
        Args:
            sam_features: SAM画像特徴量
            sparse_embeddings: スパース埋め込み
            dense_embeddings: デンス埋め込み  
            dense_pe: Dense Position Encoding
            context: エラー文脈
            
        Returns:
            torch.Tensor: 生成されたマスク
            
        Raises:
            RuntimeError: SAMデコーダ処理失敗時
        """
        try:
            mask, iou_pred = self.sam_model.mask_decoder(
                image_embeddings=sam_features,
                image_pe=dense_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            return mask
            
        except Exception as e:
            self._handle_model_error(e, context, critical=True)
    
    def _move_sam_model_to_device(self):
        """
        SAMモデルをGPUに移動（初期化専用）
        """
        if hasattr(self, 'sam_model') and self.sam_model is not None:
            if torch.cuda.is_available():
                # PyTorch公式推奨: シンプルで確実なモデル移動
                self.sam_model = self.sam_model.cuda()
                print(f"✅ SAMモデルをGPUに移動しました（統一管理）")
                
                # 同期してデバイス移動完了を確実にする
                torch.cuda.synchronize()
                print(f"✅ CUDA同期完了 - SAMモデル準備完了")
            else:
                print("⚠️ CUDA利用不可 - SAMモデルはCPUのまま")
    
    def _validate_and_route_inputs(
        self, 
        input_ids: Optional[torch.LongTensor],
        pixel_values: Optional[torch.FloatTensor],
        images_for_llama: Optional[torch.FloatTensor],
        images_for_sam: Optional[torch.FloatTensor],
        image,
        text_prompt: str
    ) -> str:
        """
        入力を検証し、適切な処理ルートを決定
        
        Args:
            入力パラメータ群
        
        Returns:
            str: 処理ルート ('single', 'dual_stream', 'single_stream')
        
        Raises:
            ValueError: 不正な入力組み合わせの場合
        """
        # 推論モード: 単一画像 + テキスト
        if image is not None and text_prompt is not None:
            return 'single'
        
        # 学習/バッチモード
        if input_ids is not None:
            # デュアルストリーム入力
            if images_for_llama is not None and images_for_sam is not None:
                return 'dual_stream'
            # 後方互換: 単一ストリーム入力
            elif pixel_values is not None:
                return 'single_stream'
        
        # 不正な入力組み合わせ
        raise ValueError(
            "適切な入力が与えられていません: "
            "(image, text_prompt) または (input_ids, images_for_llama, images_for_sam) "
            "または (input_ids, pixel_values) が必要です"
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images_for_llama: Optional[torch.FloatTensor] = None,  # Llama4用画像 (B,3,448,448 * tiles)
        images_for_sam: Optional[torch.FloatTensor] = None,    # SAM用画像 (B,3,1024,1024)
        image=None,              # PIL画像 (単一入力用)
        text_prompt: str = None, # テキストプロンプト (単一入力用)
        generate_mask: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        LISA-Llama4の統合フォワードパス
        
        処理ルート:
        1) 単一画像+テキスト (推論時)
        2) デュアルストリーム (学習時 - 新方式)
        3) 単一ストリーム (学習時 - 後方互換)
        
        Args:
            input_ids: テキストトークンID
            attention_mask: アテンションマスク
            pixel_values: 単一ストリーム用画像テンソル
            labels: 学習用ラベル
            images_for_llama: Llama4用画像テンソル
            images_for_sam: SAM用画像テンソル
            image: PIL画像 (推論用)
            text_prompt: テキストプロンプト (推論用)
            generate_mask: セグメンテーションマスク生成フラグ
            
        Returns:
            Dict[str, Any]: モデル出力 (logits, masks, losses等)
        """
        try:
            # 1. 入力検証とルーティング判定
            route = self._validate_and_route_inputs(
                input_ids, pixel_values, images_for_llama, 
                images_for_sam, image, text_prompt
            )
            
            # 2. デバイス取得
            device = self._get_model_device()
            
            # 3. ルートに応じた処理実行
            if route == 'single':
                return self._forward_single(image, text_prompt, generate_mask, device)
            elif route == 'dual_stream':
                return self._forward_dual_stream_batch(
                    input_ids, attention_mask, images_for_llama, images_for_sam,
                    labels, generate_mask, **kwargs
                )
            elif route == 'single_stream':
                return self._forward_single_stream_batch(
                    input_ids, attention_mask, pixel_values, labels, generate_mask, device
                )
                
        except Exception as e:
            print(f"フォワード中にエラー発生: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _forward_single(self, image, text_prompt, generate_mask, device):
        """単一画像+テキスト入力のフォワード処理（推論用）"""
        # 1. Processorでマルチモーダル入力を準備
        llama_inputs = self.prepare_multimodal_input(image, text_prompt)
        # デバイスに転送
        llama_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in llama_inputs.items()}
        # 2. Llamaモデルでテキスト生成 or 隠れ状態取得
        with torch.no_grad():
            outputs = self.llama_model(
                **llama_inputs,
                output_hidden_states=True,
                return_dict=True
            )
        results = {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states
        }
        # 3. SEGトークンが出現したらマスク生成
        if generate_mask and self.sam_model is not None:
            input_ids = llama_inputs.get("input_ids")
            seg_positions = self._detect_seg_tokens(input_ids)
            if len(seg_positions[0]) > 0:
                print(f"入力中にSEGトークンを{len(seg_positions[0])}個検出")
                # 画像をSAM用に整形 (1024x1024)
                sam_image = image.resize((self.sam_image_size, self.sam_image_size))
                sam_image_tensor = torch.tensor(np.array(sam_image)).permute(2, 0, 1).float().unsqueeze(0)
                sam_image_tensor = self._prepare_sam_inputs_for_device(
                    sam_image_tensor, description="SAM画像入力"
                )
                
                # SAM画像エンコーダから特徴抽出
                with torch.no_grad():
                    sam_features = self.sam_model.image_encoder(sam_image_tensor)
                # SEGトークン隠れ状態からマスク生成
                masks = self._generate_masks_from_seg_tokens_single(outputs.hidden_states[-1], seg_positions, sam_features, device)
                results["predicted_masks"] = masks
            else:
                results["predicted_masks"] = None
        return results

    def _forward_single_stream_batch(self, input_ids, attention_mask, pixel_values, labels, generate_mask, device):
        """従来型: 単一ストリームで画像（pixel_values）を直接Llamaに入力（学習用）"""
        if pixel_values.numel() == 0:
            raise ValueError("pixel_valuesが空です。画像入力が必要です。")
        # Llamaモデル前方計算
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 1
        else:
            self._debug_counter += 1
        if self._debug_counter <= 5:
            print(f"🔍 Llama入力 (single stream #{self._debug_counter}): input_ids{input_ids.shape}, pixel_values{pixel_values.shape}, labels{labels.shape if labels is not None else None}")
        elif self._debug_counter == 6:
            print("🔇 Llama入力のデバッグ出力を省略します")
        outputs = self.llama_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        results = {
            "text_loss": outputs.loss,
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states
        }
        # セグメンテーションマスク生成処理
        if generate_mask and self.sam_model is not None:
            seg_positions = self._detect_seg_tokens(input_ids)
            if len(seg_positions[0]) > 0:
                print(f"バッチ内SEGトークン数: {len(seg_positions[0])}")
                batch_size = pixel_values.shape[0]
                sam_features_list = self._prepare_sam_image_features(pixel_values, device)
                masks = self._generate_masks_from_seg_tokens_batch(outputs.hidden_states[-1], seg_positions, sam_features_list, device)
                results["predicted_masks"] = masks if masks is not None else None
            else:
                results["predicted_masks"] = None
        else:
            # マスク生成しない場合でもマスクを設定（CompositeLoss統一のため）
            results["predicted_masks"] = None
        
        # 統一CompositeLoss処理
        loss_results = self._compute_composite_loss(
            outputs, labels, results["predicted_masks"], 
            ground_truth_mask=None, description="(single stream)"
        )
        results.update(loss_results)
        
        return results

    def _forward_dual_stream_batch(
        self,
        input_ids,
        attention_mask,
        images_for_llama,
        images_for_sam,
        labels=None,
        generate_mask=True,
        seg_token_idx=None,
        **kwargs
    ):
        """
        デュアルストリーム処理：LlamaとSAM両方の処理を実行
        
        Args:
            input_ids: テキストトークンID [batch, seq_len]
            attention_mask: アテンションマスク [batch, seq_len] 
            images_for_llama: Llama用画像 (任意の形状)
            images_for_sam: SAM用画像 [batch, 3, 1024, 1024]
            labels: 学習用ラベル [batch, seq_len]
            generate_mask: セグメンテーションマスク生成フラグ
            seg_token_idx: SEGトークン位置情報
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 変数の初期化（すべてのコードパスで定義される）
        pred_masks = None
        sam_embeddings = None
        
        print(f"🎯 デュアルストリーム処理開始 (バッチ: {batch_size})")
        
        # Llama4でテキスト処理
        print("🖋 Llama4 テキスト処理...")
        
        # 5Dテンソルを4Dテンソルに変換（Llama4の内部実装要求）
        if images_for_llama.dim() == 5:
            batch_size_orig, num_tiles, channels, height, width = images_for_llama.shape
            pixel_values = images_for_llama.view(-1, channels, height, width)
            print(f"🔄 5D->4D変換: {images_for_llama.shape} -> {pixel_values.shape}")
        else:
            pixel_values = images_for_llama
            print(f"🔍 4Dテンソル使用: {pixel_values.shape}")
        
        outputs = self.llama_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )
        
        # 隠れ状態の取得
        hidden_states = outputs.hidden_states[-1]
        print(f"🔍 隠れ状態の形状: {hidden_states.shape}")
        
        # SAMの画像特徴量計算
        image_features_sam = None
        if images_for_sam is not None:
            print("🖼️  SAM 画像エンコーディング...")
            
            # SAMモデルのデバイスを取得し、入力テンソルを適切なデバイスに移動
            sam_device = next(self.sam_model.image_encoder.parameters()).device
            images_for_sam = images_for_sam.float()  # BFloat16 -> Float32 変換（SAM互換性のため）
            images_for_sam = images_for_sam.to(sam_device)
            print(f"🔄 SAM入力をデバイス {sam_device} に移動: {images_for_sam.shape}")
            
            with torch.no_grad():
                image_features_sam = self.sam_model.image_encoder(images_for_sam)
            print(f"🔍 SAM特徴量の形状: {image_features_sam.shape}")
        
        # SEGトークンの検出と処理
        seg_mask = (input_ids == self.seg_token_id)
        
        # model/losses.pyのCompositeLossを使用するための出力構造
        # NaN損失対策：Llama4からのNaN損失を手動計算で修正
        llama_loss = outputs.loss if hasattr(outputs, 'loss') else None
        if llama_loss is not None and torch.isnan(llama_loss):
            print(f"⚠️ Llama4からNaN損失を検出 - CrossEntropyで再計算")
            # NaNの場合は手動でCrossEntropy損失を計算
            if labels is not None and hasattr(outputs, 'logits'):
                import torch.nn as nn
                ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                logits_flat = outputs.logits.view(-1, outputs.logits.size(-1))
                labels_flat = labels.view(-1)
                valid_labels = (labels_flat != -100).sum().item()
                if valid_labels > 0:
                    llama_loss = ce_loss_fn(logits_flat, labels_flat)
                    print(f"✅ 手動計算損失: {llama_loss.item():.6f}")
                else:
                    llama_loss = torch.zeros(1, device=device, requires_grad=True).squeeze()
                    print(f"⚠️ 有効ラベルなし - 損失を0に設定")
            else:
                llama_loss = torch.zeros(1, device=device, requires_grad=True).squeeze()
                print(f"⚠️ ラベルまたはlogitsなし - 損失を0に設定")
        
        model_outputs = {
            "text_loss": llama_loss,
            "logits": outputs.logits if hasattr(outputs, 'logits') else None,
            "hidden_states": hidden_states,
            "predicted_masks": None,  # セグメンテーション後に設定
        }
        
        if seg_mask.any():
            print(f"🎯 SEGトークン検出: {seg_mask.sum().item()}個")
            
            # SEGトークンの隠れ状態を抽出
            seg_indices = seg_mask.nonzero(as_tuple=False)
            seg_embeddings_list = []
            
            for batch_idx, seq_idx in seg_indices:
                seg_embedding = hidden_states[batch_idx, seq_idx]
                seg_embeddings_list.append(seg_embedding)
            
            if seg_embeddings_list:
                seg_embeddings = torch.stack(seg_embeddings_list)
                
                # MLPプロジェクタのデバイスを取得し、入力を適切なデバイスに移動
                projector_device = next(self.multi_modal_projector.parameters()).device
                seg_embeddings = seg_embeddings.to(projector_device)
                print(f"🔄 SEG埋め込みをデバイス {projector_device} に移動: {seg_embeddings.shape}")
                
                # MLPプロジェクタでSAM埋め込み次元にマッピング
                projected_embeddings = self.multi_modal_projector(seg_embeddings)
                sam_embeddings = projected_embeddings
                
                # セグメンテーションマスク生成
                if generate_mask and image_features_sam is not None:
                    print(f"🔍 セグメンテーションマスク生成開始")
                    
                    # SEGトークンに対応するマスクを生成
                    predicted_masks_list = []
                    
                    for i, embedding in enumerate(projected_embeddings):
                        try:
                            # SAMでマスク生成
                            sparse_embeddings = embedding.unsqueeze(0).unsqueeze(0)
                            dense_embeddings = self.sam_model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
                            
                            # 対応するSAM特徴量を取得
                            img_idx = seg_indices[i][0].item()  # バッチインデックス
                            sam_features = image_features_sam[min(img_idx, image_features_sam.shape[0]-1):min(img_idx, image_features_sam.shape[0]-1)+1]
                            
                            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                                image_embeddings=sam_features,
                                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=False
                            )
                            
                            predicted_masks_list.append(low_res_masks)
                            
                        except Exception as e:
                            self._handle_model_error(e, f"マスク生成 (#{i})", critical=True)
                    
                    if predicted_masks_list:
                        pred_masks = torch.cat(predicted_masks_list, dim=0)
                        model_outputs["predicted_masks"] = pred_masks
                        print(f"✅ マスク生成完了: {pred_masks.shape}")
                    else:
                        print("⚠️ マスク生成に失敗")
                        model_outputs["predicted_masks"] = None
                else:
                    print(f"ℹ️ マスク生成スキップ (generate_mask={generate_mask}, sam_features={image_features_sam is not None})")
                    model_outputs["predicted_masks"] = None
            else:
                print("⚠️ SEGトークンの埋め込み抽出に失敗")
        else:
            print("ℹ️ SEGトークンなし - セグメンテーション処理をスキップ")
        
        # 統一CompositeLoss処理（dual_stream用）
        # outputsオブジェクトを構築（統一メソッド用）
        class OutputsWrapper:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits
        
        outputs_wrapper = OutputsWrapper(
            loss=model_outputs.get("text_loss"),
            logits=model_outputs.get("logits")
        )
        
        loss_results = self._compute_composite_loss(
            outputs_wrapper, labels, model_outputs["predicted_masks"],
            description="(dual stream)", **kwargs
        )
        
        return {
            "text_loss": loss_results["text_loss"],
            "logits": model_outputs["logits"],
            "hidden_states": hidden_states,
            "pred_masks": model_outputs["predicted_masks"],
            "sam_embeddings": sam_embeddings,
            "losses": loss_results["losses"],
            "model_outputs": loss_results["model_outputs"]
        }

    def _generate_masks_from_seg_tokens_single(self, hidden_states, seg_positions, sam_features, device):
        """
        単一入力用: 最後の隠れ状態とSEG位置からマスクを生成
        hidden_states: (1, seq_len, hidden_size)
        """
        masks = []
        
        # multi_modal_projectorのデバイスを取得
        projector_device = next(self.multi_modal_projector.parameters()).device
        
        for batch_idx, token_idx in zip(seg_positions[0], seg_positions[1]):
            # SEGトークン隠れベクトル抽出とデバイス統一
            seg_hidden = hidden_states[batch_idx, token_idx]  # (hidden_size,)
            seg_hidden = seg_hidden.to(projector_device)  # projectorと同じデバイスに移動
            
            # PyTorch公式推奨：データ型も統一（BFloat16 -> Float32）
            projector_dtype = next(self.multi_modal_projector.parameters()).dtype
            seg_hidden = seg_hidden.to(projector_dtype)
            
            seg_emb = self.multi_modal_projector(seg_hidden.unsqueeze(0))  # (1,256)
            sparse_embeddings = seg_emb.unsqueeze(1)  # (1,1,256)
            dense_embeddings = torch.zeros(
                (sam_features.shape[0], sam_features.shape[2], sam_features.shape[3]),
                device=device, dtype=sam_features.dtype
            )
            dense_pe = self.sam_model.prompt_encoder.get_dense_pe()
            try:
                # SAMデコーダー入力を統一デバイス管理で処理
                sam_features, dense_pe, sparse_embeddings, dense_embeddings = self._prepare_sam_inputs_for_device(
                    sam_features, dense_pe, sparse_embeddings, dense_embeddings,
                    description="SAMデコーダー入力（単一）"
                )
                
                mask, iou_pred = self.sam_model.mask_decoder(
                    image_embeddings=sam_features,
                    image_pe=dense_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )
                masks.append(mask)
            except Exception as e:
                self._handle_model_error(e, "SAMデコーダ（単一）", critical=True)
        if len(masks) == 0:
            return None
        return masks[0] if len(masks) == 1 else torch.cat(masks, dim=0)

    def _generate_masks_from_seg_tokens_batch(self, hidden_states, seg_positions, sam_features_list, device):
        """
        バッチ入力用: SEGトークン隠れ状態からマスク生成（単一ストリーム版）
        """
        if not seg_positions[0].numel():
            return None
        masks = []
        seg_count = 0
        batch_size = hidden_states.shape[0]
        
        # multi_modal_projectorのデバイスを取得
        projector_device = next(self.multi_modal_projector.parameters()).device
        
        for i in range(batch_size):
            # 画像iに対応するSEGトークンを探索
            mask_for_image = None
            for j in range(len(seg_positions[0])):
                if seg_positions[0][j] == i:
                    # i番目の画像にSEGトークンがある場合
                    token_idx = seg_positions[1][j]
                    seg_hidden = hidden_states[i, token_idx]
                    seg_hidden = seg_hidden.to(projector_device)  # projectorと同じデバイスに移動
                    
                    # PyTorch公式推奨：データ型も統一（バッチ版）
                    projector_dtype = next(self.multi_modal_projector.parameters()).dtype
                    seg_hidden = seg_hidden.to(projector_dtype)
                    
                    seg_emb = self.multi_modal_projector(seg_hidden.unsqueeze(0))  # (1,256)
                    sparse_embeddings = seg_emb.unsqueeze(1)  # (1,1,256)
                    dense_embeddings = torch.zeros(
                        (sam_features_list[i].shape[0], sam_features_list[i].shape[2], sam_features_list[i].shape[3]),
                        device=device, dtype=sam_features_list[i].dtype
                    )
                    dense_pe = self.sam_model.prompt_encoder.get_dense_pe()
                    try:
                        # SAMデコーダー入力を統一デバイス管理で処理（バッチ版）
                        sam_features_gpu, dense_pe, sparse_embeddings, dense_embeddings = self._prepare_sam_inputs_for_device(
                            sam_features_list[i], dense_pe, sparse_embeddings, dense_embeddings,
                            description=f"SAMデコーダー入力（バッチ{i}）"
                        )
                        
                        mask, _ = self.sam_model.mask_decoder(
                            image_embeddings=sam_features_gpu,
                            image_pe=dense_pe,
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False
                        )
                        mask_for_image = mask
                    except Exception as e:
                        self._handle_model_error(e, f"SAMデコーダ（バッチ{i}）", critical=True)
                    break
            if mask_for_image is None:
                # その画像にSEG要求が無い場合はエラー
                self._handle_model_error(
                    ValueError(f"画像{i}にSEGトークンが見つかりません"), 
                    f"バッチマスク生成（画像{i}）", 
                    critical=True
                )
            masks.append(mask_for_image)
        return torch.cat(masks, dim=0) if masks else None

    def generate_with_segmentation(self, image, text_prompt, max_new_tokens=100):
        """
        Llama4の正しいInference API使用
        1. apply_chat_templateでInference用入力を準備
        2. model.generate()でテキスト生成
        3. SEGトークン検出時はforward()でマスク生成
        """
        print("\n🎯 Inference開始: generate_with_segmentation")
        
        try:
            # Step 1: Inference用入力準備（apply_chat_templateを使用）
            print("📝 Step 1: Inference用入力準備")
            inputs = self.prepare_multimodal_input(image, text_prompt, for_training=False)
            device = next(self.llama_model.parameters()).device
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in inputs.items()}
            
            print(f"🔍 入力確認: input_ids {inputs['input_ids'].shape}")
            if 'pixel_values' in inputs:
                print(f"🔍 入力確認: pixel_values {inputs['pixel_values'].shape}")
            
            # Step 2: テキスト生成（Inference API）
            print("📝 Step 2: テキスト生成実行")
            with torch.inference_mode():
                generated_ids = self.llama_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # 決定的生成
                    pad_token_id=self.llama_processor.tokenizer.pad_token_id or self.llama_processor.tokenizer.eos_token_id,
                    eos_token_id=self.llama_processor.tokenizer.eos_token_id
                )
            
            # Step 3: 生成されたテキストをデコード
            print("📝 Step 3: 生成テキストデコード")
            input_len = inputs["input_ids"].shape[-1]
            new_tokens = generated_ids[0][input_len:]
            generated_text = self.llama_processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            print(f"✅ 生成テキスト: {generated_text[:100]}...")
            
            # Step 4: SEGトークンチェックとマスク生成
            print("📝 Step 4: SEGトークンチェック")
            results = {"generated_text": generated_text}
            
            if self.seg_token in generated_text:
                print(f"🎯 {self.seg_token}トークン検出！マスク生成実行")
                # Training APIを使用してマスク生成
                mask_results = self.forward(
                    image=image, 
                    text_prompt=text_prompt, 
                    generate_mask=True
                )
                results["predicted_masks"] = mask_results.get("pred_masks")
                print(f"✅ マスク生成完了: {type(results['predicted_masks'])}")
            else:
                print("ℹ️ SEGトークンなし - マスクなし")
                results["predicted_masks"] = None
                
            return results
            
        except Exception as e:
            print(f"❌ generate_with_segmentation エラー: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
                "generated_text": "",
                "predicted_masks": None
            }

    def apply_lora_configuration(self, lora_config):
        """
        LoRA設定を適用し、全体の<1%パラメータ微調整を実現
        """
        from peft import get_peft_model
        print("\n=== LoRA設定適用中 ===")
        print(f"LoRA設定: r={lora_config.r}, alpha={lora_config.lora_alpha}, modules={lora_config.target_modules}")
        self.llama_model = get_peft_model(self.llama_model, lora_config)
        print("✅ LoRAラッパーをモデルに適用しました")
        # 埋め込み層とLMヘッドの重みを明示的に固定
        print("🔒 Embedding層およびLMヘッドを凍結します")
        input_emb = None
        if hasattr(self.llama_model, 'base_model'):
            base = self.llama_model.base_model
            # Llama4の場合、model.embed_tokens が埋め込み、model.lm_head が出力ヘッド
            if hasattr(base, 'model'):
                if hasattr(base.model, 'embed_tokens'):
                    input_emb = base.model.embed_tokens
                    input_emb.weight.requires_grad = False
                    print(f"✅ 入力embed凍結: {input_emb.weight.shape}")
                if hasattr(base.model, 'lm_head'):
                    output_emb = base.model.lm_head
                    output_emb.weight.requires_grad = False
                    print(f"✅ 出力embed凍結: {output_emb.weight.shape}")
                elif hasattr(base.model, 'embed_tokens'):
                    print("✅ 入出力埋め込みが共有されています（入力凍結で対応）")
        if input_emb:
            print(f"🔒 凍結確認 (入力Embed): requires_grad={input_emb.weight.requires_grad}")
        # パラメータ統計の表示
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ratio = (trainable/total*100) if total>0 else 0
        print(f"総パラメータ: {total:,} / 学習可能: {trainable:,} ({ratio:.4f}%)")
        if ratio < 1.0:
            print(f"🎯 学習可能パラメータ率 {ratio:.4f}% (<1%) - 仕様準拠")
        else:
            print(f"⚠️ 学習可能パラメータ率 {ratio:.4f}% (>=1%) - 要確認")
        # 学習可能パラメータの内訳
        categories = {'LoRA':0, 'MLP Projector':0, 'SAM Mask Decoder':0, 'Others':0}
        for name, param in self.named_parameters():
            if param.requires_grad:
                count = param.numel()
                lname = name.lower()
                if 'lora' in lname:
                    categories['LoRA'] += count
                elif 'multi_modal_projector' in name or 'multi_modal_projector' in lname:
                    categories['MLP Projector'] += count
                elif 'sam_mask_decoder' in name:
                    categories['SAM Mask Decoder'] += count
                else:
                    categories['Others'] += count
        print("学習可能パラメータカテゴリ:")
        for cat, count in categories.items():
            if count > 0:
                print(f"  - {cat}: {count:,} ({count/total*100:.4f}%)")
        return self