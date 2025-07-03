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

from transformers import AutoProcessor, Llama4ForConditionalGeneration, PreTrainedModel, PretrainedConfig
from model.segment_anything import sam_model_registry
from model.segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from utils.constants import DEFAULT_SEG_TOKEN

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
        attn_implementation: str = "flex_attention",  # MoE対応の最適化アテンション
        device_map: str = "auto",                     # meta tensor対策
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

        # 1. Llama-4マルチモーダルモデルの初期化（Webリサーチ準拠設定）
        print(f"Llama-4モデルをロード中... ({config.llama_model_id})")
        print(f"  - アテンション実装: {config.attn_implementation}")
        print(f"  - デバイスマップ: {config.device_map}")
        print(f"  - Torch精度: {config.torch_dtype}")
        
        # flex_attention vs eagerアテンションの説明
        if config.attn_implementation == "eager":
            print("  ⚠️  注意: flex_attentionにバグがあるため、eagerアテンションを使用")
            print("     パフォーマンスは劣りますが、安定性が向上します")
        
        # torch_dtypeの変換
        if config.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif config.torch_dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16  # デフォルト
        
        try:
            self.llama_model = Llama4ForConditionalGeneration.from_pretrained(
                config.llama_model_id,
                attn_implementation=config.attn_implementation,
                device_map=config.device_map,
                torch_dtype=torch_dtype,
            )
            print("✅ Llama-4モデルの初期化完了")
        except Exception as e:
            print(f"❌ モデルロードエラー: {e}")
            if "flex_attention" in str(e):
                print("💡 提案: config_linux.pyのATTN_IMPLEMENTATIONを'eager'に変更してください")
            raise e
        
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
        
        # 3. SAMコンポーネントのロードと凍結（指定がある場合）
        if config.sam_checkpoint_path:
            print(f"SAMモデルをロード中... ({config.sam_checkpoint_path})")
            try:
                sam = sam_model_registry["vit_h"](checkpoint=config.sam_checkpoint_path)
                
                # SAMの各コンポーネントを取り出し
                self.sam_image_encoder = sam.image_encoder
                self.sam_prompt_encoder = sam.prompt_encoder
                self.sam_mask_decoder = sam.mask_decoder

                # SAMの画像エンコーダとプロンプトエンコーダは凍結
                for param in self.sam_image_encoder.parameters():
                    param.requires_grad = False
                for param in self.sam_prompt_encoder.parameters():
                    param.requires_grad = False
                # SAMのマスクデコーダは学習可能にする
                for param in self.sam_mask_decoder.parameters():
                    param.requires_grad = True
                
                # デバイスをLlamaモデルと合わせる
                device = next(self.llama_model.parameters()).device
                self.sam_image_encoder = self.sam_image_encoder.to(device)
                self.sam_prompt_encoder = self.sam_prompt_encoder.to(device)
                self.sam_mask_decoder = self.sam_mask_decoder.to(device)
                
                print("✅ SAMコンポーネントの初期化と凍結が完了しました")
            except Exception as e:
                print(f"❌ SAMのロードに失敗: {e}")
                print("SAMなしで続行します（セグメンテーション機能は無効）")
                self.sam_image_encoder = None
                self.sam_prompt_encoder = None
                self.sam_mask_decoder = None
        else:
            print("⚠️ SAMチェックポイントが指定されていません。SAM機能はオフになります。")
            self.sam_image_encoder = None
            self.sam_prompt_encoder = None
            self.sam_mask_decoder = None

        # 4. MLPプロジェクタの定義 (Llama4 hidden -> SAM埋め込みへの橋渡し)
        print("MLPプロジェクタを構築中...")
        device = next(self.llama_model.parameters()).device
        self.mlp_projector = nn.Sequential(
            nn.Linear(config.llama_hidden_size, config.llama_hidden_size),
            nn.GELU(),
            nn.Linear(config.llama_hidden_size, config.sam_prompt_embed_dim),
        ).to(device).to(torch.bfloat16)

        # 5. セグメンテーショントークンの語彙追加
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
        
        # 5.1 埋め込み層のリサイズ（追加トークンに対応）
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

        # 6. 便利のため設定値を保存
        self.llama_image_size = config.llama_image_size
        self.sam_image_size = config.sam_image_size
        self.model_max_length = config.model_max_length

        print("✅ LISA-Llama4モデルの初期化が完了しました")

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
            attn_implementation=getattr(config_module, 'ATTN_IMPLEMENTATION', "flex_attention"),
            device_map=getattr(config_module, 'DEVICE_MAP', "auto"),
            torch_dtype=getattr(config_module, 'TORCH_DTYPE', "bfloat16"),
            **kwargs
        )
        return cls(lisa_config)

    def has_sam_capability(self) -> bool:
        """SAMによるマスク生成機能が利用可能か確認"""
        return all([
            self.sam_image_encoder is not None,
            self.sam_prompt_encoder is not None,
            self.sam_mask_decoder is not None
        ])

    def prepare_multimodal_input(self, image, text_prompt):
        """
        Llama4公式のチャットテンプレートに従って入力データを準備
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
        # 公式Processorのチャットテンプレートを適用して入力取得
        inputs = self.llama_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        return inputs

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
        LISA-Llama4のフォワードパス
        1) 単一の画像+テキスト (推論時)
        2) バッチ入力 (トレーニング時) - デュアルストリーム画像対応
        """
        try:
            device = next(self.llama_model.parameters()).device

            # 推論モード: 単一画像 + テキストの場合
            if image is not None and text_prompt is not None:
                return self._forward_single(image, text_prompt, generate_mask, device)
            
            # 学習/バッチモード:
            if input_ids is not None and (images_for_llama is not None or pixel_values is not None):
                # 新しいデュアルストリーム入力
                if images_for_llama is not None and images_for_sam is not None:
                    return self._forward_dual_stream_batch(
                        input_ids, attention_mask, images_for_llama, images_for_sam,
                        labels, generate_mask, device
                    )
                # 後方互換: pixel_valuesのみ使用の場合 (images_for_llamaがNone)
                elif pixel_values is not None:
                    return self._forward_single_stream_batch(
                        input_ids, attention_mask, pixel_values, labels, generate_mask, device
                    )
            raise ValueError("適切な入力が与えられていません: (image, text_prompt) または (input_ids, images_for_llama, images_for_sam) が必要です")
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
        if generate_mask and self.sam_image_encoder is not None:
            input_ids = llama_inputs.get("input_ids")
            seg_positions = (input_ids == self.seg_token_id).nonzero(as_tuple=True)
            if len(seg_positions[0]) > 0:
                print(f"入力中にSEGトークンを{len(seg_positions[0])}個検出")
                # 画像をSAM用に整形 (1024x1024)
                sam_image = image.resize((self.sam_image_size, self.sam_image_size))
                sam_image_tensor = torch.tensor(np.array(sam_image)).permute(2, 0, 1).float().unsqueeze(0).to(device)
                # SAM画像エンコーダから特徴抽出
                with torch.no_grad():
                    sam_features = self.sam_image_encoder(sam_image_tensor)
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
        if generate_mask and self.sam_image_encoder is not None:
            seg_positions = (input_ids == self.seg_token_id).nonzero(as_tuple=True)
            if len(seg_positions[0]) > 0:
                print(f"バッチ内SEGトークン数: {len(seg_positions[0])}")
                batch_size = pixel_values.shape[0]
                sam_features_list = []
                for i in range(batch_size):
                    # pixel_valuesの各画像テンソルを(3,H,W)から(1,3,1024,1024)へ変換
                    img = pixel_values[i:i+1]  # (1,3,H,W)
                    # Llamaの画像入力がタイル処理済みの場合でも、簡易的に全体をresize (注意:情報損失の可能性)
                    sam_img = F.interpolate(img, size=(self.sam_image_size, self.sam_image_size), mode='bilinear', align_corners=False)
                    sam_img = sam_img * 255.0  # 正規化: 0-1 -> 0-255
                    with torch.no_grad():
                        sam_feat = self.sam_image_encoder(sam_img)
                    sam_features_list.append(sam_feat)
                masks = self._generate_masks_from_seg_tokens_batch(outputs.hidden_states[-1], seg_positions, sam_features_list, device)
                results["predicted_masks"] = masks if masks is not None else None
            else:
                results["predicted_masks"] = None
        else:
            # マスク生成しない場合でもSEGトークンの勾配をMLPに流す処理
            seg_positions = (input_ids == self.seg_token_id).nonzero(as_tuple=True)
            if len(seg_positions[0]) > 0:
                print(f"SEGトークン{len(seg_positions[0])}個に対してMLPプロジェクタの勾配を確保")
                mlp_loss = torch.tensor(0.0, device=device, requires_grad=True)
                for batch_idx, token_idx in zip(seg_positions[0], seg_positions[1]):
                    seg_hidden = outputs.hidden_states[-1][batch_idx, token_idx]
                    seg_embed = self.mlp_projector(seg_hidden)
                    mlp_loss = mlp_loss + seg_embed.sum() * 1e-6
                results["text_loss"] = outputs.loss + mlp_loss if outputs.loss is not None else mlp_loss
            results["predicted_masks"] = None
        return results

    def _forward_dual_stream_batch(self, input_ids, attention_mask, images_for_llama, images_for_sam, labels, generate_mask, device):
        """
        新方式: デュアルストリーム画像入力のフォワード処理（Llama用画像とSAM用画像を別々に供給）
        """
        # Path 1: SAM画像エンコーディング
        sam_features_list = []
        if generate_mask and self.sam_image_encoder is not None and images_for_sam is not None:
            if not hasattr(self, '_sam_counter'):
                self._sam_counter = 1
            else:
                self._sam_counter += 1
            if self._sam_counter <= 2:
                print(f"SAMエンコード (batch #{self._sam_counter}): images_for_sam{images_for_sam.shape}")
            elif self._sam_counter == 3:
                print("🔇 SAMエンコードのログを以降省略")
            with torch.no_grad():
                batch = images_for_sam.shape[0]
                for i in range(batch):
                    sam_img = images_for_sam[i:i+1]  # (1,3,1024,1024)
                    sam_feat = self.sam_image_encoder(sam_img)
                    sam_features_list.append(sam_feat)
        # Path 2: Llamaモデルでの言語推論
        if not hasattr(self, '_llama_counter'):
            self._llama_counter = 1
        else:
            self._llama_counter += 1
        if self._llama_counter <= 3:
            print(f"🔍 Llama入力 (dual stream #{self._llama_counter}): input_ids{input_ids.shape}, images_for_llama{images_for_llama.shape}, labels{labels.shape if labels is not None else None}")
        elif self._llama_counter == 4:
            print("🔇 デュアルストリームのデバッグ出力を省略します")
        outputs = self.llama_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=images_for_llama,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        text_loss = outputs.loss
        results = {
            "text_loss": text_loss,
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states
        }
        # 橋渡し: SEGトークン -> SAMマスク生成
        if generate_mask and sam_features_list:
            seg_positions = (input_ids == self.seg_token_id).nonzero(as_tuple=True)
            if len(seg_positions[0]) > 0:
                if not hasattr(self, '_seg_counter'):
                    self._seg_counter = 1
                else:
                    self._seg_counter += 1
                if self._seg_counter <= 2:
                    print(f"SEGトークン検出: {len(seg_positions[0])}個 (batch)")
                elif self._seg_counter == 3:
                    print("🔇 SEGトークン検出ログを省略します")
                masks = self._generate_masks_from_seg_tokens_dual(outputs.hidden_states[-1], seg_positions, sam_features_list, device)
                results["predicted_masks"] = masks
            else:
                results["predicted_masks"] = None
        else:
            # マスク非生成でもMLPプロジェクタに微小な勾配を流す
            seg_positions = (input_ids == self.seg_token_id).nonzero(as_tuple=True)
            if len(seg_positions[0]) > 0:
                if not hasattr(self, '_mlp_grad_counter'):
                    self._mlp_grad_counter = 1
                else:
                    self._mlp_grad_counter += 1
                if self._mlp_grad_counter <= 2:
                    print(f"MLPプロジェクタ用ダミー損失を追加 (SEGトークン数: {len(seg_positions[0])})")
                elif self._mlp_grad_counter == 3:
                    print("🔇 MLPプロジェクタ勾配ログを省略します")
                mlp_loss = torch.tensor(0.0, device=device, requires_grad=True)
                for batch_idx, token_idx in zip(seg_positions[0], seg_positions[1]):
                    seg_hidden = outputs.hidden_states[-1][batch_idx, token_idx]
                    seg_embed = self.mlp_projector(seg_hidden)
                    mlp_loss = mlp_loss + seg_embed.sum() * 1e-6
                results["text_loss"] = (results["text_loss"] + mlp_loss) if results["text_loss"] is not None else mlp_loss
            results["predicted_masks"] = None
        return results

    def _generate_masks_from_seg_tokens_single(self, hidden_states, seg_positions, sam_features, device):
        """
        単一入力用: 最後の隠れ状態とSEG位置からマスクを生成
        hidden_states: (1, seq_len, hidden_size)
        """
        masks = []
        for batch_idx, token_idx in zip(seg_positions[0], seg_positions[1]):
            # SEGトークン隠れベクトル抽出
            seg_hidden = hidden_states[batch_idx, token_idx]  # (hidden_size,)
            seg_emb = self.mlp_projector(seg_hidden.unsqueeze(0))  # (1,256)
            sparse_embeddings = seg_emb.unsqueeze(1)  # (1,1,256)
            dense_embeddings = torch.zeros(
                (sam_features.shape[0], sam_features.shape[2], sam_features.shape[3]),
                device=device, dtype=sam_features.dtype
            )
            dense_pe = self.sam_prompt_encoder.get_dense_pe()
            try:
                mask, iou_pred = self.sam_mask_decoder(
                    image_embeddings=sam_features,
                    image_pe=dense_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )
                masks.append(mask)
            except Exception as e:
                print(f"SAMデコーダエラー: {e}")
                dummy_mask = torch.zeros((1, 1, 256, 256), device=device, dtype=sam_features.dtype)
                masks.append(dummy_mask)
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
        for i in range(batch_size):
            # 画像iに対応するSEGトークンを探索
            mask_for_image = None
            for j in range(len(seg_positions[0])):
                if seg_positions[0][j] == i:
                    # i番目の画像にSEGトークンがある場合
                    token_idx = seg_positions[1][j]
                    seg_hidden = hidden_states[i, token_idx]
                    seg_emb = self.mlp_projector(seg_hidden.unsqueeze(0))  # (1,256)
                    sparse_embeddings = seg_emb.unsqueeze(1)  # (1,1,256)
                    dense_embeddings = torch.zeros(
                        (sam_features_list[i].shape[0], sam_features_list[i].shape[2], sam_features_list[i].shape[3]),
                        device=device, dtype=sam_features_list[i].dtype
                    )
                    dense_pe = self.sam_prompt_encoder.get_dense_pe()
                    try:
                        mask, _ = self.sam_mask_decoder(
                            image_embeddings=sam_features_list[i],
                            image_pe=dense_pe,
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False
                        )
                        mask_for_image = mask
                    except Exception as e:
                        print(f"SAMデコーダエラー(バッチ): {e}")
                        mask_for_image = torch.zeros((1, 1, 256, 256), device=device, dtype=sam_features_list[i].dtype)
                    break
            if mask_for_image is None:
                # その画像にSEG要求が無い場合はダミーマスク
                mask_for_image = torch.zeros((1, 1, 256, 256), device=device, dtype=sam_features_list[i].dtype)
            masks.append(mask_for_image)
        return torch.cat(masks, dim=0) if masks else None

    def _generate_masks_from_seg_tokens_dual(self, hidden_states, seg_positions, sam_features_list, device):
        """
        デュアルストリーム用: SEGトークン隠れ状態からマスク生成
        """
        if not seg_positions[0].numel():
            return None
        masks = []
        for batch_idx, token_idx in zip(seg_positions[0], seg_positions[1]):
            seg_hidden = hidden_states[batch_idx, token_idx]
            seg_emb = self.mlp_projector(seg_hidden.unsqueeze(0))  # (1,256)
            sparse_embeddings = seg_emb.unsqueeze(1)  # (1,1,256)
            dense_embeddings = torch.zeros(
                (sam_features_list[batch_idx].shape[0], sam_features_list[batch_idx].shape[2], sam_features_list[batch_idx].shape[3]),
                device=device, dtype=sam_features_list[batch_idx].dtype
            )
            dense_pe = self.sam_prompt_encoder.get_dense_pe()
            try:
                mask, _ = self.sam_mask_decoder(
                    image_embeddings=sam_features_list[batch_idx],
                    image_pe=dense_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )
                masks.append(mask)
            except Exception as e:
                print(f"SAMデコーダエラー(dual): {e}")
                dummy_mask = torch.zeros((1, 1, 256, 256), device=device, dtype=sam_features_list[batch_idx].dtype)
                masks.append(dummy_mask)
        if len(masks) == 0:
            return None
        return masks[0] if len(masks) == 1 else torch.cat(masks, dim=0)

    def generate_with_segmentation(self, image, text_prompt, max_new_tokens=100):
        """
        画像を含むプロンプトに対するテキスト生成と必要ならセグメンテーションを実行
        """
        # まずテキスト生成のみ実行
        inputs = self.prepare_multimodal_input(image, text_prompt)
        device = next(self.llama_model.parameters()).device
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in inputs.items()}
        with torch.inference_mode():
            generated_ids = self.llama_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.llama_processor.tokenizer.eos_token_id
            )
        # 生成テキストをデコード
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = generated_ids[0][input_len:]
        generated_text = self.llama_processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        # SEGトークンを検知してマスク生成
        results = {}
        if self.seg_token in generated_text:
            print(f"生成テキスト中に{self.seg_token}を検出。マスク生成を実行します。")
            infer_results = self.forward(image=image, text_prompt=text_prompt, generate_mask=True)
            results["generated_text"] = generated_text
            results["predicted_masks"] = infer_results.get("predicted_masks")
        else:
            results["generated_text"] = generated_text
            results["predicted_masks"] = None
        return results

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
                elif 'mlp_projector' in name or 'mlp_projector' in lname:
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