# model/gemma_lisa.py
"""
LISA-Gemma3アーキテクチャ
Gemma-3の公式API仕様に準拠した実装
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

from transformers import AutoProcessor, Gemma3ForConditionalGeneration, PreTrainedModel, PretrainedConfig
from segment_anything import sam_model_registry
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

# LISA-Gemmaモデルのカスタム設定クラス
class LisaGemmaConfig(PretrainedConfig):
    model_type = "lisa_gemma"

    def __init__(
        self,
        gemma_model_id="google/gemma-3-4b-it",
        sam_checkpoint_path=None,
        seg_token="<SEG>",
        gemma_hidden_size=2560,  # Gemma 3 4B の hidden_size
        sam_prompt_embed_dim=256,
        **kwargs,
    ):
        self.gemma_model_id = gemma_model_id
        self.sam_checkpoint_path = sam_checkpoint_path
        self.seg_token = seg_token
        self.gemma_hidden_size = gemma_hidden_size
        self.sam_prompt_embed_dim = sam_prompt_embed_dim
        super().__init__(**kwargs)

class LisaGemmaForCausalLM(PreTrainedModel):
    config_class = LisaGemmaConfig

    def __init__(self, config: LisaGemmaConfig):
        super().__init__(config)

        # 1. Gemma-3 multimodal model の初期化
        print("Gemma-3マルチモーダルモデルをロード中...")
        self.gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            config.gemma_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 2. Gemma-3用プロセッサーの初期化
        print("Gemma-3プロセッサーをロード中...")
        self.gemma_processor = AutoProcessor.from_pretrained(config.gemma_model_id)
        
        # 3. SAMコンポーネントのロードと凍結（SAMチェックポイントが存在する場合のみ）
        if config.sam_checkpoint_path and config.sam_checkpoint_path != "":
            print("SAMモデルをロード中...")
            sam = sam_model_registry["vit_h"](checkpoint=config.sam_checkpoint_path)
            
            # SAMの画像エンコーダを抽出し、凍結する
            self.sam_image_encoder = sam.image_encoder
            for param in self.sam_image_encoder.parameters():
                param.requires_grad = False
            
            # SAMのマスクデコーダを抽出し、訓練可能にする
            self.sam_mask_decoder = sam.mask_decoder
            for param in self.sam_mask_decoder.parameters():
                param.requires_grad = True
        else:
            print("SAMチェックポイントが指定されていません。SAMコンポーネントは初期化されません。")
            self.sam_image_encoder = None
            self.sam_mask_decoder = None

        # 4. MLPプロジェクタの定義 (GemmaとSAMを繋ぐ橋)
        print("MLPプロジェクタを初期化中...")
        self.mlp_projector = nn.Sequential(
            nn.Linear(config.gemma_hidden_size, config.gemma_hidden_size),
            nn.GELU(),
            nn.Linear(config.gemma_hidden_size, config.sam_prompt_embed_dim),
        ).to(torch.bfloat16)

        # 5. 特別なセグメンテーショントークンを語彙に追加
        print("セグメンテーショントークンを追加中...")
        self.seg_token = config.seg_token
        
        # トークナイザーにSEGトークンを追加
        if self.seg_token not in self.gemma_processor.tokenizer.get_vocab():
            self.gemma_processor.tokenizer.add_tokens([self.seg_token], special_tokens=True)
            self.gemma_model.resize_token_embeddings(len(self.gemma_processor.tokenizer))
            print(f"✅ {self.seg_token}トークンが追加されました")
        
        # SEGトークンのIDを取得
        self.seg_token_id = self.gemma_processor.tokenizer.convert_tokens_to_ids(self.seg_token)
        
        print("LISA-Gemmaモデルの初期化が完了しました")

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

    def forward(
        self,
        image,  # PIL Image
        text_prompt: str,
        generate_mask: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        LISA-Gemmaのフォワードパス
        """
        try:
            # 1. Gemma-3の公式方式で入力を準備
            gemma_inputs = self.prepare_multimodal_input(image, text_prompt)
            
            # デバイスに移動
            device = next(self.gemma_model.parameters()).device
            gemma_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in gemma_inputs.items()}
            
            # 2. Gemmaモデルでテキスト生成とセグメンテーション判定
            with torch.inference_mode():
                gemma_outputs = self.gemma_model(
                    **gemma_inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            results = {
                "gemma_logits": gemma_outputs.logits,
                "hidden_states": gemma_outputs.hidden_states,
            }
            
            # 3. SEGトークンが生成された場合のマスク生成
            if generate_mask and self.sam_image_encoder is not None:
                # SEGトークンの存在をチェック
                generated_ids = torch.argmax(gemma_outputs.logits, dim=-1)
                seg_positions = (generated_ids == self.seg_token_id).nonzero(as_tuple=True)
                
                if len(seg_positions[0]) > 0:
                    print(f"SEGトークンが{len(seg_positions[0])}個検出されました")
                    
                    # SAM用の画像前処理（1024x1024にリサイズ）
                    sam_image = image.resize((1024, 1024))
                    sam_image_tensor = torch.tensor(np.array(sam_image)).permute(2, 0, 1).float()
                    sam_image_tensor = sam_image_tensor.unsqueeze(0).to(device)
                    
                    # SAMの画像エンコーディング
                    with torch.no_grad():
                        sam_features = self.sam_image_encoder(sam_image_tensor)
                    
                    # SEGトークンの隠れ状態を抽出
                    last_hidden = gemma_outputs.hidden_states[-1]
                    seg_embeddings = []
                    
                    for batch_idx, token_idx in zip(seg_positions[0], seg_positions[1]):
                        seg_hidden = last_hidden[batch_idx, token_idx]
                        seg_embedding = self.mlp_projector(seg_hidden.unsqueeze(0))
                        seg_embeddings.append(seg_embedding)
                    
                    if seg_embeddings:
                        # SAMデコーダでマスク生成
                        prompt_embeddings = torch.stack(seg_embeddings)
                        
                        # ダミーのdense embeddings
                        dense_embeddings = torch.zeros(
                            (prompt_embeddings.size(0), 256, 256),
                            device=device,
                            dtype=torch.bfloat16
                        )
                        
                        # マスク予測
                        masks, iou_pred = self.sam_mask_decoder(
                            image_embeddings=sam_features,
                            image_pe=self.sam_mask_decoder.get_dense_pe(),
                            sparse_prompt_embeddings=prompt_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                        )
                        
                        results["predicted_masks"] = masks
                        results["iou_predictions"] = iou_pred
                else:
                    print("SEGトークンが検出されませんでした")
                    results["predicted_masks"] = None
            
            return results
            
        except Exception as e:
            print(f"フォワードパス中にエラーが発生: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

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