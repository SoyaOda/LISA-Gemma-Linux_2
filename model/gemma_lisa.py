# model/gemma_lisa.py
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any

from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from segment_anything import sam_model_registry
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

# LISA-Gemmaモデルのカスタム設定クラス
class LisaGemmaConfig(PretrainedConfig):
    model_type = "lisa_gemma"

    def __init__(
        self,
        gemma_model_id="google/gemma-2-2b-it",
        sam_checkpoint_path=None,
        seg_token_idx=0,
        gemma_hidden_size=2304,  # Gemma 2 2B の hidden_size
        sam_prompt_embed_dim=256,
        **kwargs,
    ):
        self.gemma_model_id = gemma_model_id
        self.sam_checkpoint_path = sam_checkpoint_path
        self.seg_token_idx = seg_token_idx
        self.gemma_hidden_size = gemma_hidden_size
        self.sam_prompt_embed_dim = sam_prompt_embed_dim
        super().__init__(**kwargs)

class LisaGemmaForCausalLM(PreTrainedModel):
    config_class = LisaGemmaConfig

    def __init__(self, config: LisaGemmaConfig):
        super().__init__(config)

        # 1. Gemma-2 LLMのロード
        self.gemma_model = AutoModelForCausalLM.from_pretrained(
            config.gemma_model_id,
            torch_dtype=torch.bfloat16,  # bf16で効率化
            trust_remote_code=True,
        )

        # 2. SAMコンポーネントのロードと凍結
        if config.sam_checkpoint_path and config.sam_checkpoint_path != "":
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
            # SAMチェックポイントが指定されていない場合はダミーを作成
            print("警告: SAMチェックポイントが指定されていません。ダミーコンポーネントを使用します。")
            self.sam_image_encoder = None
            self.sam_mask_decoder = None

        # 3. MLPプロジェクタの定義 (GemmaとSAMを繋ぐ橋)
        self.mlp_projector = nn.Sequential(
            nn.Linear(config.gemma_hidden_size, config.gemma_hidden_size),
            nn.GELU(),
            nn.Linear(config.gemma_hidden_size, config.sam_prompt_embed_dim),
        )

        # 4. モデルの他の部分のパラメータ管理
        # LLMの大部分は凍結 (LoRAでファインチューニング)
        for param in self.gemma_model.parameters():
            param.requires_grad = False
        
        # 埋め込み層とLMヘッドは訓練可能にする
        if hasattr(self.gemma_model, 'get_input_embeddings'):
            self.gemma_model.get_input_embeddings().requires_grad_(True)
        if hasattr(self.gemma_model, 'get_output_embeddings'):
            self.gemma_model.get_output_embeddings().requires_grad_(True)

    def get_input_embeddings(self) -> nn.Module:
        return self.gemma_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.gemma_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.gemma_model.get_output_embeddings()

    def forward(
        self,
        images_for_gemma: Optional[torch.Tensor] = None,
        images_for_sam: Optional[torch.Tensor] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        seg_token_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        デュアルパスウェイ・フォワードパスの実装
        """
        # ======================================================================
        # パスウェイ 1: SAMの画像エンコーディング (セグメンテーション用)
        # ======================================================================
        sam_image_features = None
        if self.sam_image_encoder is not None and images_for_sam is not None:
            # SAMの画像エンコーダは凍結されているため、勾配計算は不要
            with torch.no_grad():
                sam_image_features = self.sam_image_encoder(images_for_sam)

        # ======================================================================
        # パスウェイ 2: Gemmaの推論 (意図理解用)
        # ======================================================================
        # Gemmaモデルに画像とテキストを入力し、出力を得る
        # images_for_gemmaがあれば処理に含める
        gemma_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'output_hidden_states': True,  # 隠れ状態を取得するために必要
        }
        
        # Gemma 2の場合、pixel_valuesパラメータがある場合のみ追加
        if images_for_gemma is not None:
            gemma_kwargs['pixel_values'] = images_for_gemma
        
        gemma_outputs = self.gemma_model(**gemma_kwargs)
        
        # テキスト生成の損失（VQAタスクなどで使用）
        text_loss = gemma_outputs.loss
        
        # ======================================================================
        # 橋渡し: MLPプロジェクタによる特徴量変換
        # ======================================================================
        last_hidden_state = gemma_outputs.hidden_states[-1]
        
        # バッチ内の<SEG>トークンの隠れ状態を抽出
        seg_token_embedding = None
        if seg_token_mask is not None and seg_token_mask.sum() > 0:
            # seg_token_maskは、<SEG>トークンの位置がTrueのブールマスク
            # (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
            seg_token_mask_expanded = seg_token_mask.unsqueeze(-1).expand_as(last_hidden_state)
            
            # <SEG>トークンの隠れ状態のみを抽出 (sum > 0 の場合のみ)
            # (num_seg_tokens, hidden_size)
            h_seg_raw = last_hidden_state[seg_token_mask_expanded].view(-1, last_hidden_state.size(-1))
            
            # MLPプロジェクタを通して、SAMが理解できる埋め込みに変換
            # (num_seg_tokens, sam_prompt_embed_dim)
            seg_token_embedding = self.mlp_projector(h_seg_raw)

        # ======================================================================
        # 最終段階: SAMマスクデコーダによるマスク生成
        # ======================================================================
        predicted_masks = None
        if (seg_token_embedding is not None and 
            self.sam_mask_decoder is not None and 
            sam_image_features is not None):
            
            # SAMデコーダへの入力を作成
            # (batch_size, num_prompts, embed_dim) -> (num_seg_tokens, 1, embed_dim)
            sparse_prompt_embeddings = seg_token_embedding.unsqueeze(1)
            
            # デンスなプロンプトは使用しない
            dense_prompt_embeddings = torch.zeros(
                (seg_token_embedding.size(0), 256, 256),
                device=seg_token_embedding.device,
                dtype=seg_token_embedding.dtype
            )

            # SAMデコーダを実行してマスクを予測
            # 簡単のため、バッチ内の全ての<SEG>トークンが同じ画像特徴を使うと仮定
            # 実際の実装では、どのトークンがどの画像に対応するかを管理する必要がある
            
            # seg_token_maskから、各トークンがどのバッチインデックスに属するかを取得
            if seg_token_mask.sum() > 0:
                batch_indices = torch.where(seg_token_mask)[0]  # バッチインデックスを取得
                
                # 重複を除去して、ユニークなバッチインデックスのみを使用
                unique_batch_indices = torch.unique(batch_indices)
                
                if len(unique_batch_indices) > 0:
                    # 対応する画像特徴を選択（最初のバッチアイテムを使用）
                    corresponding_sam_features = sam_image_features[unique_batch_indices[:1]]

                    try:
                        low_res_masks, iou_predictions = self.sam_mask_decoder(
                            image_embeddings=corresponding_sam_features,
                            image_pe=self.sam_mask_decoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_prompt_embeddings[:1],  # 最初のトークンのみ使用
                            dense_prompt_embeddings=dense_prompt_embeddings[:1],
                            multimask_output=False,  # LISAは単一マスクを予測
                        )
                        
                        predicted_masks = low_res_masks
                    except Exception as e:
                        print(f"SAMデコーダエラー: {e}")
                        predicted_masks = None

        return {
            "text_loss": text_loss,
            "predicted_masks": predicted_masks,
            "logits": gemma_outputs.logits,
        } 