"""
LISA-Llama4 Model Implementation
Llama-4-Scout-17B-16E-Instruct + SAM integration for semantic segmentation
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Llama4ForConditionalGeneration, 
    AutoProcessor,
    PreTrainedModel,
    PretrainedConfig
)

from .segment_anything import build_sam_vit_h
from .losses import dice_loss, sigmoid_ce_loss


class Llama4LisaConfig(PretrainedConfig):
    """Configuration for Llama4-LISA model"""
    
    model_type = "llama4_lisa"
    
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        hidden_size: int = 4096,
        out_dim: int = 256,
        train_mask_decoder: bool = True,
        vision_pretrained: Optional[str] = None,
        sam_checkpoint_path: str = "./sam_vit_h_4b8939.pth",
        ce_loss_weight: float = 1.0,
        dice_loss_weight: float = 0.5,
        bce_loss_weight: float = 2.0,
        max_images: int = 5,
        use_native_vision: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.model_id = model_id
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.train_mask_decoder = train_mask_decoder
        self.vision_pretrained = vision_pretrained
        self.sam_checkpoint_path = sam_checkpoint_path
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.max_images = max_images
        self.use_native_vision = use_native_vision


class Llama4LisaMetaModel:
    """Meta model for Llama4-LISA integration"""
    
    def __init__(self, config: Llama4LisaConfig, **kwargs):
        super().__init__(config)
        self.config = config
        
        # Llama-4 native multimodal capabilities
        self.native_vision = config.use_native_vision
        self.max_images = config.max_images
        
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs.get("train_mask_decoder", True)
            self.config.out_dim = kwargs.get("out_dim", 256)
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config: Llama4LisaConfig):
        """Initialize LISA-specific modules"""
        
        # SAM Visual Model (external segmentation dedicated)
        print(f"🔧 Initializing SAM from: {config.sam_checkpoint_path}")
        self.visual_model = build_sam_vit_h(checkpoint=config.sam_checkpoint_path)
        
        # Freeze SAM parameters (independent from Llama-4 vision)
        for param in self.visual_model.parameters():
            param.requires_grad = False
            
        # Only train mask decoder if specified
        if config.train_mask_decoder:
            print("🎯 Training SAM mask decoder")
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
        else:
            self.visual_model.mask_decoder.eval()

        # Projection layer: Llama-4 hidden_size → SAM hidden_size
        print(f"🔗 Creating projection layer: {config.hidden_size} → {config.out_dim}")
        text_fc = [
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size, config.out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        
        # Enable training for projection layers
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class Llama4LisaModel(Llama4LisaMetaModel, Llama4ForConditionalGeneration):
    """
    Llama4-LISA Model combining:
    1. Llama-4-Scout native multimodal capabilities
    2. SAM for precise segmentation
    3. Projection layers for feature integration
    """
    
    def __init__(self, config: Llama4LisaConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        # Llama-4 specific settings
        self.config.use_cache = False  # Disabled for training
        
        print(f"✅ Llama4-LISA Model initialized")
        print(f"   - Hidden size: {config.hidden_size}")
        print(f"   - Output dim: {config.out_dim}")  
        print(f"   - Native vision: {config.use_native_vision}")
        print(f"   - Max images: {config.max_images}")


class LISALlama4ForCausalLM(Llama4LisaModel):
    """
    Main LISA-Llama4 model for causal language modeling with segmentation
    """
    
    def __init__(self, config: Llama4LisaConfig, **kwargs):
        # Loss weights configuration
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", config.ce_loss_weight)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", config.dice_loss_weight)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", config.bce_loss_weight)
        
        # SEG token ID (will be set after tokenizer initialization)
        self.seg_token_idx = kwargs.pop("seg_token_idx", None)
        
        super().__init__(config, **kwargs)
        
        print(f"🎛️ Loss weights configured:")
        print(f"   - CE Loss: {self.ce_loss_weight}")
        print(f"   - Dice Loss: {self.dice_loss_weight}")
        print(f"   - BCE Loss: {self.bce_loss_weight}")
        
    def set_seg_token_idx(self, seg_token_idx: int):
        """Set SEG token index after tokenizer initialization"""
        self.seg_token_idx = seg_token_idx
        print(f"🎯 SEG token ID set to: {seg_token_idx}")

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        """
        Extract visual embeddings using SAM image encoder
        Input: pixel_values (batch_size, 3, 1024, 1024) for SAM
        Output: image_embeddings (batch_size, 256, 64, 64)
        """
        with torch.no_grad():
            image_embeddings_list = []
            batch_size = pixel_values.shape[0]
            
            for i in range(batch_size):
                torch.cuda.empty_cache()
                # SAM image encoder
                embeddings = self.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(embeddings)
                
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
            
        return image_embeddings

    def forward(self, **kwargs):
        """Forward pass routing"""
        if "past_key_values" in kwargs:
            # Standard generation mode
            return super().forward(**kwargs)
        
        # Training/inference mode with segmentation
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,  # Llama-4 native vision
        sam_images: Optional[torch.FloatTensor] = None,    # SAM images (1024x1024)
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        seg_token_positions: Optional[torch.BoolTensor] = None,
        ground_truth_masks: Optional[List[torch.FloatTensor]] = None,
        has_masks: Optional[List[bool]] = None,
        inference: bool = False,
        **kwargs,
    ):
        """
        Main forward pass for LISA-Llama4
        """
        
        # Step 1: Process images with SAM if provided
        image_embeddings = None
        if sam_images is not None:
            image_embeddings = self.get_visual_embs(sam_images)
            batch_size = image_embeddings.shape[0]
        else:
            batch_size = input_ids.shape[0]

        # Step 2: Llama-4 forward pass (native multimodal)
        llama_outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,  # Llama-4 native vision
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )
        
        hidden_states = llama_outputs.hidden_states[-1]  # Last layer hidden states
        
        # Step 3: Extract features at SEG token positions
        pred_masks = None
        if seg_token_positions is not None and seg_token_positions.any():
            # Get hidden states at SEG token positions
            seg_hidden_states = hidden_states[seg_token_positions]
            
            # Project to SAM feature space
            projected_features = self.text_hidden_fcs[0](seg_hidden_states)
            
            # SAM mask prediction
            if image_embeddings is not None:
                pred_masks = []
                for i in range(batch_size):
                    if i < len(projected_features):
                        # Use SAM mask decoder
                        masks, _, _ = self.visual_model.mask_decoder(
                            image_embeddings=image_embeddings[i:i+1],
                            image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=projected_features[i:i+1].unsqueeze(0),
                            dense_prompt_embeddings=torch.zeros_like(
                                projected_features[i:i+1].unsqueeze(0)
                            ),
                            multimask_output=False,
                        )
                        pred_masks.append(masks.squeeze(1))
                
                pred_masks = torch.cat(pred_masks, dim=0) if pred_masks else None

        # Step 4: Compute segmentation losses if training
        total_loss = llama_outputs.loss if llama_outputs.loss is not None else 0
        seg_loss = 0
        dice_loss_val = 0
        bce_loss_val = 0
        
        if not inference and pred_masks is not None and ground_truth_masks is not None:
            # Filter samples with masks
            valid_indices = [i for i, has_mask in enumerate(has_masks or [True] * len(ground_truth_masks)) if has_mask]
            
            if valid_indices:
                valid_pred_masks = pred_masks[valid_indices]
                valid_gt_masks = torch.stack([ground_truth_masks[i] for i in valid_indices])
                
                # Compute segmentation losses
                num_masks = len(valid_indices)
                dice_loss_val = dice_loss(valid_pred_masks, valid_gt_masks, num_masks)
                bce_loss_val = sigmoid_ce_loss(valid_pred_masks, valid_gt_masks, num_masks)
                
                # Weighted loss combination
                seg_loss = (
                    self.dice_loss_weight * dice_loss_val +
                    self.bce_loss_weight * bce_loss_val
                )
                
                # Add to total loss
                if isinstance(total_loss, torch.Tensor):
                    total_loss = total_loss + seg_loss
                else:
                    total_loss = seg_loss

        # Prepare outputs
        from types import SimpleNamespace
        
        outputs = SimpleNamespace()
        outputs.loss = total_loss
        outputs.logits = llama_outputs.logits
        outputs.hidden_states = llama_outputs.hidden_states
        outputs.pred_masks = pred_masks
        outputs.seg_loss = seg_loss
        outputs.dice_loss = dice_loss_val
        outputs.bce_loss = bce_loss_val
        outputs.ce_loss = llama_outputs.loss if llama_outputs.loss is not None else 0
        
        return outputs

    def evaluate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        sam_images: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        max_new_tokens: int = 32,
        tokenizer = None,
    ):
        """Evaluation mode for inference"""
        
        # Set to eval mode
        self.eval()
        
        with torch.no_grad():
            # Generate text response
            generated_ids = self.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.2,
                top_p=0.9,
            )
            
            # Decode generated text
            if tokenizer is not None:
                generated_text = tokenizer.batch_decode(
                    generated_ids[:, input_ids.shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
            else:
                generated_text = None
            
            # Get segmentation masks
            seg_token_positions = input_ids == self.seg_token_idx
            if seg_token_positions.any():
                outputs = self.model_forward(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    sam_images=sam_images,
                    attention_mask=attention_mask,
                    seg_token_positions=seg_token_positions,
                    inference=True,
                )
                pred_masks = outputs.pred_masks
            else:
                pred_masks = None
        
        return {
            "generated_ids": generated_ids,
            "generated_text": generated_text,
            "pred_masks": pred_masks,
        }

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        config: Optional[Llama4LisaConfig] = None,
        **kwargs
    ):
        """Load pretrained Llama4-LISA model"""
        
        if config is None:
            config = Llama4LisaConfig(model_id=model_id, **kwargs)
        
        # Load base Llama-4 model
        print(f"🔄 Loading Llama-4 model from: {model_id}")
        
        # Create model instance
        model = cls(config, **kwargs)
        
        print(f"✅ Llama4-LISA model loaded successfully")
        return model


# Factory function
def create_llama4_lisa_model(
    model_id: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    sam_checkpoint_path: str = "./sam_vit_h_4b8939.pth",
    train_mask_decoder: bool = True,
    **kwargs
) -> LISALlama4ForCausalLM:
    """Factory function to create Llama4-LISA model"""
    
    config = Llama4LisaConfig(
        model_id=model_id,
        sam_checkpoint_path=sam_checkpoint_path,
        train_mask_decoder=train_mask_decoder,
        **kwargs
    )
    
    return LISALlama4ForCausalLM.from_pretrained(
        model_id=model_id,
        config=config,
        **kwargs
    ) 