"""
LISA-Llama4 Model Implementation
Llama-4-Scout-17B-16E-Instruct + SAM integration for semantic segmentation

Based on:
- Original-LISA-Code/model/LISA.py
- model/gemma_lisa.py  
- Llama-4 official specifications from web research
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Llama4ForConditionalGeneration, 
    AutoProcessor,
    PreTrainedModel,
    PretrainedConfig,
    LlamaConfig
)

from .segment_anything import build_sam_vit_h
from .losses import dice_loss, sigmoid_ce_loss
import os
import warnings

# =====================================================
# Llama4-LISA Configuration Classes (Official-based)
# =====================================================

class Llama4TextConfig(PretrainedConfig):
    """Llama4 Text Configuration based on official specifications"""
    model_type = "llama"
    
    def __init__(
        self,
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=10_000_000,  # 10M context
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        attention_bias=False,
        hidden_act="silu",
        pretraining_tp=1,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=128000,
        eos_token_id=128001,
        tie_word_embeddings=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.hidden_act = hidden_act
        self.pretraining_tp = pretraining_tp
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

class Llama4VisionConfig(PretrainedConfig):
    """Llama4 Vision Configuration based on SigLIP specifications"""
    model_type = "siglip_vision_model"
    
    def __init__(
        self,
        hidden_size=1024,
        image_size=1120,
        intermediate_size=4096,
        num_attention_heads=16,
        num_hidden_layers=24,
        num_channels=3,
        patch_size=14,
        attention_dropout=0.0,
        layer_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        
        super().__init__(**kwargs)

class Llama4LisaConfig(PretrainedConfig):
    """Complete Llama4-LISA Configuration with all required parameters"""
    model_type = "llama4_lisa"
    
    def __init__(
        self,
        # === Basic Llama-4 Settings (Official Spec) ===
        model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=10_000_000,  # 10M context
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        attention_bias=False,
        hidden_act="silu",
        
        # === MoE Settings (Scout: 16 experts) ===
        num_experts=16,
        num_experts_per_tok=1,
        
        # === Multimodal Settings ===
        text_config=None,
        vision_config=None,
        
        # === LISA-specific Settings ===
        sam_checkpoint_path="/lambda/nfs/lisa-gemma-project-fs/data/weights/sam_vit_h_4b8939.pth",
        train_mask_decoder=True,
        out_dim=256,
        
        # === Loss Weights (Original-LISA style) ===
        ce_loss_weight=1.0,
        dice_loss_weight=0.5,
        bce_loss_weight=2.0,
        
        # === Special Tokens ===
        seg_token="[SEG]",
        image_token="<image>",
        
        **kwargs
    ):
        # Store all basic Llama-4 parameters
        self.model_id = model_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.hidden_act = hidden_act
        
        # MoE configuration
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        
        # Initialize text_config with official Llama settings
        if text_config is None:
            text_config = {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_key_value_heads,
                "max_position_embeddings": max_position_embeddings,
                "rms_norm_eps": rms_norm_eps,
                "rope_theta": rope_theta,
                "attention_bias": attention_bias,
                "hidden_act": hidden_act,
                "model_type": "llama",
                "torch_dtype": "bfloat16"
            }
        
        # Initialize vision_config with SigLIP-equivalent settings
        if vision_config is None:
            vision_config = {
                "hidden_size": 1024,
                "image_size": 1120,
                "intermediate_size": 4096,
                "num_attention_heads": 16,
                "num_hidden_layers": 24,
                "num_channels": 3,
                "patch_size": 14,
                "attention_dropout": 0.0,
                "layer_norm_eps": 1e-6,
                "hidden_act": "gelu_pytorch_tanh",
                "model_type": "siglip_vision_model"
            }
        
        self.text_config = Llama4TextConfig(**text_config)
        self.vision_config = Llama4VisionConfig(**vision_config)
        
        # LISA-specific parameters
        self.sam_checkpoint_path = sam_checkpoint_path
        self.train_mask_decoder = train_mask_decoder
        self.out_dim = out_dim
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight
        
        # Special tokens
        self.seg_token = seg_token
        self.image_token = image_token
        
        super().__init__(**kwargs)

# =====================================================
# LISA-Llama4 Model Implementation
# =====================================================

class Llama4LisaForConditionalGeneration(Llama4ForConditionalGeneration):
    """
    LISA-Llama4: Integration of Llama-4 native multimodal + SAM segmentation
    
    Architecture (based on Original-LISA and Gemma-LISA patterns):
    - Llama-4 native multimodal capabilities for vision-language understanding
    - SAM integration for precise segmentation tasks
    - MLP projector connecting Llama-4 hidden states to SAM prompts
    - Dual-stream processing for optimal performance
    """
    
    config_class = Llama4LisaConfig
    
    def __init__(self, config: Llama4LisaConfig):
        super().__init__(config)
        
        print(f"🚀 Initializing LISA-Llama4 model ({config.model_id})")
        
        # Store configuration
        self.config = config
        
        # 1. Initialize SAM components (Original-LISA style)
        print(f"📦 Loading SAM components from {config.sam_checkpoint_path}")
        if config.sam_checkpoint_path and os.path.exists(config.sam_checkpoint_path):
            try:
                self.visual_model = build_sam_vit_h(config.sam_checkpoint_path)
                
                # Freeze SAM image encoder (Original-LISA pattern)
                for param in self.visual_model.parameters():
                    param.requires_grad = False
                
                # Enable mask decoder training if specified
                if config.train_mask_decoder:
                    self.visual_model.mask_decoder.train()
                    for param in self.visual_model.mask_decoder.parameters():
                        param.requires_grad = True
                    print("✅ SAM mask decoder set to trainable")
                else:
                    self.visual_model.mask_decoder.eval()
                    print("🔒 SAM mask decoder frozen")
                
                # Move SAM to same device as Llama-4
                device = next(self.parameters()).device
                self.visual_model = self.visual_model.to(device)
                
                print("✅ SAM integration complete")
                
            except Exception as e:
                print(f"❌ SAM initialization failed: {e}")
                self.visual_model = None
                warnings.warn("SAM components not available - segmentation disabled")
        else:
            print("⚠️  SAM checkpoint not found - segmentation disabled")
            self.visual_model = None
        
        # 2. MLP Projector (Gemma-LISA style) - connects Llama-4 to SAM
        print("🔗 Initializing MLP projector...")
        device = next(self.parameters()).device
        self.mlp_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.out_dim),
            nn.Dropout(0.0),
        ).to(device).to(torch.bfloat16)
        
        # Make projector trainable
        for param in self.mlp_projector.parameters():
            param.requires_grad = True
        
        # 3. Setup SEG token (Original-LISA style)
        self._setup_seg_token()
        
        print("✅ LISA-Llama4 initialization complete")
    
    def _setup_seg_token(self):
        """Setup segmentation token (based on Original-LISA)"""
        print(f"🎯 Setting up segmentation token: {self.config.seg_token}")
        
        # For now, we'll get the token ID during training
        # This is because tokenizer might not be available during model init
        self.seg_token = self.config.seg_token
        self.seg_token_id = None  # Will be set when processor is available
        
        print(f"✅ SEG token setup complete: {self.seg_token}")
    
    def set_seg_token_id(self, token_id: int):
        """Set the segmentation token ID"""
        self.seg_token_id = token_id
        print(f"🎯 SEG token ID set: {token_id}")
    
    def has_sam_capability(self) -> bool:
        """Check if SAM is available"""
        return self.visual_model is not None
    
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        """
        Get SAM image embeddings (Original-LISA style)
        
        Args:
            pixel_values: Image tensor for SAM (1024x1024)
        
        Returns:
            SAM image embeddings
        """
        if not self.has_sam_capability():
            raise RuntimeError("SAM not available - cannot generate visual embeddings")
        
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        
        return image_embeddings
    
    def forward(
        self,
        input_ids=None,
        pixel_values=None,  # Llama-4 native multimodal input
        attention_mask=None,
        labels=None,
        images=None,  # SAM-specific image input (1024x1024) - Original-LISA style
        offset=None,  # Original-LISA style batch handling
        masks_list=None,  # Ground truth segmentation masks
        label_list=None,  # Original-LISA style labels
        resize_list=None,  # Original-LISA style resize info
        inference=False,
        **kwargs
    ):
        """
        Forward pass integrating Llama-4 native multimodal with SAM segmentation
        
        Based on Original-LISA forward pattern with Llama-4 adaptations
        """
        
        # Handle different input styles for compatibility
        if "past_key_values" in kwargs:
            # Standard generation mode
            return super().forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        
        # LISA-style segmentation mode
        return self._lisa_forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
            images=images,
            offset=offset,
            masks_list=masks_list,
            label_list=label_list,
            resize_list=resize_list,
            inference=inference,
            **kwargs
        )
    
    def _lisa_forward(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        labels,
        images,
        offset,
        masks_list,
        label_list,
        resize_list,
        inference,
        **kwargs
    ):
        """
        LISA-style forward pass (based on Original-LISA)
        """
        
        # 1. Get Llama-4 native multimodal outputs
        llama4_outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        # If no segmentation required, return Llama-4 outputs
        if not self.has_sam_capability() or images is None or self.seg_token_id is None:
            return llama4_outputs
        
        # 2. Process segmentation if [SEG] tokens present
        if self.seg_token_id in input_ids:
            seg_outputs = self._process_segmentation(
                hidden_states=llama4_outputs.hidden_states[-1] if hasattr(llama4_outputs, 'hidden_states') else None,
                input_ids=input_ids,
                images=images,
                offset=offset,
                masks_list=masks_list,
                label_list=label_list,
                resize_list=resize_list,
                inference=inference
            )
            
            # Combine outputs
            total_loss = llama4_outputs.loss if llama4_outputs.loss is not None else 0
            if seg_outputs and "seg_loss" in seg_outputs:
                total_loss = total_loss + seg_outputs["seg_loss"]
            
            # Return combined results
            result = {
                "loss": total_loss,
                "logits": llama4_outputs.logits,
                "seg_loss": seg_outputs.get("seg_loss", 0) if seg_outputs else 0,
                "pred_masks": seg_outputs.get("pred_masks", None) if seg_outputs else None,
            }
            
            # Add additional outputs from llama4_outputs if available
            if hasattr(llama4_outputs, '__dict__'):
                for key, value in llama4_outputs.__dict__.items():
                    if key not in result:
                        result[key] = value
            
            return result
        
        return llama4_outputs
    
    def _process_segmentation(
        self,
        hidden_states,
        input_ids,
        images,
        offset,
        masks_list,
        label_list,
        resize_list,
        inference
    ):
        """
        Process segmentation using SAM (Original-LISA style)
        """
        try:
            # 1. Get SAM image embeddings
            image_embeddings = self.get_visual_embs(images)
            batch_size = image_embeddings.shape[0]
            
            if offset is not None:
                assert batch_size == len(offset) - 1
            
            # 2. Find [SEG] token positions (Original-LISA style)
            seg_token_mask = input_ids[:, 1:] == self.seg_token_id
            seg_token_mask = torch.cat([
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().to(seg_token_mask.device),
            ], dim=1)
            
            # Handle IMAGE_TOKEN_INDEX offset (Original-LISA hack)
            seg_token_mask = torch.cat([
                torch.zeros((seg_token_mask.shape[0], 255)).bool().to(seg_token_mask.device), 
                seg_token_mask
            ], dim=1)
            
            # 3. Extract seg token embeddings and generate masks
            if hidden_states is not None and seg_token_mask.any():
                seg_embeddings = self._extract_seg_embeddings(hidden_states, seg_token_mask)
                pred_masks = self._generate_masks(image_embeddings, seg_embeddings, batch_size)
                
                # 4. Compute segmentation losses if ground truth available
                if not inference and masks_list is not None:
                    seg_loss = self._compute_segmentation_losses(pred_masks, masks_list, batch_size)
                    return {
                        "seg_loss": seg_loss,
                        "pred_masks": pred_masks
                    }
                else:
                    return {
                        "seg_loss": 0,
                        "pred_masks": pred_masks
                    }
            
            return {"seg_loss": 0, "pred_masks": None}
            
        except Exception as e:
            print(f"⚠️ Segmentation processing failed: {e}")
            return {"seg_loss": 0, "pred_masks": None}
    
    def _extract_seg_embeddings(self, hidden_states, seg_token_mask):
        """Extract embeddings for [SEG] tokens"""
        seg_embeddings = hidden_states[seg_token_mask]
        return self.mlp_projector(seg_embeddings)
    
    def _generate_masks(self, image_embeddings, seg_embeddings, batch_size):
        """Generate segmentation masks using SAM"""
        pred_masks = []
        
        for i in range(batch_size):
            # Use SAM to generate masks
            sparse_embeddings = seg_embeddings[i:i+1]  # Take one embedding per image
            dense_embeddings = self.visual_model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
            
            low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i:i+1],
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            pred_masks.append(low_res_masks)
        
        return torch.cat(pred_masks, dim=0) if pred_masks else None
    
    def _compute_segmentation_losses(self, pred_masks, masks_list, batch_size):
        """Compute segmentation losses (Original-LISA style)"""
        ce_loss = 0
        dice_loss_val = 0
        bce_loss = 0
        num_masks = 0
        
        for i in range(batch_size):
            if i < len(masks_list) and masks_list[i] is not None:
                target_mask = masks_list[i].to(pred_masks.device)
                pred_mask = pred_masks[i]
                
                # Resize if needed
                if pred_mask.shape[-2:] != target_mask.shape[-2:]:
                    pred_mask = F.interpolate(
                        pred_mask.unsqueeze(0),
                        size=target_mask.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
                # Compute losses
                bce_loss += sigmoid_ce_loss(pred_mask, target_mask.float(), 1)
                dice_loss_val += dice_loss(pred_mask, target_mask.float(), 1)
                num_masks += 1
        
        # Combine losses with weights
        if num_masks > 0:
            total_loss = (
                self.config.bce_loss_weight * bce_loss + 
                self.config.dice_loss_weight * dice_loss_val
            ) / num_masks
        else:
            total_loss = torch.tensor(0.0, device=pred_masks.device, requires_grad=True)
        
        return total_loss

# =====================================================
# Factory Functions
# =====================================================

def create_llama4_lisa_model(
    model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    sam_checkpoint_path="/lambda/nfs/lisa-gemma-project-fs/data/weights/sam_vit_h_4b8939.pth",
    train_mask_decoder=True,
    out_dim=256,
    **kwargs
):
    """
    Create LISA-Llama4 model with official specifications
    
    Args:
        model_id: Llama-4 model identifier
        sam_checkpoint_path: Path to SAM checkpoint
        train_mask_decoder: Whether to train SAM mask decoder
        out_dim: Output dimension for projector
        **kwargs: Additional configuration
    
    Returns:
        Configured LISA-Llama4 model
    """
    
    print(f"🚀 Creating LISA-Llama4 model: {model_id}")
    
    # Create configuration
    config = Llama4LisaConfig(
        model_id=model_id,
        sam_checkpoint_path=sam_checkpoint_path,
        train_mask_decoder=train_mask_decoder,
        out_dim=out_dim,
        **kwargs
    )
    
    # Initialize model with official Llama-4 settings
    print("📦 Loading base Llama-4 model...")
    model = Llama4LisaForConditionalGeneration(config)
    
    print("✅ LISA-Llama4 model created successfully")
    return model

def create_llama4_lisa_processor(
    model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    seg_token="[SEG]"
):
    """
    Create processor for LISA-Llama4
    
    Args:
        model_id: Llama-4 model identifier  
        seg_token: Segmentation token
    
    Returns:
        Configured processor
    """
    
    print(f"🔧 Creating LISA-Llama4 processor: {model_id}")
    
    # Load official Llama-4 processor
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Add segmentation token if not present
    if seg_token not in processor.tokenizer.get_vocab():
        num_added = processor.tokenizer.add_tokens([seg_token], special_tokens=True)
        print(f"✅ Added {num_added} special tokens to processor")
    
    print("✅ LISA-Llama4 processor created successfully")
    return processor

def get_model_info(model):
    """Get model information for debugging"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": (trainable_params / total_params) * 100,
        "has_sam": model.has_sam_capability(),
        "seg_token": getattr(model, 'seg_token', None),
        "seg_token_id": getattr(model, 'seg_token_id', None),
    }
    
    return info 