"""
Llama4 Dual Stream Processing
Handles both Llama-4 native multimodal processing and SAM preprocessing
"""

from typing import Dict, List, Optional, Union, Tuple
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor
import re


class Llama4DualStreamProcessor:
    """
    Dual Stream Processor for LISA-Llama4
    
    Processes images for both:
    1. Llama-4 native multimodal processing (variable resolution)
    2. SAM processing (1024x1024 fixed)
    """
    
    def __init__(self, model_id: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"):
        """
        Initialize Llama4 dual stream processor
        
        Args:
            model_id: Hugging Face model ID for Llama-4
        """
        self.model_id = model_id
        
        # Initialize Llama-4 processor (native multimodal)
        print(f"🔄 Loading Llama-4 processor from: {model_id}")
        self.llama4_processor = AutoProcessor.from_pretrained(model_id)
        
        # Add SEG token if not present
        self.seg_token = "[SEG]"
        if self.seg_token not in self.llama4_processor.tokenizer.get_vocab():
            print(f"➕ Adding {self.seg_token} token to tokenizer")
            self.llama4_processor.tokenizer.add_tokens([self.seg_token])
            
        self.seg_token_id = self.llama4_processor.tokenizer.convert_tokens_to_ids(self.seg_token)
        print(f"🎯 SEG token ID: {self.seg_token_id}")
        
        # Image processing statistics
        self.processed_count = 0
        
    def preprocess_dual_stream(
        self, 
        image: Union[str, Image.Image], 
        text_prompt: str
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess image and text for dual stream processing
        
        Args:
            image: Image path or PIL Image
            text_prompt: Text prompt containing [SEG] token
            
        Returns:
            Dictionary containing processed data for both streams
        """
        
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image or file path")
            
        # 1. Llama-4 native processing
        llama4_inputs = self._process_llama4_stream(image, text_prompt)
        
        # 2. SAM preprocessing (1024x1024)
        sam_image = self._process_sam_stream(image)
        
        # 3. Find SEG token positions
        seg_token_positions = self._find_seg_token_positions(llama4_inputs["input_ids"])
        
        self.processed_count += 1
        
        return {
            # Llama-4 native inputs
            "llama4_pixel_values": llama4_inputs["pixel_values"],
            "input_ids": llama4_inputs["input_ids"],
            "attention_mask": llama4_inputs["attention_mask"],
            
            # SAM inputs
            "sam_images": sam_image,
            
            # SEG token information
            "seg_token_positions": seg_token_positions,
            
            # Metadata
            "original_image_size": image.size,
            "text_prompt": text_prompt,
        }
    
    def _process_llama4_stream(
        self, 
        image: Image.Image, 
        text_prompt: str
    ) -> Dict[str, torch.Tensor]:
        """
        Process image and text using Llama-4's native multimodal processor
        
        Args:
            image: PIL Image
            text_prompt: Text prompt
            
        Returns:
            Processed inputs for Llama-4
        """
        
        # Format message for Llama-4 chat template
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt}
            ]
        }]
        
        # Apply chat template and tokenize
        try:
            processed = self.llama4_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": processed["pixel_values"],
                "input_ids": processed["input_ids"],
                "attention_mask": processed["attention_mask"]
            }
            
        except Exception as e:
            print(f"⚠️ Error in Llama-4 processing: {e}")
            raise
    
    def _process_sam_stream(self, image: Image.Image) -> torch.Tensor:
        """
        Process image for SAM (1024x1024 with aspect ratio preservation)
        
        Args:
            image: PIL Image
            
        Returns:
            Processed image tensor for SAM (3, 1024, 1024)
        """
        
        target_size = 1024
        
        # Get original dimensions
        w, h = image.size
        
        # Calculate scale factor (preserve aspect ratio)
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized_image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # Create padded canvas
        padded_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        
        # Calculate paste position (center)
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        
        # Paste resized image onto padded canvas
        padded_image.paste(resized_image, (paste_x, paste_y))
        
        # Convert to tensor
        image_array = np.array(padded_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # CHW format
        
        return image_tensor
    
    def _find_seg_token_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Find positions of [SEG] tokens in input_ids
        
        Args:
            input_ids: Tokenized input (batch_size, seq_len)
            
        Returns:
            Boolean tensor indicating SEG token positions
        """
        
        seg_positions = (input_ids == self.seg_token_id)
        return seg_positions
    
    def batch_preprocess(
        self, 
        images: List[Union[str, Image.Image]], 
        text_prompts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Batch preprocessing for multiple samples
        
        Args:
            images: List of images (paths or PIL Images)
            text_prompts: List of text prompts
            
        Returns:
            Batched processed data
        """
        
        if len(images) != len(text_prompts):
            raise ValueError("Number of images and text prompts must match")
        
        batch_data = []
        
        for image, text_prompt in zip(images, text_prompts):
            processed = self.preprocess_dual_stream(image, text_prompt)
            batch_data.append(processed)
        
        # Stack tensors
        batched = {
            "llama4_pixel_values": torch.cat([item["llama4_pixel_values"] for item in batch_data], dim=0),
            "input_ids": torch.cat([item["input_ids"] for item in batch_data], dim=0),
            "attention_mask": torch.cat([item["attention_mask"] for item in batch_data], dim=0),
            "sam_images": torch.stack([item["sam_images"] for item in batch_data], dim=0),
            "seg_token_positions": torch.cat([item["seg_token_positions"] for item in batch_data], dim=0),
        }
        
        return batched
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics"""
        return {
            "processed_count": self.processed_count,
            "seg_token_id": self.seg_token_id,
            "vocab_size": len(self.llama4_processor.tokenizer),
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.processed_count = 0


def create_llama4_processor(model_id: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct") -> Llama4DualStreamProcessor:
    """
    Factory function to create Llama4 dual stream processor
    
    Args:
        model_id: Hugging Face model ID
        
    Returns:
        Llama4DualStreamProcessor instance
    """
    return Llama4DualStreamProcessor(model_id=model_id)


# Testing functions
def test_dual_stream_processing():
    """Test dual stream processing functionality"""
    
    print("🧪 Testing Llama4 Dual Stream Processing")
    
    # Create processor
    processor = create_llama4_processor()
    
    # Create test image
    test_image = Image.new('RGB', (800, 600), color='red')
    test_prompt = "Show me the red area in this image. [SEG]"
    
    try:
        # Process
        result = processor.preprocess_dual_stream(test_image, test_prompt)
        
        print("✅ Processing successful!")
        print(f"   Llama-4 pixel values shape: {result['llama4_pixel_values'].shape}")
        print(f"   Input IDs shape: {result['input_ids'].shape}")
        print(f"   SAM image shape: {result['sam_images'].shape}")
        print(f"   SEG token positions: {result['seg_token_positions'].sum().item()} found")
        print(f"   Original image size: {result['original_image_size']}")
        
        # Check expected shapes
        assert result['sam_images'].shape == (3, 1024, 1024), "SAM image shape incorrect"
        assert result['seg_token_positions'].any(), "No SEG tokens found"
        
        print("✅ All tests passed!")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    test_dual_stream_processing() 