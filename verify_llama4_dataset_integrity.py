#!/usr/bin/env python3
"""
Llama4-LISA Dataset Integrity Verification Script
Verifies dataset loading and processing for LISA-Llama4
"""

import os
import sys
import torch
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configurations
try:
    import config_llama4 as config
    print("✅ Using config_llama4.py")
except ImportError:
    try:
        import config_linux as config
        print("⚠️ Falling back to config_linux.py")
    except ImportError:
        print("❌ No valid configuration found")
        sys.exit(1)

from utils.llama4_processing import Llama4DualStreamProcessor
from PIL import Image
import numpy as np


def verify_processor_initialization():
    """Verify Llama4 dual stream processor initialization"""
    print("\n" + "=" * 70)
    print("🔧 Testing Llama4 Processor Initialization")
    print("=" * 70)
    
    try:
        # Initialize processor
        processor = Llama4DualStreamProcessor(model_id=config.MODEL_ID)
        
        print(f"📋 Processor Details:")
        print(f"   Model ID: {processor.model_id}")
        print(f"   SEG token: {processor.seg_token}")
        print(f"   SEG token ID: {processor.seg_token_id}")
        
        # Check tokenizer
        vocab_size = len(processor.llama4_processor.tokenizer)
        print(f"   Vocabulary size: {vocab_size}")
        
        # Test SEG token
        test_text = "Show me the object [SEG] in this image."
        tokens = processor.llama4_processor.tokenizer.tokenize(test_text)
        print(f"   Test tokenization: {len(tokens)} tokens")
        
        # Check if SEG token is properly added
        seg_in_vocab = processor.seg_token in processor.llama4_processor.tokenizer.get_vocab()
        print(f"   SEG token in vocab: {seg_in_vocab}")
        
        if seg_in_vocab:
            print("✅ Processor initialization successful")
            return processor
        else:
            print("❌ SEG token not properly added")
            return None
            
    except Exception as e:
        print(f"❌ Processor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_dual_stream_processing(processor):
    """Verify dual stream processing functionality"""
    print("\n" + "=" * 70)
    print("⚡ Testing Dual Stream Processing")
    print("=" * 70)
    
    try:
        # Create test image
        test_image = Image.new('RGB', (800, 600), color=(255, 0, 0))  # Red image
        test_prompt = "Can you segment the red area in this image? [SEG]"
        
        print("🔧 Processing test sample...")
        print(f"   Image size: {test_image.size}")
        print(f"   Text prompt: {test_prompt}")
        
        # Process with dual stream
        result = processor.preprocess_dual_stream(test_image, test_prompt)
        
        print("\n📊 Processing Results:")
        print(f"   Llama-4 pixel values shape: {result['llama4_pixel_values'].shape}")
        print(f"   Input IDs shape: {result['input_ids'].shape}")
        print(f"   Attention mask shape: {result['attention_mask'].shape}")
        print(f"   SAM image shape: {result['sam_images'].shape}")
        print(f"   SEG token positions shape: {result['seg_token_positions'].shape}")
        print(f"   SEG tokens found: {result['seg_token_positions'].sum().item()}")
        print(f"   Original image size: {result['original_image_size']}")
        
        # Verify expected shapes
        expected_sam_shape = (3, 1024, 1024)
        if result['sam_images'].shape == expected_sam_shape:
            print("✅ SAM image shape correct")
        else:
            print(f"❌ SAM image shape incorrect: expected {expected_sam_shape}, got {result['sam_images'].shape}")
        
        # Verify SEG token detection
        if result['seg_token_positions'].any():
            print("✅ SEG token properly detected")
        else:
            print("❌ SEG token not detected")
        
        # Check tensor types and ranges
        print("\n🔍 Tensor Validation:")
        print(f"   Pixel values dtype: {result['llama4_pixel_values'].dtype}")
        print(f"   Pixel values range: [{result['llama4_pixel_values'].min():.3f}, {result['llama4_pixel_values'].max():.3f}]")
        print(f"   SAM image dtype: {result['sam_images'].dtype}")
        print(f"   SAM image range: [{result['sam_images'].min():.3f}, {result['sam_images'].max():.3f}]")
        print(f"   Input IDs dtype: {result['input_ids'].dtype}")
        print(f"   Input IDs range: [{result['input_ids'].min()}, {result['input_ids'].max()}]")
        
        print("✅ Dual stream processing successful")
        return result
        
    except Exception as e:
        print(f"❌ Dual stream processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_batch_processing(processor):
    """Verify batch processing capabilities"""
    print("\n" + "=" * 70)
    print("📦 Testing Batch Processing")
    print("=" * 70)
    
    try:
        # Create multiple test images
        images = [
            Image.new('RGB', (600, 400), color=(255, 0, 0)),    # Red
            Image.new('RGB', (800, 600), color=(0, 255, 0)),    # Green
            Image.new('RGB', (400, 800), color=(0, 0, 255)),    # Blue
        ]
        
        prompts = [
            "Segment the red region [SEG]",
            "Show me the green area [SEG]", 
            "Can you find the blue part? [SEG]"
        ]
        
        print(f"🔧 Processing batch of {len(images)} samples...")
        
        # Process batch
        batch_result = processor.batch_preprocess(images, prompts)
        
        print("\n📊 Batch Processing Results:")
        print(f"   Batch size: {len(images)}")
        print(f"   Llama-4 pixel values shape: {batch_result['llama4_pixel_values'].shape}")
        print(f"   Input IDs shape: {batch_result['input_ids'].shape}")
        print(f"   Attention mask shape: {batch_result['attention_mask'].shape}")
        print(f"   SAM images shape: {batch_result['sam_images'].shape}")
        print(f"   SEG token positions shape: {batch_result['seg_token_positions'].shape}")
        
        # Verify batch dimensions
        batch_size = len(images)
        if batch_result['sam_images'].shape[0] == batch_size:
            print("✅ Batch processing successful")
            return batch_result
        else:
            print("❌ Batch processing failed")
            return None
            
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main dataset integrity verification function"""
    print("🚀 LISA-Llama4 Dataset Integrity Verification")
    print("=" * 70)
    print(f"Model: {config.MODEL_ID}")
    print("=" * 70)
    
    # Track verification results
    results = {
        "processor_init": False,
        "dual_stream": False,
        "batch_processing": False,
    }
    
    # Run verifications
    try:
        # 1. Processor initialization
        processor = verify_processor_initialization()
        if processor is not None:
            results["processor_init"] = True
            
            # 2. Dual stream processing
            if verify_dual_stream_processing(processor):
                results["dual_stream"] = True
                
                # 3. Batch processing
                if verify_batch_processing(processor):
                    results["batch_processing"] = True
        
    except KeyboardInterrupt:
        print("\n⚠️ Verification interrupted by user")
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 DATASET INTEGRITY VERIFICATION SUMMARY")
    print("=" * 70)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test.replace('_', ' ').title():<20}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    success_rate = passed_tests / total_tests * 100
    
    print("\n" + "=" * 70)
    print(f"🎯 Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 Dataset integrity verification successful!")
        return 0
    else:
        print("❌ Dataset integrity verification failed")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 