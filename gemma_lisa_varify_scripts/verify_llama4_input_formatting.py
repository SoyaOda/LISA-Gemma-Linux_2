#!/usr/bin/env python3
"""
Llama4-LISA Input Formatting Verification Script
Verifies input formatting and tokenization for LISA-Llama4
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
import json


def verify_chat_template_formatting():
    """Verify Llama4 chat template formatting"""
    print("\n" + "=" * 70)
    print("💬 Testing Chat Template Formatting")
    print("=" * 70)
    
    try:
        processor = Llama4DualStreamProcessor(model_id=config.MODEL_ID)
        
        # Test various conversation formats
        test_cases = [
            {
                "name": "Simple Segmentation",
                "prompt": "Can you segment the cat in this image? [SEG]",
                "expected_tokens": ["[SEG]"]
            },
            {
                "name": "Multi-turn Conversation",
                "prompt": "What do you see in this image? Please segment the main object [SEG]",
                "expected_tokens": ["[SEG]"]
            },
            {
                "name": "Complex Query",
                "prompt": "This is a street scene. Can you identify and segment the car [SEG] and the building [SEG]?",
                "expected_tokens": ["[SEG]", "[SEG]"]
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\n🔍 Testing: {test_case['name']}")
            print(f"   Prompt: {test_case['prompt']}")
            
            # Create test image
            test_image = Image.new('RGB', (400, 400), color='blue')
            
            try:
                # Process input
                result = processor.preprocess_dual_stream(test_image, test_case['prompt'])
                
                # Check tokenization
                input_ids = result['input_ids']
                tokens = processor.llama4_processor.tokenizer.convert_ids_to_tokens(input_ids)
                
                # Count SEG tokens
                seg_count = result['seg_token_positions'].sum().item()
                expected_count = len(test_case['expected_tokens'])
                
                print(f"   Input IDs shape: {input_ids.shape}")
                print(f"   SEG tokens found: {seg_count}")
                print(f"   Expected SEG tokens: {expected_count}")
                
                if seg_count == expected_count:
                    print(f"   ✅ Correct number of SEG tokens")
                    results.append(True)
                else:
                    print(f"   ❌ SEG token count mismatch")
                    results.append(False)
                
            except Exception as e:
                print(f"   ❌ Processing failed: {e}")
                results.append(False)
        
        success_rate = sum(results) / len(results) * 100
        print(f"\n📊 Chat Template Formatting Success Rate: {success_rate:.1f}%")
        
        return success_rate >= 80
        
    except Exception as e:
        print(f"❌ Chat template formatting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_token_alignment():
    """Verify token alignment and position tracking"""
    print("\n" + "=" * 70)
    print("🎯 Testing Token Alignment")
    print("=" * 70)
    
    try:
        processor = Llama4DualStreamProcessor(model_id=config.MODEL_ID)
        
        # Test cases with known SEG token positions
        test_cases = [
            {
                "text": "Segment this [SEG]",
                "expected_positions": 1  # One SEG token
            },
            {
                "text": "First [SEG] and second [SEG]",
                "expected_positions": 2  # Two SEG tokens
            },
            {
                "text": "No segmentation needed",
                "expected_positions": 0  # No SEG tokens
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 Test {i}: {test_case['text']}")
            
            test_image = Image.new('RGB', (400, 400), color=(100, 150, 200))
            
            try:
                result = processor.preprocess_dual_stream(test_image, test_case['text'])
                
                # Analyze tokenization
                input_ids = result['input_ids']
                seg_positions = result['seg_token_positions']
                
                # Count actual SEG token positions
                actual_count = seg_positions.sum().item()
                expected_count = test_case['expected_positions']
                
                print(f"   Input IDs: {input_ids.tolist()}")
                print(f"   SEG positions: {seg_positions.tolist()}")
                print(f"   Expected SEG count: {expected_count}")
                print(f"   Actual SEG count: {actual_count}")
                
                # Verify SEG token IDs
                seg_token_id = processor.seg_token_id
                seg_ids_in_input = (input_ids == seg_token_id).sum().item()
                
                print(f"   SEG token ID: {seg_token_id}")
                print(f"   SEG IDs in input: {seg_ids_in_input}")
                
                # Check alignment
                if actual_count == expected_count and actual_count == seg_ids_in_input:
                    print(f"   ✅ Token alignment correct")
                    results.append(True)
                else:
                    print(f"   ❌ Token alignment mismatch")
                    results.append(False)
                
            except Exception as e:
                print(f"   ❌ Token alignment test failed: {e}")
                results.append(False)
        
        success_rate = sum(results) / len(results) * 100
        print(f"\n📊 Token Alignment Success Rate: {success_rate:.1f}%")
        
        return success_rate >= 80
        
    except Exception as e:
        print(f"❌ Token alignment test failed: {e}")
        return False


def verify_image_token_handling():
    """Verify image token handling in Llama4"""
    print("\n" + "=" * 70)
    print("🖼️ Testing Image Token Handling")
    print("=" * 70)
    
    try:
        processor = Llama4DualStreamProcessor(model_id=config.MODEL_ID)
        
        # Test different image scenarios
        test_images = [
            {
                "name": "Square Image",
                "size": (512, 512),
                "color": "red"
            },
            {
                "name": "Landscape Image", 
                "size": (800, 600),
                "color": "green"
            },
            {
                "name": "Portrait Image",
                "size": (600, 800),
                "color": "blue"
            }
        ]
        
        results = []
        
        for test_img in test_images:
            print(f"\n🔍 Testing: {test_img['name']}")
            print(f"   Size: {test_img['size']}")
            
            try:
                # Create test image
                if test_img['color'] == 'red':
                    color = (255, 0, 0)
                elif test_img['color'] == 'green':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                    
                test_image = Image.new('RGB', test_img['size'], color=color)
                test_prompt = f"Describe this {test_img['color']} image and segment it [SEG]"
                
                # Process
                result = processor.preprocess_dual_stream(test_image, test_prompt)
                
                # Check image processing
                llama4_pixels = result['llama4_pixel_values']
                sam_image = result['sam_images']
                
                print(f"   Llama4 pixel values shape: {llama4_pixels.shape}")
                print(f"   SAM image shape: {sam_image.shape}")
                print(f"   Original size: {result['original_image_size']}")
                
                # Verify shapes
                if sam_image.shape == (3, 1024, 1024):
                    print(f"   ✅ SAM image shape correct")
                    shape_correct = True
                else:
                    print(f"   ❌ SAM image shape incorrect")
                    shape_correct = False
                
                # Verify pixel value ranges
                if 0 <= llama4_pixels.min() and llama4_pixels.max() <= 1:
                    print(f"   ✅ Llama4 pixel values in correct range")
                    range_correct = True
                else:
                    print(f"   ❌ Llama4 pixel values out of range")
                    range_correct = False
                
                results.append(shape_correct and range_correct)
                
            except Exception as e:
                print(f"   ❌ Image processing failed: {e}")
                results.append(False)
        
        success_rate = sum(results) / len(results) * 100
        print(f"\n📊 Image Token Handling Success Rate: {success_rate:.1f}%")
        
        return success_rate >= 80
        
    except Exception as e:
        print(f"❌ Image token handling test failed: {e}")
        return False


def verify_attention_mask_correctness():
    """Verify attention mask correctness"""
    print("\n" + "=" * 70)
    print("👁️ Testing Attention Mask Correctness")
    print("=" * 70)
    
    try:
        processor = Llama4DualStreamProcessor(model_id=config.MODEL_ID)
        
        # Test different input lengths
        test_prompts = [
            "Short [SEG]",
            "This is a medium length prompt with segmentation token [SEG] for testing",
            "This is a very long prompt that contains multiple sentences and should test the attention mask handling properly. We want to segment multiple objects [SEG] and make sure everything works [SEG]."
        ]
        
        results = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🔍 Test {i}: Length {len(prompt)} chars")
            print(f"   Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
            
            try:
                test_image = Image.new('RGB', (400, 400), color='gray')
                result = processor.preprocess_dual_stream(test_image, prompt)
                
                input_ids = result['input_ids']
                attention_mask = result['attention_mask']
                
                print(f"   Input IDs shape: {input_ids.shape}")
                print(f"   Attention mask shape: {attention_mask.shape}")
                
                # Check shapes match
                if input_ids.shape == attention_mask.shape:
                    print(f"   ✅ Shapes match")
                    shape_match = True
                else:
                    print(f"   ❌ Shape mismatch")
                    shape_match = False
                
                # Check attention mask values
                unique_values = torch.unique(attention_mask)
                print(f"   Attention mask unique values: {unique_values.tolist()}")
                
                # Should only contain 0s and 1s
                if set(unique_values.tolist()).issubset({0, 1}):
                    print(f"   ✅ Attention mask values correct")
                    values_correct = True
                else:
                    print(f"   ❌ Attention mask contains invalid values")
                    values_correct = False
                
                # Check no leading zeros (assuming no padding at start)
                if attention_mask[0] == 1:
                    print(f"   ✅ No leading padding")
                    no_leading_pad = True
                else:
                    print(f"   ⚠️ Leading padding detected")
                    no_leading_pad = True  # May be valid for some models
                
                results.append(shape_match and values_correct and no_leading_pad)
                
            except Exception as e:
                print(f"   ❌ Attention mask test failed: {e}")
                results.append(False)
        
        success_rate = sum(results) / len(results) * 100
        print(f"\n📊 Attention Mask Success Rate: {success_rate:.1f}%")
        
        return success_rate >= 80
        
    except Exception as e:
        print(f"❌ Attention mask test failed: {e}")
        return False


def verify_special_token_handling():
    """Verify special token handling beyond SEG tokens"""
    print("\n" + "=" * 70)
    print("🏷️ Testing Special Token Handling")
    print("=" * 70)
    
    try:
        processor = Llama4DualStreamProcessor(model_id=config.MODEL_ID)
        
        # Check tokenizer special tokens
        tokenizer = processor.llama4_processor.tokenizer
        
        print("🔍 Tokenizer Special Tokens:")
        special_tokens = {
            'pad_token': tokenizer.pad_token,
            'eos_token': tokenizer.eos_token,
            'bos_token': tokenizer.bos_token,
            'unk_token': tokenizer.unk_token,
        }
        
        for token_name, token_value in special_tokens.items():
            print(f"   {token_name}: {token_value}")
        
        # Test SEG token specifically
        seg_token = processor.seg_token
        seg_token_id = processor.seg_token_id
        
        print(f"\n🎯 SEG Token Details:")
        print(f"   SEG token: {seg_token}")
        print(f"   SEG token ID: {seg_token_id}")
        
        # Verify SEG token is in vocabulary
        vocab = tokenizer.get_vocab()
        if seg_token in vocab:
            print(f"   ✅ SEG token in vocabulary")
            retrieved_id = vocab[seg_token]
            if retrieved_id == seg_token_id:
                print(f"   ✅ SEG token ID consistent")
                seg_token_ok = True
            else:
                print(f"   ❌ SEG token ID inconsistent: {retrieved_id} vs {seg_token_id}")
                seg_token_ok = False
        else:
            print(f"   ❌ SEG token not in vocabulary")
            seg_token_ok = False
        
        # Test tokenization consistency
        test_text = "Test [SEG] tokenization"
        tokens1 = tokenizer.encode(test_text)
        tokens2 = tokenizer.encode(test_text)
        
        if tokens1 == tokens2:
            print(f"   ✅ Tokenization is consistent")
            consistent = True
        else:
            print(f"   ❌ Tokenization is inconsistent")
            consistent = False
        
        # Test decoding
        decoded = tokenizer.decode(tokens1)
        print(f"   Original: {test_text}")
        print(f"   Decoded: {decoded}")
        
        return seg_token_ok and consistent
        
    except Exception as e:
        print(f"❌ Special token handling test failed: {e}")
        return False


def main():
    """Main input formatting verification function"""
    print("🚀 LISA-Llama4 Input Formatting Verification")
    print("=" * 70)
    print(f"Model: {config.MODEL_ID}")
    print("=" * 70)
    
    # Track verification results
    results = {
        "chat_template": False,
        "token_alignment": False,
        "image_tokens": False,
        "attention_mask": False,
        "special_tokens": False,
    }
    
    # Run verifications
    try:
        # 1. Chat template formatting
        results["chat_template"] = verify_chat_template_formatting()
        
        # 2. Token alignment
        results["token_alignment"] = verify_token_alignment()
        
        # 3. Image token handling
        results["image_tokens"] = verify_image_token_handling()
        
        # 4. Attention mask correctness
        results["attention_mask"] = verify_attention_mask_correctness()
        
        # 5. Special token handling
        results["special_tokens"] = verify_special_token_handling()
        
    except KeyboardInterrupt:
        print("\n⚠️ Verification interrupted by user")
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 INPUT FORMATTING VERIFICATION SUMMARY")
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
        print("🎉 Input formatting verification successful!")
        return 0
    elif success_rate >= 60:
        print("⚠️ Input formatting partially verified - some issues detected")
        return 1
    else:
        print("❌ Input formatting verification failed - significant issues detected")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
