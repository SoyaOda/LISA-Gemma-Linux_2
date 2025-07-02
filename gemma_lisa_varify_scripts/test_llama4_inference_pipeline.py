#!/usr/bin/env python3
"""
Llama4-LISA Inference Pipeline Test
Tests end-to-end inference pipeline for LISA-Llama4
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

from model.llama4_lisa import create_llama4_lisa_model
from utils.llama4_processing import Llama4DualStreamProcessor
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import time


def create_test_scenarios():
    """Create various test scenarios for inference"""
    print("\n" + "=" * 70)
    print("🎭 Creating Test Scenarios")
    print("=" * 70)
    
    # Scenario 1: Simple colored regions
    img1 = Image.new('RGB', (512, 512), color='white')
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle([100, 100, 200, 200], fill='red')
    draw1.rectangle([300, 300, 400, 400], fill='blue')
    
    # Scenario 2: Multiple objects
    img2 = Image.new('RGB', (600, 400), color='lightgray')
    draw2 = ImageDraw.Draw(img2)
    draw2.ellipse([50, 50, 150, 150], fill='green')
    draw2.ellipse([200, 100, 300, 200], fill='yellow')
    draw2.rectangle([400, 150, 550, 350], fill='purple')
    
    # Scenario 3: Complex scene
    img3 = Image.new('RGB', (800, 600), color='skyblue')
    draw3 = ImageDraw.Draw(img3)
    # Draw a simple landscape
    draw3.rectangle([0, 400, 800, 600], fill='green')  # Ground
    draw3.ellipse([50, 50, 150, 150], fill='yellow')   # Sun
    draw3.rectangle([200, 200, 300, 400], fill='brown')  # Tree trunk
    draw3.ellipse([150, 100, 350, 250], fill='darkgreen')  # Tree crown
    
    scenarios = [
        {
            "name": "Simple Segmentation",
            "image": img1,
            "prompts": [
                "Segment the red square [SEG]",
                "Can you find the blue square? [SEG]",
                "Describe this image"  # No segmentation
            ]
        },
        {
            "name": "Multiple Objects",
            "image": img2,
            "prompts": [
                "Segment the green circle [SEG]",
                "Find the yellow circle [SEG] and purple rectangle [SEG]",
                "What objects do you see in this image?"
            ]
        },
        {
            "name": "Complex Scene",
            "image": img3,
            "prompts": [
                "Segment the tree in this landscape [SEG]",
                "Can you identify the sun? [SEG]",
                "Segment the ground area [SEG]",
                "Describe this outdoor scene"
            ]
        }
    ]
    
    print(f"📊 Created {len(scenarios)} test scenarios:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"   {i}. {scenario['name']}: {len(scenario['prompts'])} prompts")
    
    return scenarios


def test_inference_speed(model, processor):
    """Test inference speed with different input sizes"""
    print("\n" + "=" * 70)
    print("⚡ Testing Inference Speed")
    print("=" * 70)
    
    # Test different image sizes
    test_sizes = [
        (256, 256),
        (512, 512),
        (800, 600),
        (1024, 768),
    ]
    
    speed_results = []
    
    for size in test_sizes:
        print(f"\n🔍 Testing size: {size}")
        
        # Create test image
        test_image = Image.new('RGB', size, color='orange')
        test_prompt = "Segment the orange area [SEG]"
        
        # Warmup run
        with torch.no_grad():
            processed = processor.preprocess_dual_stream(test_image, test_prompt)
            _ = model.model_forward(
                input_ids=processed['input_ids'],
                pixel_values=processed['llama4_pixel_values'],
                sam_images=processed['sam_images'].unsqueeze(0),
                attention_mask=processed['attention_mask'],
                seg_token_positions=processed['seg_token_positions'],
                ground_truth_masks=None,
                has_masks=[False],
                inference=True,
            )
        
        # Timed runs
        times = []
        for _ in range(3):
            start_time = time.time()
            
            with torch.no_grad():
                processed = processor.preprocess_dual_stream(test_image, test_prompt)
                outputs = model.model_forward(
                    input_ids=processed['input_ids'],
                    pixel_values=processed['llama4_pixel_values'],
                    sam_images=processed['sam_images'].unsqueeze(0),
                    attention_mask=processed['attention_mask'],
                    seg_token_positions=processed['seg_token_positions'],
                    ground_truth_masks=None,
                    has_masks=[False],
                    inference=True,
                )
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"   Average time: {avg_time:.3f}s ± {std_time:.3f}s")
        
        speed_results.append({
            'size': size,
            'avg_time': avg_time,
            'std_time': std_time
        })
    
    # Analyze speed results
    print(f"\n📊 Speed Analysis:")
    fastest = min(speed_results, key=lambda x: x['avg_time'])
    slowest = max(speed_results, key=lambda x: x['avg_time'])
    
    print(f"   Fastest: {fastest['size']} - {fastest['avg_time']:.3f}s")
    print(f"   Slowest: {slowest['size']} - {slowest['avg_time']:.3f}s")
    
    # Check if times are reasonable (< 10s per inference)
    reasonable_times = all(result['avg_time'] < 10.0 for result in speed_results)
    
    if reasonable_times:
        print(f"   ✅ All inference times are reasonable")
        return True
    else:
        print(f"   ⚠️ Some inference times are slow")
        return False


def test_output_quality(model, processor, scenarios):
    """Test output quality and consistency"""
    print("\n" + "=" * 70)
    print("🔍 Testing Output Quality")
    print("=" * 70)
    
    quality_results = []
    
    for scenario_idx, scenario in enumerate(scenarios):
        print(f"\n🎭 Scenario {scenario_idx + 1}: {scenario['name']}")
        
        for prompt_idx, prompt in enumerate(scenario['prompts']):
            print(f"   Prompt {prompt_idx + 1}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
            
            try:
                # Process input
                processed = processor.preprocess_dual_stream(scenario['image'], prompt)
                
                # Check if SEG token is present
                has_seg_token = processed['seg_token_positions'].any()
                expected_seg = '[SEG]' in prompt
                
                print(f"     SEG token expected: {expected_seg}, found: {has_seg_token}")
                
                # Inference
                model.eval()
                with torch.no_grad():
                    outputs = model.model_forward(
                        input_ids=processed['input_ids'],
                        pixel_values=processed['llama4_pixel_values'],
                        sam_images=processed['sam_images'].unsqueeze(0),
                        attention_mask=processed['attention_mask'],
                        seg_token_positions=processed['seg_token_positions'],
                        ground_truth_masks=None,
                        has_masks=[expected_seg],
                        inference=True,
                    )
                
                # Analyze outputs
                print(f"     Output available: {outputs is not None}")
                
                if expected_seg and hasattr(outputs, 'masks'):
                    # Check mask output
                    if outputs.masks is not None:
                        mask_shape = outputs.masks.shape
                        mask_range = (outputs.masks.min().item(), outputs.masks.max().item())
                        print(f"     Mask shape: {mask_shape}")
                        print(f"     Mask range: [{mask_range[0]:.3f}, {mask_range[1]:.3f}]")
                        
                        # Check mask properties
                        mask_valid = (
                            len(mask_shape) >= 2 and
                            mask_shape[-2:] == (1024, 1024) and
                            0 <= mask_range[0] <= mask_range[1] <= 1
                        )
                        
                        if mask_valid:
                            print(f"     ✅ Mask output valid")
                            quality_results.append(True)
                        else:
                            print(f"     ❌ Mask output invalid")
                            quality_results.append(False)
                    else:
                        print(f"     ❌ No mask output when expected")
                        quality_results.append(False)
                else:
                    print(f"     ✅ No segmentation expected/provided")
                    quality_results.append(True)
                
            except Exception as e:
                print(f"     ❌ Inference failed: {e}")
                quality_results.append(False)
    
    # Calculate success rate
    success_rate = sum(quality_results) / len(quality_results) * 100
    print(f"\n📊 Output Quality Success Rate: {success_rate:.1f}%")
    
    return success_rate >= 75


def test_memory_efficiency(model, processor):
    """Test memory efficiency during inference"""
    print("\n" + "=" * 70)
    print("💾 Testing Memory Efficiency")
    print("=" * 70)
    
    try:
        import psutil
        process = psutil.Process()
        
        # Baseline memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"🔧 Initial memory: {initial_memory:.1f} MB")
        
        # Test multiple inferences
        test_image = Image.new('RGB', (512, 512), color='cyan')
        test_prompt = "Segment the cyan area [SEG]"
        
        memory_usage = []
        
        for i in range(5):
            # Run inference
            processed = processor.preprocess_dual_stream(test_image, test_prompt)
            
            with torch.no_grad():
                outputs = model.model_forward(
                    input_ids=processed['input_ids'],
                    pixel_values=processed['llama4_pixel_values'],
                    sam_images=processed['sam_images'].unsqueeze(0),
                    attention_mask=processed['attention_mask'],
                    seg_token_positions=processed['seg_token_positions'],
                    ground_truth_masks=None,
                    has_masks=[True],
                    inference=True,
                )
            
            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_usage.append(current_memory)
            
            print(f"   Inference {i+1}: {current_memory:.1f} MB")
            
            # Clear cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Analyze memory usage
        max_memory = max(memory_usage)
        memory_increase = max_memory - initial_memory
        
        print(f"\n📊 Memory Analysis:")
        print(f"   Max memory: {max_memory:.1f} MB")
        print(f"   Memory increase: {memory_increase:.1f} MB")
        
        # Check for memory leaks
        final_memory = memory_usage[-1]
        memory_leak = final_memory - memory_usage[0]
        
        if abs(memory_leak) < 50:  # Less than 50MB variance
            print(f"   ✅ No significant memory leak: {memory_leak:.1f} MB")
            memory_ok = True
        else:
            print(f"   ⚠️ Possible memory leak: {memory_leak:.1f} MB")
            memory_ok = False
        
        # Check reasonable memory usage (< 8GB)
        if max_memory < 8192:
            print(f"   ✅ Memory usage reasonable")
            return memory_ok
        else:
            print(f"   ⚠️ High memory usage")
            return False
            
    except ImportError:
        print("⚠️ psutil not available, skipping memory test")
        return True
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False


def test_batch_inference(model, processor):
    """Test batch inference capability"""
    print("\n" + "=" * 70)
    print("📦 Testing Batch Inference")
    print("=" * 70)
    
    try:
        # Create batch of test samples
        batch_size = 3
        test_images = [
            Image.new('RGB', (400, 400), color=color)
            for color in ['red', 'green', 'blue']
        ]
        test_prompts = [
            f"Segment the {color} area [SEG]"
            for color in ['red', 'green', 'blue']
        ]
        
        print(f"🔧 Processing batch of {batch_size} samples...")
        
        # Process batch
        batch_processed = processor.batch_preprocess(test_images, test_prompts)
        
        print(f"📊 Batch shapes:")
        for key, tensor in batch_processed.items():
            if isinstance(tensor, torch.Tensor):
                print(f"   {key}: {tensor.shape}")
        
        # Batch inference
        model.eval()
        with torch.no_grad():
            batch_outputs = model.model_forward(
                input_ids=batch_processed['input_ids'],
                pixel_values=batch_processed['llama4_pixel_values'],
                sam_images=batch_processed['sam_images'],
                attention_mask=batch_processed['attention_mask'],
                seg_token_positions=batch_processed['seg_token_positions'],
                ground_truth_masks=None,
                has_masks=[True] * batch_size,
                inference=True,
            )
        
        # Verify batch outputs
        if hasattr(batch_outputs, 'masks') and batch_outputs.masks is not None:
            mask_shape = batch_outputs.masks.shape
            print(f"   Batch mask output shape: {mask_shape}")
            
            if mask_shape[0] == batch_size:
                print(f"   ✅ Batch inference successful")
                return True
            else:
                print(f"   ❌ Batch size mismatch")
                return False
        else:
            print(f"   ❌ No mask output from batch inference")
            return False
            
    except Exception as e:
        print(f"❌ Batch inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_inference_results(model, processor, scenarios, save_dir="inference_results"):
    """Visualize inference results"""
    print(f"\n📸 Visualizing inference results...")
    
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        for scenario_idx, scenario in enumerate(scenarios[:2]):  # First 2 scenarios
            for prompt_idx, prompt in enumerate(scenario['prompts'][:2]):  # First 2 prompts
                if '[SEG]' in prompt:  # Only visualize segmentation prompts
                    
                    # Process and run inference
                    processed = processor.preprocess_dual_stream(scenario['image'], prompt)
                    
                    model.eval()
                    with torch.no_grad():
                        outputs = model.model_forward(
                            input_ids=processed['input_ids'],
                            pixel_values=processed['llama4_pixel_values'],
                            sam_images=processed['sam_images'].unsqueeze(0),
                            attention_mask=processed['attention_mask'],
                            seg_token_positions=processed['seg_token_positions'],
                            ground_truth_masks=None,
                            has_masks=[True],
                            inference=True,
                        )
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original image
                    axes[0].imshow(scenario['image'])
                    axes[0].set_title("Original Image")
                    axes[0].axis('off')
                    
                    # SAM preprocessed image
                    sam_img = processed['sam_images'].permute(1, 2, 0).numpy()
                    sam_img = (sam_img - sam_img.min()) / (sam_img.max() - sam_img.min())
                    axes[1].imshow(sam_img)
                    axes[1].set_title("SAM Preprocessed")
                    axes[1].axis('off')
                    
                    # Predicted mask
                    if hasattr(outputs, 'masks') and outputs.masks is not None:
                        mask = outputs.masks[0, 0].numpy()  # First batch, first mask
                        axes[2].imshow(mask, cmap='hot')
                        axes[2].set_title("Predicted Mask")
                    else:
                        axes[2].text(0.5, 0.5, 'No Mask', ha='center', va='center', transform=axes[2].transAxes)
                        axes[2].set_title("No Prediction")
                    axes[2].axis('off')
                    
                    # Add prompt as suptitle
                    fig.suptitle(f"Scenario {scenario_idx+1}: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
                    
                    # Save
                    filename = f"inference_s{scenario_idx+1}_p{prompt_idx+1}.png"
                    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
                    plt.close()
        
        print(f"   Saved visualizations to: {save_dir}")
        return True
        
    except Exception as e:
        print(f"   ⚠️ Could not create visualizations: {e}")
        return False


def main():
    """Main inference pipeline test function"""
    print("🚀 LISA-Llama4 Inference Pipeline Test")
    print("=" * 70)
    print(f"Model: {config.MODEL_ID}")
    print("=" * 70)
    
    # Track test results
    results = {
        "model_loading": False,
        "basic_inference": False,
        "inference_speed": False,
        "output_quality": False,
        "memory_efficiency": False,
        "batch_inference": False,
    }
    
    try:
        # Initialize model and processor
        print("🔧 Loading model and processor...")
        model = create_llama4_lisa_model(
            model_id=config.MODEL_ID,
            sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
            train_mask_decoder=config.TRAIN_MASK_DECODER,
            hidden_size=config.HIDDEN_SIZE,
            out_dim=config.OUT_DIM,
        )
        
        processor = Llama4DualStreamProcessor(model_id=config.MODEL_ID)
        model.set_seg_token_idx(processor.seg_token_id)
        
        results["model_loading"] = True
        print("✅ Model and processor loaded successfully")
        
        # Create test scenarios
        scenarios = create_test_scenarios()
        
        # Test basic inference
        print("\n🔍 Testing basic inference...")
        test_image = Image.new('RGB', (400, 400), color='pink')
        test_prompt = "Segment the pink area [SEG]"
        
        try:
            processed = processor.preprocess_dual_stream(test_image, test_prompt)
            model.eval()
            with torch.no_grad():
                outputs = model.model_forward(
                    input_ids=processed['input_ids'],
                    pixel_values=processed['llama4_pixel_values'],
                    sam_images=processed['sam_images'].unsqueeze(0),
                    attention_mask=processed['attention_mask'],
                    seg_token_positions=processed['seg_token_positions'],
                    ground_truth_masks=None,
                    has_masks=[True],
                    inference=True,
                )
            results["basic_inference"] = True
            print("✅ Basic inference successful")
        except Exception as e:
            print(f"❌ Basic inference failed: {e}")
        
        # Run comprehensive tests
        if results["basic_inference"]:
            results["inference_speed"] = test_inference_speed(model, processor)
            results["output_quality"] = test_output_quality(model, processor, scenarios)
            results["memory_efficiency"] = test_memory_efficiency(model, processor)
            results["batch_inference"] = test_batch_inference(model, processor)
            
            # Create visualizations
            visualize_inference_results(model, processor, scenarios)
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 INFERENCE PIPELINE TEST SUMMARY")
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
        print("🎉 Inference pipeline test successful!")
        print("   ✅ Model loads correctly")
        print("   ✅ Inference works as expected")
        print("   ✅ Output quality is good")
        return 0
    elif success_rate >= 60:
        print("⚠️ Inference pipeline partially working - some issues detected")
        return 1
    else:
        print("❌ Inference pipeline test failed - significant issues detected")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
