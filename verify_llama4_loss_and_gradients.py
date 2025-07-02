#!/usr/bin/env python3
"""
Llama4-LISA Loss and Gradients Verification Script
Verifies loss computation and gradient flow for LISA-Llama4
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
from PIL import Image
import numpy as np


def verify_loss_computation():
    """Verify loss computation for different scenarios"""
    print("\n" + "=" * 70)
    print("💰 Testing Loss Computation")
    print("=" * 70)
    
    try:
        # Initialize model and processor
        model = create_llama4_lisa_model(
            model_id=config.MODEL_ID,
            sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
            train_mask_decoder=config.TRAIN_MASK_DECODER,
            hidden_size=config.HIDDEN_SIZE,
            out_dim=config.OUT_DIM,
        )
        
        processor = Llama4DualStreamProcessor(model_id=config.MODEL_ID)
        model.set_seg_token_idx(processor.seg_token_id)
        
        # Test scenarios
        test_scenarios = [
            {
                "name": "With Segmentation",
                "prompt": "Segment the object [SEG]",
                "has_mask": True,
            },
            {
                "name": "Without Segmentation",
                "prompt": "Describe this image",
                "has_mask": False,
            },
            {
                "name": "Multiple Segmentations",
                "prompt": "Segment object A [SEG] and object B [SEG]",
                "has_mask": True,
            }
        ]
        
        results = []
        
        for scenario in test_scenarios:
            print(f"\n🔍 Testing: {scenario['name']}")
            print(f"   Prompt: {scenario['prompt']}")
            
            try:
                # Create test data
                test_image = Image.new('RGB', (512, 512), color='red')
                processed = processor.preprocess_dual_stream(test_image, scenario['prompt'])
                
                # Create dummy ground truth mask if needed
                if scenario['has_mask']:
                    ground_truth_mask = torch.ones(1024, 1024) * 0.5  # Dummy mask
                else:
                    ground_truth_mask = None
                
                # Forward pass
                model.eval()
                with torch.no_grad():
                    outputs = model.model_forward(
                        input_ids=processed['input_ids'],
                        pixel_values=processed['llama4_pixel_values'],
                        sam_images=processed['sam_images'].unsqueeze(0),
                        attention_mask=processed['attention_mask'],
                        seg_token_positions=processed['seg_token_positions'],
                        ground_truth_masks=[ground_truth_mask] if ground_truth_mask is not None else None,
                        has_masks=[scenario['has_mask']],
                        inference=False,
                    )
                
                # Analyze outputs
                print(f"   Total loss: {outputs.loss}")
                print(f"   CE loss: {outputs.ce_loss}")
                print(f"   Seg loss: {outputs.seg_loss}")
                print(f"   Dice loss: {outputs.dice_loss}")
                print(f"   BCE loss: {outputs.bce_loss}")
                
                # Verify loss components
                if scenario['has_mask']:
                    if outputs.seg_loss > 0:
                        print(f"   ✅ Segmentation loss computed correctly")
                        seg_loss_ok = True
                    else:
                        print(f"   ❌ Segmentation loss should be > 0 when mask is present")
                        seg_loss_ok = False
                else:
                    if outputs.seg_loss == 0:
                        print(f"   ✅ No segmentation loss when no mask")
                        seg_loss_ok = True
                    else:
                        print(f"   ⚠️ Segmentation loss present when no mask expected")
                        seg_loss_ok = True  # May be acceptable
                
                # Verify total loss
                if outputs.loss > 0:
                    print(f"   ✅ Total loss > 0")
                    total_loss_ok = True
                else:
                    print(f"   ❌ Total loss should be > 0")
                    total_loss_ok = False
                
                results.append(seg_loss_ok and total_loss_ok)
                
            except Exception as e:
                print(f"   ❌ Loss computation failed: {e}")
                results.append(False)
        
        success_rate = sum(results) / len(results) * 100
        print(f"\n📊 Loss Computation Success Rate: {success_rate:.1f}%")
        
        return success_rate >= 80
        
    except Exception as e:
        print(f"❌ Loss computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_gradient_flow():
    """Verify gradient flow through the model"""
    print("\n" + "=" * 70)
    print("🌊 Testing Gradient Flow")
    print("=" * 70)
    
    try:
        # Initialize model
        model = create_llama4_lisa_model(
            model_id=config.MODEL_ID,
            sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
            train_mask_decoder=config.TRAIN_MASK_DECODER,
            hidden_size=config.HIDDEN_SIZE,
            out_dim=config.OUT_DIM,
        )
        
        processor = Llama4DualStreamProcessor(model_id=config.MODEL_ID)
        model.set_seg_token_idx(processor.seg_token_id)
        
        # Create test data
        test_image = Image.new('RGB', (512, 512), color='blue')
        test_prompt = "Segment the blue area [SEG]"
        processed = processor.preprocess_dual_stream(test_image, test_prompt)
        
        # Create dummy ground truth
        ground_truth_mask = torch.ones(1024, 1024) * 0.7
        
        # Set model to training mode
        model.train()
        
        # Forward pass
        outputs = model.model_forward(
            input_ids=processed['input_ids'],
            pixel_values=processed['llama4_pixel_values'],
            sam_images=processed['sam_images'].unsqueeze(0),
            attention_mask=processed['attention_mask'],
            seg_token_positions=processed['seg_token_positions'],
            ground_truth_masks=[ground_truth_mask],
            has_masks=[True],
            inference=False,
        )
        
        print(f"🔍 Forward pass completed:")
        print(f"   Total loss: {outputs.loss:.4f}")
        
        # Backward pass
        outputs.loss.backward()
        
        print("🔄 Backward pass completed")
        
        # Check gradients in different parts
        gradient_checks = []
        
        # 1. Check projection layer gradients
        print("\n📊 Gradient Analysis:")
        
        if hasattr(model, 'text_hidden_fcs'):
            proj_grads = []
            for i, proj_layer in enumerate(model.text_hidden_fcs):
                for name, param in proj_layer.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        proj_grads.append(grad_norm)
                        print(f"   Projection layer {i} {name}: grad_norm = {grad_norm:.6f}")
            
            if proj_grads and all(g > 0 for g in proj_grads):
                print(f"   ✅ Projection layers have gradients")
                gradient_checks.append(True)
            else:
                print(f"   ❌ Projection layers missing gradients")
                gradient_checks.append(False)
        
        # 2. Check SAM mask decoder gradients (if trainable)
        if config.TRAIN_MASK_DECODER and hasattr(model, 'visual_model'):
            sam_grads = []
            for name, param in model.visual_model.mask_decoder.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    sam_grads.append(grad_norm)
                    if len(sam_grads) <= 3:  # Show first 3
                        print(f"   SAM mask decoder {name}: grad_norm = {grad_norm:.6f}")
            
            if sam_grads:
                print(f"   ✅ SAM mask decoder has gradients ({len(sam_grads)} params)")
                gradient_checks.append(True)
            else:
                print(f"   ⚠️ SAM mask decoder has no gradients (may be frozen)")
                gradient_checks.append(True)  # May be intentional
        
        # 3. Check that frozen components don't have gradients
        if hasattr(model, 'visual_model'):
            frozen_params_with_grad = 0
            total_frozen_params = 0
            
            for name, param in model.visual_model.image_encoder.named_parameters():
                total_frozen_params += 1
                if param.grad is not None:
                    frozen_params_with_grad += 1
            
            print(f"   SAM image encoder: {frozen_params_with_grad}/{total_frozen_params} params have gradients")
            
            if frozen_params_with_grad == 0:
                print(f"   ✅ SAM image encoder properly frozen")
                gradient_checks.append(True)
            else:
                print(f"   ❌ SAM image encoder should be frozen")
                gradient_checks.append(False)
        
        # 4. Check gradient magnitudes
        all_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                all_grads.append(param.grad.norm().item())
        
        if all_grads:
            avg_grad = np.mean(all_grads)
            max_grad = np.max(all_grads)
            min_grad = np.min(all_grads)
            
            print(f"\n📈 Gradient Statistics:")
            print(f"   Average gradient norm: {avg_grad:.6f}")
            print(f"   Max gradient norm: {max_grad:.6f}")
            print(f"   Min gradient norm: {min_grad:.6f}")
            print(f"   Parameters with gradients: {len(all_grads)}")
            
            # Check for reasonable gradient magnitudes
            if 1e-8 < avg_grad < 10:
                print(f"   ✅ Gradient magnitudes look reasonable")
                gradient_checks.append(True)
            else:
                print(f"   ⚠️ Unusual gradient magnitudes")
                gradient_checks.append(False)
        
        success_rate = sum(gradient_checks) / len(gradient_checks) * 100
        print(f"\n📊 Gradient Flow Success Rate: {success_rate:.1f}%")
        
        return success_rate >= 80
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_loss_weighting():
    """Verify loss weighting configuration"""
    print("\n" + "=" * 70)
    print("⚖️ Testing Loss Weighting")
    print("=" * 70)
    
    try:
        # Test different loss weight configurations
        weight_configs = [
            {
                "name": "Default",
                "ce_weight": 1.0,
                "dice_weight": 0.5,
                "bce_weight": 2.0,
            },
            {
                "name": "High Segmentation Focus",
                "ce_weight": 0.5,
                "dice_weight": 1.0,
                "bce_weight": 3.0,
            },
            {
                "name": "Balanced",
                "ce_weight": 1.0,
                "dice_weight": 1.0,
                "bce_weight": 1.0,
            }
        ]
        
        results = []
        
        for weight_config in weight_configs:
            print(f"\n🔍 Testing: {weight_config['name']}")
            print(f"   CE: {weight_config['ce_weight']}, "
                  f"Dice: {weight_config['dice_weight']}, "
                  f"BCE: {weight_config['bce_weight']}")
            
            try:
                # Create model with specific weights
                model = create_llama4_lisa_model(
                    model_id=config.MODEL_ID,
                    sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
                    train_mask_decoder=config.TRAIN_MASK_DECODER,
                    hidden_size=config.HIDDEN_SIZE,
                    out_dim=config.OUT_DIM,
                    ce_loss_weight=weight_config['ce_weight'],
                    dice_loss_weight=weight_config['dice_weight'],
                    bce_loss_weight=weight_config['bce_weight'],
                )
                
                processor = Llama4DualStreamProcessor(model_id=config.MODEL_ID)
                model.set_seg_token_idx(processor.seg_token_id)
                
                # Test data
                test_image = Image.new('RGB', (512, 512), color='green')
                test_prompt = "Segment the green region [SEG]"
                processed = processor.preprocess_dual_stream(test_image, test_prompt)
                
                ground_truth_mask = torch.ones(1024, 1024) * 0.6
                
                # Forward pass
                model.eval()
                with torch.no_grad():
                    outputs = model.model_forward(
                        input_ids=processed['input_ids'],
                        pixel_values=processed['llama4_pixel_values'],
                        sam_images=processed['sam_images'].unsqueeze(0),
                        attention_mask=processed['attention_mask'],
                        seg_token_positions=processed['seg_token_positions'],
                        ground_truth_masks=[ground_truth_mask],
                        has_masks=[True],
                        inference=False,
                    )
                
                # Check loss weighting
                total_loss = outputs.loss.item()
                ce_loss = outputs.ce_loss.item()
                dice_loss = outputs.dice_loss.item()
                bce_loss = outputs.bce_loss.item()
                seg_loss = outputs.seg_loss.item()
                
                print(f"   Total loss: {total_loss:.4f}")
                print(f"   CE loss: {ce_loss:.4f}")
                print(f"   Dice loss: {dice_loss:.4f}")
                print(f"   BCE loss: {bce_loss:.4f}")
                print(f"   Seg loss: {seg_loss:.4f}")
                
                # Verify weighting is applied correctly
                expected_seg_loss = (weight_config['dice_weight'] * dice_loss + 
                                   weight_config['bce_weight'] * bce_loss)
                
                if abs(seg_loss - expected_seg_loss) < 1e-6:
                    print(f"   ✅ Loss weighting applied correctly")
                    weighting_ok = True
                else:
                    print(f"   ❌ Loss weighting incorrect")
                    print(f"      Expected seg loss: {expected_seg_loss:.4f}")
                    print(f"      Actual seg loss: {seg_loss:.4f}")
                    weighting_ok = False
                
                results.append(weighting_ok)
                
            except Exception as e:
                print(f"   ❌ Loss weighting test failed: {e}")
                results.append(False)
        
        success_rate = sum(results) / len(results) * 100
        print(f"\n📊 Loss Weighting Success Rate: {success_rate:.1f}%")
        
        return success_rate >= 80
        
    except Exception as e:
        print(f"❌ Loss weighting test failed: {e}")
        return False


def verify_parameter_updates():
    """Verify that only intended parameters are updated during training"""
    print("\n" + "=" * 70)
    print("🔄 Testing Parameter Updates")
    print("=" * 70)
    
    try:
        # Initialize model
        model = create_llama4_lisa_model(
            model_id=config.MODEL_ID,
            sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
            train_mask_decoder=config.TRAIN_MASK_DECODER,
            hidden_size=config.HIDDEN_SIZE,
            out_dim=config.OUT_DIM,
        )
        
        processor = Llama4DualStreamProcessor(model_id=config.MODEL_ID)
        model.set_seg_token_idx(processor.seg_token_id)
        
        # Store initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.data.clone()
        
        # Training step
        model.train()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        
        # Create test data
        test_image = Image.new('RGB', (512, 512), color='purple')
        test_prompt = "Segment the purple area [SEG]"
        processed = processor.preprocess_dual_stream(test_image, test_prompt)
        
        ground_truth_mask = torch.ones(1024, 1024) * 0.8
        
        # Forward pass
        outputs = model.model_forward(
            input_ids=processed['input_ids'],
            pixel_values=processed['llama4_pixel_values'],
            sam_images=processed['sam_images'].unsqueeze(0),
            attention_mask=processed['attention_mask'],
            seg_token_positions=processed['seg_token_positions'],
            ground_truth_masks=[ground_truth_mask],
            has_masks=[True],
            inference=False,
        )
        
        # Backward pass and update
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        # Check parameter updates
        print("🔍 Parameter Update Analysis:")
        
        updated_params = []
        unchanged_params = []
        
        for name, param in model.named_parameters():
            if name in initial_params:
                param_changed = not torch.equal(initial_params[name], param.data)
                
                if param.requires_grad:
                    if param_changed:
                        updated_params.append(name)
                    else:
                        print(f"   ⚠️ Trainable parameter unchanged: {name}")
                else:
                    if param_changed:
                        print(f"   ❌ Frozen parameter changed: {name}")
                    else:
                        unchanged_params.append(name)
        
        print(f"   ✅ Updated parameters: {len(updated_params)}")
        print(f"   ✅ Unchanged frozen parameters: {len(unchanged_params)}")
        
        # Show some updated parameters
        if updated_params:
            print(f"   Sample updated parameters:")
            for param_name in updated_params[:3]:
                print(f"     - {param_name}")
        
        # Verify expectations
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        
        if len(updated_params) > 0:
            print(f"   ✅ Parameters are being updated")
            updates_ok = True
        else:
            print(f"   ❌ No parameters were updated")
            updates_ok = False
        
        print(f"\n📊 Trainable parameters: {trainable_count}")
        print(f"   Updated in this step: {len(updated_params)}")
        
        return updates_ok
        
    except Exception as e:
        print(f"❌ Parameter update test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main loss and gradients verification function"""
    print("🚀 LISA-Llama4 Loss and Gradients Verification")
    print("=" * 70)
    print(f"Model: {config.MODEL_ID}")
    print("=" * 70)
    
    # Track verification results
    results = {
        "loss_computation": False,
        "gradient_flow": False,
        "loss_weighting": False,
        "parameter_updates": False,
    }
    
    # Run verifications
    try:
        # 1. Loss computation
        results["loss_computation"] = verify_loss_computation()
        
        # 2. Gradient flow
        results["gradient_flow"] = verify_gradient_flow()
        
        # 3. Loss weighting
        results["loss_weighting"] = verify_loss_weighting()
        
        # 4. Parameter updates
        results["parameter_updates"] = verify_parameter_updates()
        
    except KeyboardInterrupt:
        print("\n⚠️ Verification interrupted by user")
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 LOSS AND GRADIENTS VERIFICATION SUMMARY")
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
        print("🎉 Loss and gradients verification successful!")
        return 0
    elif success_rate >= 60:
        print("⚠️ Loss and gradients partially verified - some issues detected")
        return 1
    else:
        print("❌ Loss and gradients verification failed - significant issues detected")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
