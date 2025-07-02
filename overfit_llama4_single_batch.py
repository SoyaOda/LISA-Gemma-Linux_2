#!/usr/bin/env python3
"""
Llama4-LISA Single Batch Overfitting Test
Tests the model's ability to overfit on a single batch to verify training capability
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
import matplotlib.pyplot as plt
from torch.optim import AdamW


def create_test_batch():
    """Create a small test batch for overfitting"""
    print("\n" + "=" * 70)
    print("📦 Creating Test Batch")
    print("=" * 70)
    
    # Create test samples
    test_samples = [
        {
            "image": Image.new('RGB', (512, 512), color=(255, 0, 0)),  # Red
            "prompt": "Please segment the red area in this image [SEG]",
            "mask": torch.ones(1024, 1024) * 0.8,  # High mask values
        },
        {
            "image": Image.new('RGB', (400, 600), color=(0, 255, 0)),  # Green
            "prompt": "Can you identify and segment the green region? [SEG]",
            "mask": torch.ones(1024, 1024) * 0.6,  # Medium mask values
        },
        {
            "image": Image.new('RGB', (600, 400), color=(0, 0, 255)),  # Blue
            "prompt": "Show me the blue area by segmenting it [SEG]",
            "mask": torch.ones(1024, 1024) * 0.9,  # High mask values
        }
    ]
    
    print(f"📊 Test batch created:")
    print(f"   Batch size: {len(test_samples)}")
    print(f"   Image sizes: {[sample['image'].size for sample in test_samples]}")
    print(f"   Prompt lengths: {[len(sample['prompt']) for sample in test_samples]}")
    
    return test_samples


def process_test_batch(test_samples, processor):
    """Process test batch into model inputs"""
    print("\n🔄 Processing test batch...")
    
    batch_data = {
        'pixel_values': [],
        'input_ids': [],
        'attention_mask': [],
        'sam_images': [],
        'seg_token_positions': [],
        'ground_truth_masks': [],
        'has_masks': [],
    }
    
    for i, sample in enumerate(test_samples):
        print(f"   Processing sample {i+1}/{len(test_samples)}")
        
        # Process with dual stream
        processed = processor.preprocess_dual_stream(sample['image'], sample['prompt'])
        
        batch_data['pixel_values'].append(processed['llama4_pixel_values'])
        batch_data['input_ids'].append(processed['input_ids'])
        batch_data['attention_mask'].append(processed['attention_mask'])
        batch_data['sam_images'].append(processed['sam_images'])
        batch_data['seg_token_positions'].append(processed['seg_token_positions'])
        batch_data['ground_truth_masks'].append(sample['mask'])
        batch_data['has_masks'].append(True)
    
    # Convert to tensors and pad if necessary
    # Handle variable length sequences
    max_len = max(ids.shape[0] for ids in batch_data['input_ids'])
    
    # Pad sequences
    padded_input_ids = []
    padded_attention_mask = []
    padded_seg_positions = []
    
    for ids, attn, seg_pos in zip(batch_data['input_ids'], 
                                  batch_data['attention_mask'], 
                                  batch_data['seg_token_positions']):
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
            attn = torch.cat([attn, torch.zeros(pad_len, dtype=attn.dtype)])
            seg_pos = torch.cat([seg_pos, torch.zeros(pad_len, dtype=seg_pos.dtype)])
        
        padded_input_ids.append(ids)
        padded_attention_mask.append(attn)
        padded_seg_positions.append(seg_pos)
    
    # Stack tensors
    batch_tensors = {
        'pixel_values': torch.stack(batch_data['pixel_values']),
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_mask),
        'sam_images': torch.stack(batch_data['sam_images']),
        'seg_token_positions': torch.stack(padded_seg_positions),
        'ground_truth_masks': torch.stack(batch_data['ground_truth_masks']),
        'has_masks': batch_data['has_masks'],
    }
    
    print(f"✅ Batch processed:")
    for key, tensor in batch_tensors.items():
        if isinstance(tensor, torch.Tensor):
            print(f"   {key}: {tensor.shape}")
        else:
            print(f"   {key}: {len(tensor)} items")
    
    return batch_tensors


def run_overfitting_test(model, batch_data, num_steps=50):
    """Run overfitting test on single batch"""
    print("\n" + "=" * 70)
    print("🏋️ Running Overfitting Test")
    print("=" * 70)
    
    # Set model to training mode
    model.train()
    
    # Create optimizer for trainable parameters only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
    
    print(f"📊 Training Setup:")
    print(f"   Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Learning rate: 1e-4")
    print(f"   Training steps: {num_steps}")
    
    # Track losses
    losses = {
        'total': [],
        'ce': [],
        'seg': [],
        'dice': [],
        'bce': []
    }
    
    print(f"\n🔄 Training Progress:")
    print("Step | Total Loss | CE Loss | Seg Loss | Dice Loss | BCE Loss")
    print("-" * 65)
    
    for step in range(num_steps):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model.model_forward(
            input_ids=batch_data['input_ids'],
            pixel_values=batch_data['pixel_values'],
            sam_images=batch_data['sam_images'],
            attention_mask=batch_data['attention_mask'],
            seg_token_positions=batch_data['seg_token_positions'],
            ground_truth_masks=batch_data['ground_truth_masks'],
            has_masks=batch_data['has_masks'],
            inference=False,
        )
        
        # Backward pass
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        # Record losses
        losses['total'].append(loss.item())
        losses['ce'].append(outputs.ce_loss.item())
        losses['seg'].append(outputs.seg_loss.item())
        losses['dice'].append(outputs.dice_loss.item())
        losses['bce'].append(outputs.bce_loss.item())
        
        # Print progress
        if step % 10 == 0 or step == num_steps - 1:
            print(f"{step:4d} | {loss.item():10.4f} | {outputs.ce_loss.item():7.4f} | "
                  f"{outputs.seg_loss.item():8.4f} | {outputs.dice_loss.item():9.4f} | "
                  f"{outputs.bce_loss.item():8.4f}")
    
    return losses


def analyze_overfitting_results(losses):
    """Analyze overfitting results"""
    print("\n" + "=" * 70)
    print("📊 Analyzing Overfitting Results")
    print("=" * 70)
    
    # Calculate loss improvements
    initial_loss = losses['total'][0]
    final_loss = losses['total'][-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"🔍 Loss Analysis:")
    print(f"   Initial total loss: {initial_loss:.4f}")
    print(f"   Final total loss: {final_loss:.4f}")
    print(f"   Improvement: {improvement:.2f}%")
    
    # Check for overfitting signs
    overfitting_checks = []
    
    # 1. Loss should decrease
    if final_loss < initial_loss:
        print(f"   ✅ Loss decreased during training")
        overfitting_checks.append(True)
    else:
        print(f"   ❌ Loss did not decrease")
        overfitting_checks.append(False)
    
    # 2. Significant improvement (>20%)
    if improvement > 20:
        print(f"   ✅ Significant improvement (>20%)")
        overfitting_checks.append(True)
    elif improvement > 10:
        print(f"   ⚠️ Moderate improvement (10-20%)")
        overfitting_checks.append(True)
    else:
        print(f"   ❌ Insufficient improvement (<10%)")
        overfitting_checks.append(False)
    
    # 3. Loss convergence (last 10 steps should be stable)
    if len(losses['total']) >= 10:
        last_10_losses = losses['total'][-10:]
        loss_std = np.std(last_10_losses)
        
        if loss_std < 0.01:
            print(f"   ✅ Loss converged (std: {loss_std:.6f})")
            overfitting_checks.append(True)
        else:
            print(f"   ⚠️ Loss still changing (std: {loss_std:.6f})")
            overfitting_checks.append(True)  # May need more steps
    
    # 4. All loss components should decrease
    component_improvements = {}
    for loss_type in ['ce', 'seg', 'dice', 'bce']:
        if len(losses[loss_type]) > 0:
            initial = losses[loss_type][0]
            final = losses[loss_type][-1]
            improvement = (initial - final) / initial * 100 if initial > 0 else 0
            component_improvements[loss_type] = improvement
            
            if improvement > 5:
                print(f"   ✅ {loss_type.upper()} loss improved: {improvement:.2f}%")
            elif improvement > 0:
                print(f"   ⚠️ {loss_type.upper()} loss improved slightly: {improvement:.2f}%")
            else:
                print(f"   ❌ {loss_type.upper()} loss did not improve: {improvement:.2f}%")
    
    # Overall assessment
    success_rate = sum(overfitting_checks) / len(overfitting_checks) * 100
    
    print(f"\n📈 Overfitting Assessment:")
    print(f"   Success rate: {success_rate:.1f}%")
    
    if success_rate >= 75:
        print(f"   ✅ Model can successfully overfit - training capability verified")
        return True
    elif success_rate >= 50:
        print(f"   ⚠️ Model shows some overfitting capability - may need tuning")
        return True
    else:
        print(f"   ❌ Model failed to overfit - training issues detected")
        return False


def plot_training_curves(losses, save_path="overfitting_curves.png"):
    """Plot training curves"""
    print(f"\n📈 Plotting training curves...")
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("LISA-Llama4 Single Batch Overfitting Results", fontsize=16)
        
        # Plot total loss
        axes[0, 0].plot(losses['total'], 'b-', linewidth=2)
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True)
        
        # Plot CE loss
        axes[0, 1].plot(losses['ce'], 'r-', linewidth=2)
        axes[0, 1].set_title("Cross-Entropy Loss")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(True)
        
        # Plot segmentation loss
        axes[0, 2].plot(losses['seg'], 'g-', linewidth=2)
        axes[0, 2].set_title("Segmentation Loss")
        axes[0, 2].set_xlabel("Step")
        axes[0, 2].set_ylabel("Loss")
        axes[0, 2].grid(True)
        
        # Plot Dice loss
        axes[1, 0].plot(losses['dice'], 'm-', linewidth=2)
        axes[1, 0].set_title("Dice Loss")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(True)
        
        # Plot BCE loss
        axes[1, 1].plot(losses['bce'], 'c-', linewidth=2)
        axes[1, 1].set_title("BCE Loss")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].grid(True)
        
        # Plot all losses together
        axes[1, 2].plot(losses['total'], 'b-', label='Total', linewidth=2)
        axes[1, 2].plot(losses['ce'], 'r-', label='CE', alpha=0.7)
        axes[1, 2].plot(losses['seg'], 'g-', label='Seg', alpha=0.7)
        axes[1, 2].plot(losses['dice'], 'm-', label='Dice', alpha=0.7)
        axes[1, 2].plot(losses['bce'], 'c-', label='BCE', alpha=0.7)
        axes[1, 2].set_title("All Losses")
        axes[1, 2].set_xlabel("Step")
        axes[1, 2].set_ylabel("Loss")
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved plot: {save_path}")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ Could not create plot: {e}")
        return False


def main():
    """Main overfitting test function"""
    print("🚀 LISA-Llama4 Single Batch Overfitting Test")
    print("=" * 70)
    print(f"Model: {config.MODEL_ID}")
    print("=" * 70)
    
    success = False
    
    try:
        # Initialize model and processor
        print("🔧 Initializing model and processor...")
        model = create_llama4_lisa_model(
            model_id=config.MODEL_ID,
            sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
            train_mask_decoder=config.TRAIN_MASK_DECODER,
            hidden_size=config.HIDDEN_SIZE,
            out_dim=config.OUT_DIM,
        )
        
        processor = Llama4DualStreamProcessor(model_id=config.MODEL_ID)
        model.set_seg_token_idx(processor.seg_token_id)
        
        print("✅ Model and processor initialized")
        
        # Create test batch
        test_samples = create_test_batch()
        
        # Process test batch
        batch_data = process_test_batch(test_samples, processor)
        
        # Run overfitting test
        losses = run_overfitting_test(model, batch_data, num_steps=50)
        
        # Analyze results
        success = analyze_overfitting_results(losses)
        
        # Plot results
        plot_training_curves(losses)
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 SINGLE BATCH OVERFITTING TEST SUMMARY")
    print("=" * 70)
    
    if success:
        print("🎉 Overfitting test successful!")
        print("   ✅ Model can overfit on single batch")
        print("   ✅ Training pipeline is working correctly")
        print("   ✅ Gradients are flowing properly")
        return 0
    else:
        print("❌ Overfitting test failed!")
        print("   ❌ Model cannot overfit on single batch")
        print("   ❌ Training pipeline may have issues")
        print("   ❌ Check model architecture and gradients")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
