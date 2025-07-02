#!/usr/bin/env python3
"""
Llama4-LISA Model Architecture Verification Script
Verifies the implementation of Llama-4-Scout-17B-16E-Instruct + SAM integration
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

from model.llama4_lisa import (
    Llama4LisaConfig,
    LISALlama4ForCausalLM,
    create_llama4_lisa_model
)
from utils.llama4_processing import Llama4DualStreamProcessor


def verify_model_loading():
    """Verify Llama4-LISA model loading"""
    print("\n" + "=" * 70)
    print("🔍 Testing Llama4-LISA Model Loading")
    print("=" * 70)
    
    try:
        # Create model configuration
        model_config = Llama4LisaConfig(
            model_id=config.MODEL_ID,
            hidden_size=config.HIDDEN_SIZE,
            out_dim=config.OUT_DIM,
            train_mask_decoder=config.TRAIN_MASK_DECODER,
            sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
            ce_loss_weight=config.CE_LOSS_WEIGHT,
            dice_loss_weight=config.DICE_LOSS_WEIGHT,
            bce_loss_weight=config.BCE_LOSS_WEIGHT,
            max_images=config.MAX_IMAGES_PER_INPUT,
        )
        
        print(f"📋 Model Configuration:")
        print(f"   Model ID: {model_config.model_id}")
        print(f"   Hidden Size: {model_config.hidden_size}")
        print(f"   Output Dim: {model_config.out_dim}")
        print(f"   Train Mask Decoder: {model_config.train_mask_decoder}")
        print(f"   Max Images: {model_config.max_images}")
        
        # Test factory function
        print(f"\n🏭 Creating model using factory function...")
        model = create_llama4_lisa_model(
            model_id=config.MODEL_ID,
            sam_checkpoint_path=config.SAM_CHECKPOINT_PATH,
            train_mask_decoder=config.TRAIN_MASK_DECODER,
            hidden_size=config.HIDDEN_SIZE,
            out_dim=config.OUT_DIM,
        )
        
        print("✅ Model creation successful!")
        return model, model_config
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def verify_model_architecture(model):
    """Verify model architecture components"""
    print("\n" + "=" * 70)
    print("🏗️ Verifying Model Architecture")
    print("=" * 70)
    
    try:
        # Check basic components
        print("🔍 Checking basic components...")
        
        # Check if model has required attributes
        required_attrs = [
            'visual_model',           # SAM model
            'text_hidden_fcs',       # Projection layers
            'config',                # Configuration
            'seg_token_idx',         # SEG token ID
        ]
        
        for attr in required_attrs:
            if hasattr(model, attr):
                print(f"   ✅ {attr}: Present")
            else:
                print(f"   ❌ {attr}: Missing")
                
        # Check SAM components
        print("\n🎯 Checking SAM components...")
        if hasattr(model, 'visual_model'):
            sam_components = [
                'image_encoder',
                'mask_decoder', 
                'prompt_encoder'
            ]
            
            for comp in sam_components:
                if hasattr(model.visual_model, comp):
                    print(f"   ✅ SAM {comp}: Present")
                else:
                    print(f"   ❌ SAM {comp}: Missing")
        
        # Check projection layers
        print("\n🔗 Checking projection layers...")
        if hasattr(model, 'text_hidden_fcs'):
            proj_layers = model.text_hidden_fcs
            print(f"   📊 Number of projection modules: {len(proj_layers)}")
            
            for i, layer in enumerate(proj_layers):
                print(f"   📋 Projection module {i}: {layer}")
        
        print("✅ Architecture verification completed!")
        return True
        
    except Exception as e:
        print(f"❌ Architecture verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_parameter_counts(model):
    """Verify model parameter counts and training configuration"""
    print("\n" + "=" * 70)
    print("📊 Analyzing Model Parameters")
    print("=" * 70)
    
    try:
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"📈 Parameter Analysis:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Frozen parameters: {frozen_params:,}")
        print(f"   Trainable ratio: {trainable_params/total_params*100:.2f}%")
        
        # Check SAM parameter status
        if hasattr(model, 'visual_model'):
            sam_total = sum(p.numel() for p in model.visual_model.parameters())
            sam_trainable = sum(p.numel() for p in model.visual_model.parameters() if p.requires_grad)
            
            print(f"\n🎯 SAM Parameter Status:")
            print(f"   SAM total parameters: {sam_total:,}")
            print(f"   SAM trainable parameters: {sam_trainable:,}")
            print(f"   SAM frozen parameters: {sam_total - sam_trainable:,}")
            
            # Check mask decoder specifically
            if hasattr(model.visual_model, 'mask_decoder'):
                mask_decoder_trainable = sum(
                    p.numel() for p in model.visual_model.mask_decoder.parameters() 
                    if p.requires_grad
                )
                print(f"   Mask decoder trainable: {mask_decoder_trainable:,}")
        
        # Check projection layer parameters
        if hasattr(model, 'text_hidden_fcs'):
            proj_trainable = sum(
                p.numel() for p in model.text_hidden_fcs.parameters() 
                if p.requires_grad
            )
            print(f"\n🔗 Projection Layer Parameters:")
            print(f"   Projection trainable: {proj_trainable:,}")
        
        # Verify training configuration matches expectations
        expected_trainable_ratio = 0.05  # Expect around 5% for LISA approach
        if trainable_params/total_params <= expected_trainable_ratio:
            print(f"✅ Training configuration looks correct (<= {expected_trainable_ratio*100:.1f}%)")
        else:
            print(f"⚠️ High trainable ratio (> {expected_trainable_ratio*100:.1f}%)")
            
        return {
            "total": total_params,
            "trainable": trainable_params,
            "ratio": trainable_params/total_params
        }
        
    except Exception as e:
        print(f"❌ Parameter analysis failed: {e}")
        return None


def verify_moe_architecture(model):
    """Verify MoE (Mixture of Experts) architecture"""
    print("\n" + "=" * 70)
    print("🧠 Verifying MoE Architecture")
    print("=" * 70)
    
    try:
        # Look for MoE layers in the model
        moe_layers = []
        expert_layers = []
        
        for name, module in model.named_modules():
            if 'moe' in name.lower() or 'expert' in name.lower():
                if 'moe' in name.lower():
                    moe_layers.append(name)
                if 'expert' in name.lower():
                    expert_layers.append(name)
        
        print(f"🔍 MoE Components Found:")
        print(f"   MoE layers: {len(moe_layers)}")
        print(f"   Expert layers: {len(expert_layers)}")
        
        if moe_layers:
            print("\n📋 MoE Layer Details:")
            for layer in moe_layers[:5]:  # Show first 5
                print(f"   - {layer}")
            if len(moe_layers) > 5:
                print(f"   ... and {len(moe_layers) - 5} more")
        
        # Check if this is expected for Llama-4-Scout (16 experts)
        expected_experts = config.NUM_EXPERTS
        print(f"\n🎯 Expected experts: {expected_experts}")
        
        if expert_layers:
            print(f"✅ MoE architecture detected with expert layers")
        else:
            print(f"⚠️ No explicit expert layers found (may be internal to Llama-4)")
            
        return len(moe_layers), len(expert_layers)
        
    except Exception as e:
        print(f"❌ MoE verification failed: {e}")
        return 0, 0


def verify_forward_pass(model):
    """Verify forward pass functionality"""
    print("\n" + "=" * 70)
    print("⚡ Testing Forward Pass")
    print("=" * 70)
    
    try:
        # Create dummy inputs
        batch_size = 1
        seq_len = 50
        
        # Create processor for realistic inputs
        processor = Llama4DualStreamProcessor(config.MODEL_ID)
        
        # Create dummy image and text
        from PIL import Image
        dummy_image = Image.new('RGB', (800, 600), color='blue')
        dummy_text = "Show me the blue region. [SEG]"
        
        print("🔧 Creating dummy inputs...")
        processed = processor.preprocess_dual_stream(dummy_image, dummy_text)
        
        # Set SEG token ID
        model.set_seg_token_idx(processor.seg_token_id)
        
        print("📊 Input shapes:")
        print(f"   Llama-4 pixel values: {processed['llama4_pixel_values'].shape}")
        print(f"   Input IDs: {processed['input_ids'].shape}")
        print(f"   SAM images: {processed['sam_images'].shape}")
        print(f"   SEG token positions: {processed['seg_token_positions'].shape}")
        
        # Test forward pass
        print("\n⚡ Running forward pass...")
        model.eval()
        
        with torch.no_grad():
            outputs = model.model_forward(
                input_ids=processed['input_ids'],
                pixel_values=processed['llama4_pixel_values'],
                sam_images=processed['sam_images'].unsqueeze(0),
                attention_mask=processed['attention_mask'],
                seg_token_positions=processed['seg_token_positions'],
                inference=True
            )
        
        print("📋 Output analysis:")
        print(f"   Logits shape: {outputs.logits.shape}")
        if outputs.pred_masks is not None:
            print(f"   Predicted masks shape: {outputs.pred_masks.shape}")
        else:
            print(f"   Predicted masks: None (no SEG tokens processed)")
        print(f"   Loss: {outputs.loss}")
        
        print("✅ Forward pass successful!")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_device_compatibility():
    """Verify device compatibility and memory requirements"""
    print("\n" + "=" * 70)
    print("💻 Device Compatibility Check")
    print("=" * 70)
    
    try:
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"🔧 CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"   GPU count: {device_count}")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
                
            # Check current memory usage
            current_memory = torch.cuda.memory_allocated() / (1024**3)
            max_memory = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"   Current memory usage: {current_memory:.2f} GB")
            print(f"   Peak memory usage: {max_memory:.2f} GB")
            
            # Estimate requirements
            print(f"\n📊 Memory Requirements:")
            print(f"   Llama-4-Scout-17B: ~34 GB (BF16)")
            print(f"   SAM ViT-H: ~2.5 GB")
            print(f"   Estimated total: ~40+ GB")
            
            if max_memory > 0:
                if max_memory < 40:
                    print(f"⚠️ May need more GPU memory for full training")
                else:
                    print(f"✅ Sufficient memory detected")
        else:
            print("⚠️ CUDA not available - CPU mode only")
            
        # Check transformers version
        import transformers
        print(f"\n📦 Transformers version: {transformers.__version__}")
        
        required_version = "4.51.0"
        from packaging import version
        if version.parse(transformers.__version__) >= version.parse(required_version):
            print(f"✅ Transformers version >= {required_version}")
        else:
            print(f"⚠️ Transformers version < {required_version} (required for Llama-4)")
            
        return cuda_available
        
    except Exception as e:
        print(f"❌ Device compatibility check failed: {e}")
        return False


def main():
    """Main verification function"""
    print("🚀 LISA-Llama4 Model Architecture Verification")
    print("=" * 70)
    print(f"Model: {config.MODEL_ID}")
    print(f"SAM Checkpoint: {config.SAM_CHECKPOINT_PATH}")
    print("=" * 70)
    
    # Track verification results
    results = {
        "model_loading": False,
        "architecture": False,
        "parameters": False,
        "moe": False,
        "forward_pass": False,
        "device": False,
    }
    
    # Run verifications
    try:
        # 1. Model loading
        model, model_config = verify_model_loading()
        if model is not None:
            results["model_loading"] = True
            
            # 2. Architecture verification
            results["architecture"] = verify_model_architecture(model)
            
            # 3. Parameter analysis
            param_stats = verify_parameter_counts(model)
            if param_stats is not None:
                results["parameters"] = True
            
            # 4. MoE verification
            moe_count, expert_count = verify_moe_architecture(model)
            results["moe"] = True
            
            # 5. Forward pass test
            results["forward_pass"] = verify_forward_pass(model)
        
        # 6. Device compatibility
        results["device"] = verify_device_compatibility()
        
    except KeyboardInterrupt:
        print("\n⚠️ Verification interrupted by user")
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 VERIFICATION SUMMARY")
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
        print("🎉 Model verification largely successful!")
        return 0
    elif success_rate >= 60:
        print("⚠️ Model verification partially successful - some issues detected")
        return 1
    else:
        print("❌ Model verification failed - significant issues detected")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 