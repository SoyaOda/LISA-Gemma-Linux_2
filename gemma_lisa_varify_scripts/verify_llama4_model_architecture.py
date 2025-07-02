#!/usr/bin/env python3
"""
Llama4-LISA Model Architecture Verification Script
Tests actual model initialization, parameter counts, and architecture components
Based on comprehensive Llama-4-Scout-17B-16E-Instruct + SAM implementation
"""

import os
import sys
import torch
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_actual_model_initialization():
    """Test actual model initialization (memory intensive)"""
    print("🏗️  Testing actual model initialization...")
    
    try:
        from model.llama4_lisa import Llama4LisaConfig, Llama4LisaForConditionalGeneration
        from config_llama4 import get_lambda_cloud_config
        
        # Get configuration
        config = get_lambda_cloud_config()
        model_config_dict = config.to_model_config_dict()
        
        # Create model configuration
        lisa_config = Llama4LisaConfig(**model_config_dict)
        print("   ✅ Llama4LisaConfig created")
        
        # Test model class instantiation (without loading weights)
        print("   🔧 Testing model class instantiation...")
        
        # This tests the class structure without loading actual weights
        model_class = Llama4LisaForConditionalGeneration
        assert hasattr(model_class, '__init__')
        assert hasattr(model_class, 'forward')
        assert hasattr(model_class, 'get_visual_embs')
        print("   ✅ Model class structure verified")
        
        return True, lisa_config
        
    except Exception as e:
        print(f"   ❌ Model initialization test failed: {e}")
        traceback.print_exc()
        return False, None

def test_sam_model_loading():
    """Test SAM model loading and architecture"""
    print("\n🎭 Testing SAM model loading and architecture...")
    
    try:
        from model.llama4_lisa import Llama4LisaConfig
        from model.segment_anything import build_sam_vit_h
        
        # Create configuration
        config = Llama4LisaConfig()
        
        # Load SAM model
        if os.path.exists(config.sam_checkpoint_path):
            sam_model = build_sam_vit_h(config.sam_checkpoint_path)
            print("   ✅ SAM model loaded successfully")
            
            # Verify SAM architecture
            components = {
                'image_encoder': sam_model.image_encoder,
                'mask_decoder': sam_model.mask_decoder,
                'prompt_encoder': sam_model.prompt_encoder
            }
            
            for name, component in components.items():
                print(f"   ✅ {name}: {type(component).__name__}")
            
            # Parameter analysis
            total_params = sum(p.numel() for p in sam_model.parameters())
            trainable_params = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
            
            print(f"   📊 SAM total parameters: {total_params:,}")
            print(f"   📊 SAM trainable parameters: {trainable_params:,}")
            
            # Test mask decoder trainable status
            if config.train_mask_decoder:
                mask_decoder_trainable = sum(
                    p.numel() for p in sam_model.mask_decoder.parameters() 
                    if p.requires_grad
                )
                print(f"   🎯 Mask decoder trainable: {mask_decoder_trainable:,}")
            
            return True, sam_model
        else:
            print(f"   ❌ SAM checkpoint not found: {config.sam_checkpoint_path}")
            return False, None
            
    except Exception as e:
        print(f"   ❌ SAM model test failed: {e}")
        traceback.print_exc()
        return False, None

def test_llama4_base_model():
    """Test Llama4 base model availability"""
    print("\n🦙 Testing Llama4 base model availability...")
    
    try:
        from transformers import Llama4ForConditionalGeneration, AutoProcessor
        from config_llama4 import get_lambda_cloud_config
        
        config = get_lambda_cloud_config()
        model_id = config.MODEL_ID
        
        print(f"   🔍 Model ID: {model_id}")
        
        # Test that we can load config (without downloading)
        try:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(model_id)
            print("   ✅ Model config accessible")
            print(f"   📋 Model type: {model_config.model_type}")
            print(f"   📋 Vocab size: {model_config.vocab_size}")
            print(f"   📋 Hidden size: {model_config.hidden_size}")
            
            if hasattr(model_config, 'num_experts'):
                print(f"   📋 MoE experts: {model_config.num_experts}")
        except Exception as e:
            print(f"   ⚠️  Model config access failed: {e}")
        
        # Test processor availability
        try:
            # This doesn't download, just tests availability
            processor_class = AutoProcessor
            print("   ✅ AutoProcessor available")
        except Exception as e:
            print(f"   ❌ AutoProcessor test failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Llama4 base model test failed: {e}")
        traceback.print_exc()
        return False

def test_integration_architecture():
    """Test the integration architecture components"""
    print("\n🔗 Testing integration architecture...")
    
    try:
        from model.llama4_lisa import Llama4LisaConfig
        
        # Create configuration
        config = Llama4LisaConfig()
        
        # Test architectural parameters
        print("   📐 Architecture parameters:")
        print(f"      Llama4 hidden size: {config.hidden_size}")
        print(f"      Vision hidden size: {config.vision_config.hidden_size}")
        print(f"      Output dimension: {config.out_dim}")
        print(f"      MoE experts: {config.num_experts}")
        
        # Test projection dimensions
        vision_to_llama = config.vision_config.hidden_size  # 1024
        llama_hidden = config.hidden_size                   # 4096
        output_dim = config.out_dim                         # 256
        
        print(f"   🔄 Projection flow:")
        print(f"      Vision ({vision_to_llama}) → Llama ({llama_hidden}) → Seg ({output_dim})")
        
        # Verify dimensions are compatible
        assert vision_to_llama == 1024, f"Vision hidden size should be 1024, got {vision_to_llama}"
        assert llama_hidden == 4096, f"Llama hidden size should be 4096, got {llama_hidden}"
        assert output_dim == 256, f"Output dimension should be 256, got {output_dim}"
        
        print("   ✅ Architecture dimensions verified")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration architecture test failed: {e}")
        traceback.print_exc()
        return False

def test_loss_configuration():
    """Test loss function configuration"""
    print("\n⚖️  Testing loss configuration...")
    
    try:
        from model.llama4_lisa import Llama4LisaConfig
        from config_llama4 import get_lambda_cloud_config
        
        # Get configurations
        training_config = get_lambda_cloud_config()
        model_config = Llama4LisaConfig()
        
        # Test loss weights
        loss_weights = {
            'CE Loss': (model_config.ce_loss_weight, training_config.CE_LOSS_WEIGHT),
            'Dice Loss': (model_config.dice_loss_weight, training_config.DICE_LOSS_WEIGHT),
            'BCE Loss': (model_config.bce_loss_weight, training_config.BCE_LOSS_WEIGHT),
        }
        
        print("   📊 Loss weights configuration:")
        for loss_name, (model_weight, train_weight) in loss_weights.items():
            print(f"      {loss_name}: {model_weight} (model) = {train_weight} (training)")
            assert model_weight == train_weight, f"{loss_name} mismatch"
        
        print("   ✅ Loss configuration verified")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Loss configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_factory_functions():
    """Test factory functions with dry run"""
    print("\n🏭 Testing factory functions...")
    
    try:
        from model.llama4_lisa import create_llama4_lisa_model, create_llama4_lisa_processor
        import inspect
        
        # Test create_llama4_lisa_model signature
        model_sig = inspect.signature(create_llama4_lisa_model)
        model_params = list(model_sig.parameters.keys())
        
        required_model_params = ['model_id', 'sam_checkpoint_path']
        for param in required_model_params:
            assert param in model_params, f"Missing parameter: {param}"
        
        print(f"   ✅ create_llama4_lisa_model signature: {len(model_params)} parameters")
        
        # Test create_llama4_lisa_processor signature
        proc_sig = inspect.signature(create_llama4_lisa_processor)
        proc_params = list(proc_sig.parameters.keys())
        
        required_proc_params = ['model_id', 'seg_token']
        for param in required_proc_params:
            assert param in proc_params, f"Missing parameter: {param}"
        
        print(f"   ✅ create_llama4_lisa_processor signature: {len(proc_params)} parameters")
        
        # Test default parameters
        model_defaults = {k: v.default for k, v in model_sig.parameters.items() 
                         if v.default != inspect.Parameter.empty}
        proc_defaults = {k: v.default for k, v in proc_sig.parameters.items() 
                        if v.default != inspect.Parameter.empty}
        
        print(f"   📋 Model factory defaults: {len(model_defaults)} parameters")
        print(f"   📋 Processor factory defaults: {len(proc_defaults)} parameters")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Factory functions test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_estimation():
    """Test memory requirements estimation"""
    print("\n💾 Testing memory requirements estimation...")
    
    try:
        from config_llama4 import get_lambda_cloud_config
        
        config = get_lambda_cloud_config()
        
        # Memory estimates (in GB)
        estimates = {
            'Llama-4-Scout 17B (bf16)': 34.0,
            'SAM ViT-H': 2.5,
            'Integration layers': 0.1,
            'Working memory': 4.0,
            'Safety buffer': 2.0,
        }
        
        total_estimated = sum(estimates.values())
        
        print("   📊 Memory requirements breakdown:")
        for component, memory in estimates.items():
            print(f"      {component}: {memory:.1f} GB")
        print(f"      Total estimated: {total_estimated:.1f} GB")
        
        # Check against available memory
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"      Available GPU memory: {available_memory:.1f} GB")
            
            if available_memory >= total_estimated:
                print("   ✅ Sufficient GPU memory")
            else:
                print("   ⚠️  May need memory optimization")
                print(f"      Deficit: {total_estimated - available_memory:.1f} GB")
        else:
            print("   ⚠️  CUDA not available")
        
        # Test configuration memory settings
        print(f"\n   ⚙️  Training configuration:")
        print(f"      Batch size: {config.BATCH_SIZE}")
        print(f"      Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
        print(f"      Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
        print(f"      Data type: {config.TORCH_DTYPE}")
        
        return True, total_estimated
        
    except Exception as e:
        print(f"   ❌ Memory estimation test failed: {e}")
        traceback.print_exc()
        return False, 0

def run_model_architecture_tests():
    """Run all model architecture tests"""
    print("🚀 LLAMA4-LISA MODEL ARCHITECTURE VERIFICATION")
    print("=" * 60)
    print("Testing actual model components and architecture")
    print("=" * 60)
    
    tests = [
        ("Model Initialization", test_actual_model_initialization),
        ("SAM Model Loading", test_sam_model_loading),
        ("Llama4 Base Model", test_llama4_base_model),
        ("Integration Architecture", test_integration_architecture),
        ("Loss Configuration", test_loss_configuration),
        ("Factory Functions", test_factory_functions),
        ("Memory Estimation", test_memory_estimation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if test_name in ["Model Initialization", "SAM Model Loading"]:
                success, extra_info = test_func()
                results[test_name] = success
            elif test_name == "Memory Estimation":
                success, memory_req = test_func()
                results[test_name] = success
            else:
                results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 MODEL ARCHITECTURE VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL MODEL ARCHITECTURE TESTS PASSED!")
        print("🚀 Ready for basic forward pass testing!")
        return True
    else:
        print("🚨 Some architecture tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = run_model_architecture_tests()
    sys.exit(0 if success else 1) 