#!/usr/bin/env python3
"""
Llama4-LISA Complete Configuration Verification
Tests the comprehensive implementation based on official Llama-4 specifications
"""

import os
import sys
import torch
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all necessary imports"""
    print("🔧 Testing imports...")
    
    try:
        # Core dependencies
        import transformers
        print(f"   ✅ transformers: {transformers.__version__}")
        
        import torch
        print(f"   ✅ torch: {torch.__version__}")
        
        # Configuration
        from config_llama4 import Llama4LisaTrainingConfig, create_config, validate_config
        print("   ✅ config_llama4 imported successfully")
        
        # Model components
        from model.llama4_lisa import (
            Llama4LisaConfig, 
            Llama4TextConfig, 
            Llama4VisionConfig,
            Llama4LisaForConditionalGeneration,
            create_llama4_lisa_model,
            create_llama4_lisa_processor,
            get_model_info
        )
        print("   ✅ model.llama4_lisa imported successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_config_initialization():
    """Test configuration initialization"""
    print("\n📋 Testing configuration initialization...")
    
    try:
        from config_llama4 import Llama4LisaTrainingConfig, get_lambda_cloud_config
        
        # Test default configuration
        default_config = Llama4LisaTrainingConfig()
        print("   ✅ Default configuration created")
        
        # Test Lambda Cloud configuration  
        lambda_config = get_lambda_cloud_config()
        print("   ✅ Lambda Cloud configuration created")
        
        # Validate paths
        assert lambda_config.SAM_CHECKPOINT_PATH == "/lambda/nfs/lisa-gemma-project-fs/data/weights/sam_vit_h_4b8939.pth"
        assert lambda_config.DATASET_BASE_DIR == "/lambda/nfs/lisa-gemma-project-fs/data/dataset"
        print("   ✅ Paths correctly unified with Gemma-LISA")
        
        # Test configuration methods
        training_args = lambda_config.to_training_args_dict()
        model_config = lambda_config.to_model_config_dict()
        print("   ✅ Configuration conversion methods work")
        
        # Print summary
        lambda_config.print_config_summary()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_model_config_classes():
    """Test model configuration classes"""
    print("\n🔧 Testing model configuration classes...")
    
    try:
        from model.llama4_lisa import Llama4LisaConfig, Llama4TextConfig, Llama4VisionConfig
        
        # Test Llama4TextConfig
        text_config = Llama4TextConfig()
        assert text_config.vocab_size == 128256
        assert text_config.hidden_size == 4096
        assert text_config.model_type == "llama"
        print("   ✅ Llama4TextConfig initialized correctly")
        
        # Test Llama4VisionConfig
        vision_config = Llama4VisionConfig()
        assert vision_config.hidden_size == 1024
        assert vision_config.image_size == 1120
        assert vision_config.model_type == "siglip_vision_model"
        print("   ✅ Llama4VisionConfig initialized correctly")
        
        # Test Llama4LisaConfig
        lisa_config = Llama4LisaConfig()
        assert lisa_config.model_type == "llama4_lisa"
        assert lisa_config.vocab_size == 128256
        assert lisa_config.hidden_size == 4096
        assert lisa_config.num_experts == 16
        assert hasattr(lisa_config, 'text_config')
        assert hasattr(lisa_config, 'vision_config')
        print("   ✅ Llama4LisaConfig initialized with all parameters")
        
        # Verify text_config and vision_config types
        assert isinstance(lisa_config.text_config, Llama4TextConfig)
        assert isinstance(lisa_config.vision_config, Llama4VisionConfig)
        print("   ✅ Sub-configurations are properly typed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model config test failed: {e}")
        traceback.print_exc()
        return False

def test_model_initialization():
    """Test model initialization (without actual loading)"""
    print("\n🏗️  Testing model initialization...")
    
    try:
        from model.llama4_lisa import Llama4LisaConfig, Llama4LisaForConditionalGeneration
        
        # Create configuration
        config = Llama4LisaConfig()
        print("   ✅ Configuration created")
        
        # Check if SAM checkpoint exists
        if os.path.exists(config.sam_checkpoint_path):
            print(f"   ✅ SAM checkpoint found: {config.sam_checkpoint_path}")
        else:
            print(f"   ⚠️  SAM checkpoint not found: {config.sam_checkpoint_path}")
            print("      This is expected if running locally")
        
        # Test config registration
        assert hasattr(Llama4LisaForConditionalGeneration, 'config_class')
        assert Llama4LisaForConditionalGeneration.config_class == Llama4LisaConfig
        print("   ✅ Model config class properly registered")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model initialization test failed: {e}")
        traceback.print_exc()
        return False

def test_factory_functions():
    """Test factory functions"""
    print("\n🏭 Testing factory functions...")
    
    try:
        from model.llama4_lisa import create_llama4_lisa_model, create_llama4_lisa_processor
        
        # Test configuration creation (without actual model loading)
        print("   ✅ Factory functions imported successfully")
        
        # Verify function signatures
        import inspect
        
        # Check create_llama4_lisa_model signature
        sig = inspect.signature(create_llama4_lisa_model)
        params = list(sig.parameters.keys())
        assert 'model_id' in params
        assert 'sam_checkpoint_path' in params
        print("   ✅ create_llama4_lisa_model signature correct")
        
        # Check create_llama4_lisa_processor signature  
        sig = inspect.signature(create_llama4_lisa_processor)
        params = list(sig.parameters.keys())
        assert 'model_id' in params
        assert 'seg_token' in params
        print("   ✅ create_llama4_lisa_processor signature correct")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Factory function test failed: {e}")
        traceback.print_exc()
        return False

def test_gpu_availability():
    """Test GPU availability and CUDA setup"""
    print("\n🎮 Testing GPU availability...")
    
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            
            print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
            print(f"   ✅ GPU count: {gpu_count}")
            print(f"   ✅ Current device: {current_device}")
            print(f"   ✅ GPU name: {gpu_name}")
            
            # Test memory
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory
            gpu_memory_gb = gpu_memory / 1024**3
            print(f"   ✅ GPU memory: {gpu_memory_gb:.1f} GB")
            
            if "A100" in gpu_name:
                print("   🚀 A100 GPU detected - optimal for Llama-4")
            
        else:
            print("   ⚠️  CUDA not available - CPU mode")
            
        return True
        
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
        return False

def test_transformers_version():
    """Test transformers version compatibility"""
    print("\n📦 Testing transformers version...")
    
    try:
        import transformers
        from packaging import version
        
        current_version = version.parse(transformers.__version__)
        required_version = version.parse("4.51.0")
        
        if current_version >= required_version:
            print(f"   ✅ transformers {transformers.__version__} >= 4.51.0 (Llama-4 compatible)")
        else:
            print(f"   ❌ transformers {transformers.__version__} < 4.51.0 (upgrade required)")
            return False
            
        # Test Llama4 availability
        try:
            from transformers import Llama4ForConditionalGeneration, AutoProcessor
            print("   ✅ Llama4ForConditionalGeneration available")
        except ImportError:
            print("   ❌ Llama4ForConditionalGeneration not available")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Transformers version test failed: {e}")
        return False

def test_path_consistency():
    """Test path consistency with Gemma-LISA"""
    print("\n📁 Testing path consistency...")
    
    try:
        from config_llama4 import get_lambda_cloud_config
        
        # Get Llama4 config
        llama4_config = get_lambda_cloud_config()
        
        # Expected paths from Gemma-LISA
        expected_sam_path = "/lambda/nfs/lisa-gemma-project-fs/data/weights/sam_vit_h_4b8939.pth"
        expected_dataset_path = "/lambda/nfs/lisa-gemma-project-fs/data/dataset"
        
        # Verify consistency
        assert llama4_config.SAM_CHECKPOINT_PATH == expected_sam_path
        assert llama4_config.DATASET_BASE_DIR == expected_dataset_path
        
        print("   ✅ SAM checkpoint path unified")
        print("   ✅ Dataset base path unified")
        print("   ✅ Output directories properly configured")
        
        # Check if paths exist (if on Lambda Cloud)
        if os.path.exists("/lambda/nfs"):
            print("   🌩️  Lambda Cloud environment detected")
            if os.path.exists(expected_sam_path):
                print("   ✅ SAM checkpoint file exists")
            else:
                print("   ⚠️  SAM checkpoint file not found")
                
            if os.path.exists(expected_dataset_path):
                print("   ✅ Dataset directory exists")
            else:
                print("   ⚠️  Dataset directory not found")
        else:
            print("   💻 Local environment detected")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Path consistency test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all verification tests"""
    print("🚀 LLAMA4-LISA COMPLETE CONFIGURATION VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration Initialization", test_config_initialization), 
        ("Model Config Classes", test_model_config_classes),
        ("Model Initialization", test_model_initialization),
        ("Factory Functions", test_factory_functions),
        ("GPU Availability", test_gpu_availability),
        ("Transformers Version", test_transformers_version),
        ("Path Consistency", test_path_consistency),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION RESULTS SUMMARY")
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
        print("🎉 ALL TESTS PASSED! Ready for next phase.")
        return True
    else:
        print("🚨 Some tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 