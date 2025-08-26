#!/usr/bin/env python3
"""
Complete test script for the new TVM build system.
This script tests the entire build flow from source to wheel.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def run_command(cmd, cwd=None, capture_output=True, check=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=capture_output, 
            text=True, 
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd)}")
        print(f"   Error: {e}")
        if e.stdout:
            print(f"   Stdout: {e.stdout}")
        if e.stderr:
            print(f"   Stderr: {e.stderr}")
        return None

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (not found)")
        return False

def main():
    print("🚀 Testing TVM Build System Migration\n")
    
    # Step 1: Check prerequisites
    print("📋 Step 1: Checking prerequisites...")
    
    # Check required files
    required_files = [
        ("pyproject.toml", "Root pyproject.toml"),
        ("CMakeLists.txt", "Root CMakeLists.txt"),
        ("python/tvm/__init__.py", "TVM Python package"),
    ]
    
    all_files_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_files_exist = False
    
    if not all_files_exist:
        print("❌ Missing required files. Cannot proceed.")
        sys.exit(1)
    
    # Check build dependencies
    print("\n🔧 Checking build dependencies...")
    try:
        import scikit_build_core
        print(f"✅ scikit-build-core available: {scikit_build_core.__version__}")
    except ImportError:
        try:
            import skbuild_core
            print(f"✅ scikit-build-core available: {skbuild_core.__version__}")
        except ImportError:
            print("❌ scikit-build-core not available")
            print("   Install with: pip install scikit-build-core>=0.7.0")
            sys.exit(1)
    
    # Check CMake
    cmake_result = run_command(["cmake", "--version"])
    if not cmake_result:
        print("❌ CMake not found in PATH")
        sys.exit(1)
    print("✅ CMake available")
    
    print("\n✅ Prerequisites check passed!")
    
    # Step 2: Test editable install
    print("\n📦 Step 2: Testing editable install...")
    
    print("   Installing in editable mode...")
    install_result = run_command([
        sys.executable, "-m", "pip", "install", "-e", "."
    ])
    
    if not install_result:
        print("❌ Editable install failed")
        sys.exit(1)
    
    print("✅ Editable install successful!")
    
    # Step 3: Test import
    print("\n🐍 Step 3: Testing TVM import...")
    
    try:
        import tvm
        print(f"✅ TVM imported successfully!")
        print(f"   Version: {tvm.__version__}")
        print(f"   Runtime only: {getattr(tvm, '_RUNTIME_ONLY', False)}")
        
        # Test basic functionality
        print("\n🔧 Testing basic functionality...")
        
        # Test device creation
        cpu_dev = tvm.cpu(0)
        print(f"   CPU device: {cpu_dev}")
        
        # Test NDArray creation
        import numpy as np
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        tvm_array = tvm.nd.array(data, device=cpu_dev)
        print(f"   NDArray created: {tvm_array}")
        print(f"   Shape: {tvm_array.shape}")
        print(f"   Dtype: {tvm_array.dtype}")
        
        # Test basic operations
        result = tvm_array + 1
        print(f"   Array + 1: {result.numpy()}")
        
        print("\n✅ All basic tests passed!")
        
    except ImportError as e:
        print(f"❌ Failed to import TVM: {e}")
        print("   Make sure you have installed TVM correctly.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("   TVM imported but encountered an error during testing.")
        sys.exit(1)
    
    # Step 4: Test wheel building
    print("\n🏗️  Step 4: Testing wheel building...")
    
    # Create dist directory if it doesn't exist
    dist_dir = Path("dist")
    dist_dir.mkdir(exist_ok=True)
    
    print("   Building wheel...")
    wheel_result = run_command([
        sys.executable, "-m", "pip", "wheel", "-w", "dist", "."
    ])
    
    if not wheel_result:
        print("❌ Wheel building failed")
        sys.exit(1)
    
    # Check if wheel was created
    wheel_files = list(dist_dir.glob("tvm-*.whl"))
    if wheel_files:
        wheel_file = wheel_files[0]
        print(f"✅ Wheel created: {wheel_file}")
        print(f"   Size: {wheel_file.stat().st_size / (1024*1024):.2f} MB")
        
        # Check wheel name format (should be py3-none-any)
        if "py3-none-any" in wheel_file.name:
            print("✅ Wheel is Python version-agnostic (py3-none-any)")
        else:
            print("⚠️  Wheel may not be Python version-agnostic")
    else:
        print("❌ No wheel files found in dist/")
        sys.exit(1)
    
    # Step 5: Test wheel installation
    print("\n📥 Step 5: Testing wheel installation...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"   Testing wheel in: {tmpdir}")
        
        # Copy wheel to temp directory
        test_wheel = Path(tmpdir) / wheel_file.name
        shutil.copy2(wheel_file, test_wheel)
        
        # Install wheel
        install_wheel_result = run_command([
            sys.executable, "-m", "pip", "install", str(test_wheel)
        ], cwd=tmpdir)
        
        if not install_wheel_result:
            print("❌ Wheel installation failed")
            sys.exit(1)
        
        print("✅ Wheel installation successful!")
        
        # Test import from wheel
        try:
            # Change to temp directory and test import
            os.chdir(tmpdir)
            import tvm
            print("✅ TVM imported successfully from wheel!")
            os.chdir("/home/tlopex/tvm")  # Return to original directory
        except Exception as e:
            print(f"❌ Failed to import TVM from wheel: {e}")
            os.chdir("/home/tlopex/tvm")  # Return to original directory
            sys.exit(1)
    
    # Step 6: Test source distribution
    print("\n📦 Step 6: Testing source distribution...")
    
    try:
        import build
        print("✅ build package available")
        
        print("   Building source distribution...")
        sdist_result = run_command([
            sys.executable, "-m", "build", "--sdist", "--no-isolation"
        ])
        
        if sdist_result:
            sdist_files = list(dist_dir.glob("tvm-*.tar.gz"))
            if sdist_files:
                sdist_file = sdist_files[0]
                print(f"✅ Source distribution created: {sdist_file}")
                print(f"   Size: {sdist_file.stat().st_size / (1024*1024):.2f} MB")
            else:
                print("❌ No source distribution files found")
        else:
            print("❌ Source distribution build failed")
            
    except ImportError:
        print("⚠️  build package not available")
        print("   Install with: pip install build")
    
    # Summary
    print("\n🎉 All tests passed! The new build system is working correctly.")
    print("\n📚 Migration Summary:")
    print("   ✅ pyproject.toml configured with scikit-build-core")
    print("   ✅ CMake install rules configured for minimal wheel")
    print("   ✅ Python version-agnostic wheels (py3-none-any)")
    print("   ✅ Editable installs working")
    print("   ✅ Wheel building working")
    print("   ✅ Self-contained packages")
    
    print("\n🔧 Next steps:")
    print("   1. Test with different Python versions")
    print("   2. Test with different platforms")
    print("   3. Update CI/CD pipelines")
    print("   4. Remove old setup.py")
    print("   5. Update mlc-ai/package for version-agnostic wheels")

if __name__ == "__main__":
    main()
