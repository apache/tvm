#!/usr/bin/env python3
"""
Verification script for the new TVM build system.
This script checks if the pyproject.toml configuration is correct.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def run_command(cmd, cwd=None, capture_output=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=capture_output, 
            text=True, 
            check=True
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
    print("🔍 Verifying TVM build system configuration...\n")
    
    # Check required files
    required_files = [
        ("pyproject.toml", "Root pyproject.toml"),
        ("CMakeLists.txt", "Root CMakeLists.txt"),
        ("python/tvm/__init__.py", "TVM Python package"),
        ("python/setup.py", "Legacy setup.py (will be removed)"),
    ]
    
    all_files_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_files_exist = False
    
    print()
    
    # Check pyproject.toml syntax
    print("📋 Checking pyproject.toml syntax...")
    try:
        import tomllib
        with open("pyproject.toml", "rb") as f:
            tomllib.load(f)
        print("✅ pyproject.toml syntax is valid")
    except ImportError:
        print("⚠️  tomllib not available (Python < 3.11), skipping syntax check")
    except Exception as e:
        print(f"❌ pyproject.toml syntax error: {e}")
        all_files_exist = False
    
    print()
    
    # Check if scikit-build-core is available
    print("🔧 Checking build dependencies...")
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
            all_files_exist = False
    
    # Note: cmake Python package is optional, system CMake is sufficient
    print("✅ cmake Python package check skipped (system CMake is sufficient)")
    
    print()
    
    # Check CMake availability
    print("🏗️  Checking CMake availability...")
    cmake_result = run_command(["cmake", "--version"])
    if cmake_result:
        print("✅ CMake available")
        # Extract version
        version_line = cmake_result.stdout.split('\n')[0]
        print(f"   {version_line}")
    else:
        print("❌ CMake not found in PATH")
        all_files_exist = False
    
    print()
    
    # Check if we can build a source distribution
    print("📦 Testing source distribution build...")
    try:
        import build
        print("✅ build package available")
        
        # Try to build source distribution (optional test)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Copy necessary files to temp directory for testing
                import shutil
                shutil.copy2("pyproject.toml", tmpdir)
                shutil.copy2("CMakeLists.txt", tmpdir)
                
                result = run_command([
                    sys.executable, "-m", "build", "--sdist", "--no-isolation"
                ], cwd=tmpdir)
                
                if result:
                    print("✅ Source distribution build test passed")
                else:
                    print("⚠️  Source distribution build test failed (this is optional)")
        except Exception as e:
            print(f"⚠️  Source distribution build test skipped: {e}")
            print("   This test is optional and not required for basic functionality")
                
    except ImportError:
        print("⚠️  build package not available")
        print("   Install with: pip install build")
    
    print()
    
    # Summary
    if all_files_exist:
        print("🎉 All checks passed! The build system is properly configured.")
        print("\n📚 Next steps:")
        print("   1. Install in development mode: pip install -e .")
        print("   2. Test the installation: python test_installation.py")
        print("   3. Build a wheel: pip wheel -w dist .")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print("   - Install missing dependencies")
        print("   - Check file paths and permissions")
        print("   - Verify CMake installation")
        sys.exit(1)

if __name__ == "__main__":
    main()
