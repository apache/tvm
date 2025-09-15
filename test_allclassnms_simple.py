#!/usr/bin/env python3
"""
Simple test script for AllClassNMS implementation
This test checks file structure and basic syntax without importing TVM
"""

import os
import re

def test_file_structure():
    """Test that all required files exist."""
    print("Testing file structure...")
    
    required_files = [
        "include/tvm/relax/attrs/vision.h",
        "src/relax/op/vision/nms.h", 
        "src/relax/op/vision/nms.cc",
        "python/tvm/relax/op/vision/__init__.py",
        "python/tvm/relax/op/vision/_ffi_api.py",
        "python/tvm/relax/op/vision/nms.py",
        "python/tvm/topi/vision/nms.py",
        "python/tvm/topi/vision/nms_util.py",
        "python/tvm/relax/transform/legalize_ops/vision.py",
        "tests/python/relax/test_op_vision.py",
        "tests/python/relax/test_tvmscript_parser_op_vision.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def test_python_syntax():
    """Test Python syntax of all Python files."""
    print("\nTesting Python syntax...")
    
    python_files = [
        "python/tvm/relax/op/vision/__init__.py",
        "python/tvm/relax/op/vision/_ffi_api.py",
        "python/tvm/relax/op/vision/nms.py",
        "python/tvm/topi/vision/nms.py",
        "python/tvm/topi/vision/nms_util.py",
        "python/tvm/relax/transform/legalize_ops/vision.py",
        "tests/python/relax/test_op_vision.py",
        "tests/python/relax/test_tvmscript_parser_op_vision.py"
    ]
    
    all_valid = True
    for file_path in python_files:
        if not os.path.exists(file_path):
            print(f"✗ {file_path} - FILE NOT FOUND")
            all_valid = False
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic syntax check
            compile(content, file_path, 'exec')
            print(f"✓ {file_path} - syntax valid")
            
        except SyntaxError as e:
            print(f"✗ {file_path} - syntax error: {e}")
            all_valid = False
        except Exception as e:
            print(f"✗ {file_path} - error: {e}")
            all_valid = False
    
    return all_valid

def test_cpp_syntax():
    """Test C++ syntax of header and source files."""
    print("\nTesting C++ syntax...")
    
    cpp_files = [
        "include/tvm/relax/attrs/vision.h",
        "src/relax/op/vision/nms.h",
        "src/relax/op/vision/nms.cc"
    ]
    
    all_valid = True
    for file_path in cpp_files:
        if not os.path.exists(file_path):
            print(f"✗ {file_path} - FILE NOT FOUND")
            all_valid = False
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic checks for C++ syntax
            if file_path.endswith('.h'):
                if '#ifndef' in content and '#define' in content and '#endif' in content:
                    print(f"✓ {file_path} - header guards present")
                else:
                    print(f"✗ {file_path} - missing header guards")
                    all_valid = False
            else:
                if '#include' in content and 'namespace' in content:
                    print(f"✓ {file_path} - basic structure present")
                else:
                    print(f"✗ {file_path} - missing basic structure")
                    all_valid = False
                    
        except Exception as e:
            print(f"✗ {file_path} - error: {e}")
            all_valid = False
    
    return all_valid

def test_onnx_frontend_integration():
    """Test that AllClassNMS is properly integrated in ONNX frontend."""
    print("\nTesting ONNX frontend integration...")
    
    onnx_frontend_path = "python/tvm/relax/frontend/onnx/onnx_frontend.py"
    
    if not os.path.exists(onnx_frontend_path):
        print(f"✗ ONNX frontend file not found: {onnx_frontend_path}")
        return False
    
    try:
        with open(onnx_frontend_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for AllClassNMS class
        if 'class AllClassNMS(OnnxOpConverter):' in content:
            print("✓ AllClassNMS class found in ONNX frontend")
        else:
            print("✗ AllClassNMS class not found in ONNX frontend")
            return False
        
        # Check for registration in convert map
        if '"AllClassNMS": AllClassNMS' in content:
            print("✓ AllClassNMS registered in convert map")
        else:
            print("✗ AllClassNMS not registered in convert map")
            return False
        
        # Check for vision operation usage
        if 'relax.op.vision.all_class_non_max_suppression' in content:
            print("✓ Vision operation used in implementation")
        else:
            print("✗ Vision operation not used in implementation")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading ONNX frontend: {e}")
        return False

def test_test_files():
    """Test that test files are properly structured."""
    print("\nTesting test files...")
    
    test_files = [
        "tests/python/relax/test_frontend_onnx.py",
        "tests/python/relax/test_op_vision.py",
        "tests/python/relax/test_tvmscript_parser_op_vision.py"
    ]
    
    all_valid = True
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"✗ {file_path} - FILE NOT FOUND")
            all_valid = False
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for test functions
            if 'def test_' in content:
                print(f"✓ {file_path} - contains test functions")
            else:
                print(f"✗ {file_path} - no test functions found")
                all_valid = False
                
        except Exception as e:
            print(f"✗ {file_path} - error: {e}")
            all_valid = False
    
    return all_valid

def main():
    """Run all tests."""
    print("=" * 60)
    print("AllClassNMS Implementation Test (Simple)")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
        ("C++ Syntax", test_cpp_syntax),
        ("ONNX Frontend Integration", test_onnx_frontend_integration),
        ("Test Files", test_test_files),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:25} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! AllClassNMS implementation structure is complete.")
        print("\nNext steps:")
        print("1. Build TVM: make -j$(nproc)")
        print("2. Run pytest tests:")
        print("   python -m pytest tests/python/relax/test_frontend_onnx.py::test_allclassnms -v")
        print("   python -m pytest tests/python/relax/test_op_vision.py -v")
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
