#!/usr/bin/env python3
"""
Test script for AllClassNMS implementation
Run this from TVM root directory: python test_allclassnms_implementation.py
"""

import sys
import os

# Add TVM Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import tvm
        print("✓ TVM imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TVM: {e}")
        return False
    
    try:
        from tvm import relax
        print("✓ Relax imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Relax: {e}")
        return False
    
    try:
        from tvm.script import relax as R
        print("✓ Relax script imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Relax script: {e}")
        return False
    
    try:
        from tvm.relax.op import vision
        print("✓ Vision module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import vision module: {e}")
        return False
    
    return True

def test_allclassnms_function():
    """Test AllClassNMS function call."""
    print("\nTesting AllClassNMS function...")
    
    try:
        from tvm import relax
        from tvm.script import relax as R
        from tvm.relax.op import vision
        
        # Create test variables
        boxes = relax.Var('boxes', R.Tensor((1, 10, 4), 'float32'))
        scores = relax.Var('scores', R.Tensor((1, 3, 10), 'float32'))
        
        # Test function call
        result = vision.all_class_non_max_suppression(
            boxes, 
            scores, 
            relax.const(5, dtype='int64'),
            relax.const(0.5, dtype='float32'),
            relax.const(0.1, dtype='float32'),
            output_format='onnx'
        )
        
        print("✓ AllClassNMS function call successful")
        print(f"  Result type: {type(result)}")
        
        # Test with BlockBuilder
        bb = relax.BlockBuilder()
        with bb.function("test_func", [boxes, scores]):
            result = bb.emit(result)
            bb.emit_func_output(result)
        
        print("✓ BlockBuilder integration successful")
        
        return True
        
    except Exception as e:
        print(f"✗ AllClassNMS function failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_onnx_frontend():
    """Test ONNX frontend integration."""
    print("\nTesting ONNX frontend integration...")
    
    try:
        # Check if AllClassNMS is in the convert map
        from tvm.relax.frontend.onnx.onnx_frontend import _get_convert_map
        
        convert_map = _get_convert_map()
        if "AllClassNMS" in convert_map:
            print("✓ AllClassNMS found in ONNX convert map")
            print(f"  Converter class: {convert_map['AllClassNMS']}")
        else:
            print("✗ AllClassNMS not found in ONNX convert map")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ ONNX frontend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
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

def main():
    """Run all tests."""
    print("=" * 60)
    print("AllClassNMS Implementation Test")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("AllClassNMS Function", test_allclassnms_function),
        ("ONNX Frontend", test_onnx_frontend),
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
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! AllClassNMS implementation is complete.")
        print("\nTo run the actual ONNX test:")
        print("  python -m pytest tests/python/relax/test_frontend_onnx.py::test_allclassnms -v")
        print("\nTo run vision operation tests:")
        print("  python -m pytest tests/python/relax/test_op_vision.py -v")
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
