#!/usr/bin/env python3

import numpy as np
import tvm
from tvm import te
from tvm.topi.vision.nms import all_class_non_max_suppression

def test_simple_fix():
    """Test the simple fix for score threshold."""
    
    # Create test data
    boxes_data = np.array([[[0.0, 0.0, 1.0, 1.0],    # Box 0
                            [2.0, 0.0, 3.0, 1.0],    # Box 1
                            [0.0, 2.0, 1.0, 3.0]]],  # Box 2
                        dtype=np.float32)
    
    # Scores: 0.9, 0.3, 0.1 - only first two should pass score threshold 0.2
    scores_data = np.array([[[0.9, 0.3, 0.1]]], dtype=np.float32)
    
    print(f"Input scores: {scores_data[0, 0]}")
    print(f"Score threshold: 0.2")
    print(f"Expected: 2 boxes (0.9 and 0.3 >= 0.2)")
    
    # Create TVM tensors
    boxes = te.placeholder((1, 3, 4), dtype="float32", name="boxes")
    scores = te.placeholder((1, 1, 3), dtype="float32", name="scores")
    
    # Call NMS
    result = all_class_non_max_suppression(boxes, scores, 3, 0.1, 0.2, 'onnx')
    
    if isinstance(result, list) and len(result) >= 1:
        selected_indices = result[0]
        actual_count = selected_indices.shape[0]
        print(f"Actual output boxes: {actual_count}")
        
        if actual_count == 2:
            print("✓ SUCCESS: score_threshold is working!")
        else:
            print("✗ FAILED: score_threshold is still not working")
            print("This means my TIR code fix is not effective")
    else:
        print("✗ FAILED: Unexpected result format")

if __name__ == "__main__":
    test_simple_fix()
