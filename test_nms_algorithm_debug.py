#!/usr/bin/env python3

import numpy as np
import tvm
from tvm import te
from tvm.topi.vision.nms import all_class_non_max_suppression

def test_nms_algorithm_debug():
    """Debug NMS algorithm step by step."""
    
    print("=== NMS Algorithm Debug ===")
    
    # Create test data
    boxes_data = np.array([[[0.0, 0.0, 1.0, 1.0],    # Box 0
                            [2.0, 0.0, 3.0, 1.0],    # Box 1
                            [0.0, 2.0, 1.0, 3.0]]],  # Box 2
                        dtype=np.float32)
    
    scores_data = np.array([[[0.9, 0.3, 0.1]]], dtype=np.float32)
    
    print(f"Input boxes: {boxes_data[0]}")
    print(f"Input scores: {scores_data[0, 0]}")
    print(f"Score threshold: 0.2")
    print(f"Expected: Only boxes 0 and 1 should be selected (scores 0.9 and 0.3 >= 0.2)")
    
    # Create TVM tensors
    boxes = te.placeholder(boxes_data.shape, dtype="float32", name="boxes")
    scores = te.placeholder(scores_data.shape, dtype="float32", name="scores")
    
    # Call NMS directly
    print(f"\nCalling all_class_non_max_suppression...")
    nms_result = all_class_non_max_suppression(
        boxes,
        scores,
        max_output_boxes_per_class=3,
        iou_threshold=0.1,
        score_threshold=0.2,
        output_format="onnx"
    )
    
    print(f"NMS result type: {type(nms_result)}")
    print(f"NMS result length: {len(nms_result)}")
    
    # Check the result structure
    for i, tensor in enumerate(nms_result):
        print(f"Result {i}: {tensor}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
    
    # The issue might be in the NMS algorithm itself
    print(f"\nDebugging NMS algorithm...")
    print(f"The algorithm should:")
    print(f"1. Calculate valid_count = 2 (scores >= 0.2)")
    print(f"2. Only process the first 2 boxes (indices 0, 1)")
    print(f"3. Apply NMS to these 2 boxes")
    print(f"4. Return only the selected boxes")
    
    print(f"\nBut it seems to be processing all 3 boxes instead of just 2")
    print(f"This suggests that valid_count is not being used correctly")

if __name__ == "__main__":
    test_nms_algorithm_debug()
