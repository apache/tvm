#!/usr/bin/env python3

import numpy as np
import tvm
from tvm import te
from tvm.topi.vision.nms import all_class_non_max_suppression

def test_nms_direct():
    """Test NMS algorithm directly without Relax."""
    
    print("=== Direct NMS Test ===")
    
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
    nms_result = all_class_non_max_suppression(
        boxes,
        scores,
        max_output_boxes_per_class=3,
        iou_threshold=0.1,
        score_threshold=0.2,
        output_format="onnx"
    )
    
    print(f"\nNMS result type: {type(nms_result)}")
    print(f"NMS result length: {len(nms_result)}")
    
    # Build and run
    target = tvm.target.Target("llvm")
    with tvm.target.Target(target):
        s = tvm.te.create_schedule([nms_result[0].op])
        func = tvm.build(s, [boxes, scores] + nms_result, target)
        
        # Run the function
        ctx = tvm.cpu()
        tvm_boxes = tvm.nd.array(boxes_data, ctx)
        tvm_scores = tvm.nd.array(scores_data, ctx)
        
        # Allocate output arrays
        tvm_outputs = []
        for i, tensor in enumerate(nms_result):
            tvm_outputs.append(tvm.nd.array(np.zeros(tensor.shape, dtype=tensor.dtype), ctx))
        
        # Call the function
        func(tvm_boxes, tvm_scores, *tvm_outputs)
        
        print(f"\nTVM NMS outputs:")
        for i, output in enumerate(tvm_outputs):
            print(f"Output {i} shape: {output.shape}")
            print(f"Output {i}:\n{output.numpy()}")
        
        # Analyze the results
        selected_indices = tvm_outputs[0].numpy()
        num_total_detections = tvm_outputs[1].numpy()
        
        print(f"\nAnalysis:")
        print(f"Selected indices shape: {selected_indices.shape}")
        print(f"Num total detections: {num_total_detections}")
        
        # Check which boxes were selected
        print(f"\nSelected boxes:")
        for i, box_idx in enumerate(selected_indices):
            if box_idx[0] >= 0:  # Valid entry
                score = scores_data[0, box_idx[1], box_idx[2]]
                print(f"  {i}: batch={box_idx[0]}, class={box_idx[1]}, box={box_idx[2]} (score={score})")
        
        # Check if score threshold is being applied
        print(f"\nScore threshold analysis:")
        print(f"Scores: {scores_data[0, 0]}")
        print(f"Score threshold: 0.2")
        print(f"Expected valid boxes: {np.sum(scores_data[0, 0] >= 0.2)}")
        print(f"Actual selected boxes: {len([x for x in selected_indices if x[0] >= 0])}")

if __name__ == "__main__":
    test_nms_direct()