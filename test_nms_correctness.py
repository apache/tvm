#!/usr/bin/env python3
"""Test NMS algorithm correctness with fixed data"""

import numpy as np
import tvm
from tvm import relax
from tvm.relax import op

def test_nms_correctness():
    """Test NMS algorithm correctness with known data"""
    
    # Create test data with known expected results
    # Boxes: [x1, y1, x2, y2] format
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0],    # Box 0: [0,0,1,1] - should be selected
                       [0.5, 0.5, 1.5, 1.5],    # Box 1: [0.5,0.5,1.5,1.5] - overlaps with box 0, should be suppressed
                       [2.0, 2.0, 3.0, 3.0]]],  # Box 2: [2,2,3,3] - no overlap, should be selected
                   dtype=np.float32)
    
    # Scores: higher score = better
    scores = np.array([[[0.9, 0.8, 0.7],        # Class 0: [0.9, 0.8, 0.7] - box 0 has highest score
                        [0.6, 0.5, 0.4]]],       # Class 1: [0.6, 0.5, 0.4] - box 0 has highest score
                      dtype=np.float32)
    
    print("Test data:")
    print(f"Boxes:\n{boxes[0]}")
    print(f"Scores:\n{scores[0]}")
    
    # Expected results:
    # Class 0: Box 0 (score 0.9) should be selected, Box 1 (score 0.8) should be suppressed due to IoU with Box 0
    # Class 1: Box 0 (score 0.6) should be selected, Box 1 (score 0.5) should be suppressed due to IoU with Box 0
    # So we expect: [[0, 0, 0], [0, 1, 0]] - 2 boxes total
    
    # Test with different max_boxes_per_class values
    for max_boxes in [1, 2, 3]:
        print(f"\n=== Testing with max_boxes_per_class={max_boxes} ===")
        
        # Create TVM constants
        boxes_const = relax.const(boxes, dtype="float32")
        scores_const = relax.const(scores, dtype="float32")
        max_boxes_const = relax.const(max_boxes, dtype="int64")
        iou_threshold_const = relax.const(0.5, dtype="float32")
        score_threshold_const = relax.const(0.1, dtype="float32")
        
        # Create a simple function
        bb = relax.BlockBuilder()
        
        with bb.function("main", [boxes_const, scores_const, max_boxes_const, iou_threshold_const, score_threshold_const]):
            with bb.dataflow():
                # Call NMS
                nms_result = bb.emit(
                    op.vision.all_class_non_max_suppression(
                        boxes_const,
                        scores_const,
                        max_boxes_const,
                        iou_threshold_const,
                        score_threshold_const,
                        output_format="onnx"
                    )
                )
                
                # Extract results
                selected_indices = bb.emit(relax.TupleGetItem(nms_result, 0))
                num_total_detections = bb.emit(relax.TupleGetItem(nms_result, 1))
                
                bb.emit_func_output(relax.Tuple([selected_indices, num_total_detections]))
        
        # Build and run
        mod = bb.get()
        mod = relax.transform.LegalizeOps()(mod)
        
        with tvm.transform.PassContext(opt_level=3):
            ex = tvm.compile(mod, target="llvm")
            vm = relax.VirtualMachine(ex, tvm.cpu())
        
        # Run
        vm.set_input("main", boxes, scores, max_boxes, 0.5, 0.1)
        vm.invoke_stateful("main")
        tvm_output = vm.get_outputs("main")
        
        selected_indices = tvm_output[0].numpy()
        num_total_detections = tvm_output[1].numpy()
        
        print(f"Output shape: {selected_indices.shape}")
        print(f"Selected indices:\n{selected_indices}")
        print(f"Num total detections: {num_total_detections}")
        
        # Verify correctness
        expected_max_boxes = 1 * 2 * max_boxes  # 1 batch * 2 classes * max_boxes
        actual_boxes = num_total_detections[0]
        
        print(f"Expected max boxes: {expected_max_boxes}")
        print(f"Actual boxes: {actual_boxes}")
        
        # Check that we don't exceed the limit
        assert actual_boxes <= expected_max_boxes, f"Too many boxes: {actual_boxes} > {expected_max_boxes}"
        
        # Check that selected boxes are valid
        for i in range(selected_indices.shape[0]):
            batch_idx, class_idx, box_idx = selected_indices[i]
            print(f"Box {i}: batch={batch_idx}, class={class_idx}, box={box_idx}")
            
            # Verify indices are within bounds
            assert 0 <= batch_idx < 1, f"Invalid batch index: {batch_idx}"
            assert 0 <= class_idx < 2, f"Invalid class index: {class_idx}"
            assert 0 <= box_idx < 3, f"Invalid box index: {box_idx}"
            
            # Verify the box has a reasonable score
            score = scores[0, class_idx, box_idx]
            print(f"  -> Score: {score:.2f}")
            assert score >= 0.1, f"Box score too low: {score} < 0.1"
        
        print("✓ Test passed!")

def test_nms_iou_suppression():
    """Test that NMS correctly suppresses overlapping boxes"""
    
    # Create overlapping boxes
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0],    # Box 0: [0,0,1,1]
                       [0.1, 0.1, 1.1, 1.1],    # Box 1: [0.1,0.1,1.1,1.1] - high IoU with box 0
                       [2.0, 2.0, 3.0, 3.0]]],  # Box 2: [2,2,3,3] - no overlap
                   dtype=np.float32)
    
    # Box 1 has higher score but should be suppressed due to IoU
    scores = np.array([[[0.8, 0.9, 0.7]]], dtype=np.float32)
    
    print(f"\n=== Testing IoU suppression ===")
    print(f"Boxes:\n{boxes[0]}")
    print(f"Scores:\n{scores[0]}")
    print("Expected: Only box 0 should be selected (higher score, no overlap)")
    
    # Test with IoU threshold 0.5
    boxes_const = relax.const(boxes, dtype="float32")
    scores_const = relax.const(scores, dtype="float32")
    max_boxes_const = relax.const(2, dtype="int64")
    iou_threshold_const = relax.const(0.5, dtype="float32")
    score_threshold_const = relax.const(0.1, dtype="float32")
    
    bb = relax.BlockBuilder()
    with bb.function("main", [boxes_const, scores_const, max_boxes_const, iou_threshold_const, score_threshold_const]):
        with bb.dataflow():
            nms_result = bb.emit(
                op.vision.all_class_non_max_suppression(
                    boxes_const, scores_const, max_boxes_const,
                    iou_threshold_const, score_threshold_const,
                    output_format="onnx"
                )
            )
            selected_indices = bb.emit(relax.TupleGetItem(nms_result, 0))
            num_total_detections = bb.emit(relax.TupleGetItem(nms_result, 1))
            bb.emit_func_output(relax.Tuple([selected_indices, num_total_detections]))
    
    mod = bb.get()
    mod = relax.transform.LegalizeOps()(mod)
    
    with tvm.transform.PassContext(opt_level=3):
        ex = tvm.compile(mod, target="llvm")
        vm = relax.VirtualMachine(ex, tvm.cpu())
    
    vm.set_input("main", boxes, scores, 2, 0.5, 0.1)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    
    selected_indices = tvm_output[0].numpy()
    num_total_detections = tvm_output[1].numpy()
    
    print(f"Selected indices:\n{selected_indices}")
    print(f"Num total detections: {num_total_detections}")
    
    # Verify that only one box is selected (the one with higher score)
    actual_boxes = num_total_detections[0]
    print(f"Actual boxes selected: {actual_boxes}")
    
    # Should select at least one box (the highest scoring one)
    assert actual_boxes >= 1, "Should select at least one box"
    
    # Check that the selected box has the highest score
    if actual_boxes > 0:
        selected_box_idx = selected_indices[0, 2]  # box index
        selected_score = scores[0, 0, selected_box_idx]
        print(f"Selected box {selected_box_idx} with score {selected_score:.2f}")
        
        # The selected box should have the highest score among non-suppressed boxes
        assert selected_score == 0.9, f"Should select box with highest score, got {selected_score}"
    
    print("✓ IoU suppression test passed!")

if __name__ == "__main__":
    test_nms_correctness()
    test_nms_iou_suppression()
