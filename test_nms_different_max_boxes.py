#!/usr/bin/env python3

import numpy as np
import tvm
import tvm.relax as relax
from tvm import topi

def test_nms_different_max_boxes():
    """Test NMS with different max_boxes values"""
    
    # Create test data
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0],
                       [0.1, 0.1, 1.1, 1.1],
                       [0.2, 0.2, 1.2, 1.2]]], dtype=np.float32)
    
    scores = np.array([[[0.9, 0.8, 0.7],
                        [0.6, 0.5, 0.4]]], dtype=np.float32)
    
    print("Test data:")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Boxes:\n{boxes[0]}")
    print(f"Scores:\n{scores[0]}")
    
    # Test different max_boxes values
    for max_boxes in [1, 2, 3]:
        print(f"\n=== Testing with max_boxes={max_boxes} ===")
        
        # Create Relax function
        bb = relax.BlockBuilder()
        
        with bb.function("main", [relax.Var("boxes"), relax.Var("scores"), relax.Var("max_boxes")]):
            # Input parameters
            boxes_var = bb.emit(relax.const(boxes))
            scores_var = bb.emit(relax.const(scores))
            max_boxes_var = bb.emit(relax.const(max_boxes, dtype="int64"))
            iou_thresh = bb.emit(relax.const(0.5, dtype="float32"))
            score_thresh = bb.emit(relax.const(0.0, dtype="float32"))
            
            # Call NMS
            nms_result = bb.emit(
                relax.op.vision.all_class_non_max_suppression(
                    boxes_var, scores_var, max_boxes_var, iou_thresh, score_thresh
                )
            )
            
            # Extract results
            selected_indices = bb.emit(relax.TupleGetItem(nms_result, 0))
            num_total_detections = bb.emit(relax.TupleGetItem(nms_result, 1))
            
            bb.emit_func_output(relax.Tuple([selected_indices, num_total_detections]))
        
        # Build and run
        mod = bb.get()
        print("Module created successfully")
        
        # Legalize
        print("Legalizing...")
        mod = relax.transform.LegalizeOps()(mod)
        print("Legalization completed")
        
        # Compile
        print("Compiling...")
        mod = relax.transform.VMShapeLower()(mod)
        mod = relax.transform.VMBuild()(mod)
        print("Compilation completed")
        
        # Create VM
        vm = relax.VirtualMachine(mod, tvm.cpu())
        print("VM created")
        
        # Run
        print("Running...")
        result = vm["main"](boxes, scores, max_boxes)
        print("Run completed")
        
        selected_indices, num_total_detections = result
        selected_indices = selected_indices.numpy()
        num_total_detections = num_total_detections.numpy()
        
        print(f"Output shape: {selected_indices.shape}")
        print(f"num_total_detections: {num_total_detections}")
        print(f"Expected max boxes per class: {max_boxes}")
        print(f"Expected total boxes: {max_boxes * 2}")  # 2 classes
        print(f"Actual total boxes: {num_total_detections[0]}")
        
        # Show only the valid part
        valid_count = int(num_total_detections[0])
        if valid_count > 0:
            print(f"Valid indices (first {valid_count} rows):")
            print(selected_indices[:valid_count])
        else:
            print("No valid detections")

if __name__ == "__main__":
    test_nms_different_max_boxes()
