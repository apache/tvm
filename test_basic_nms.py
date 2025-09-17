#!/usr/bin/env python3

import numpy as np
import tvm
import tvm.relax as relax
from tvm import topi

def test_basic_nms():
    """Test basic NMS without dynamic shape"""
    
    # Create test data
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0],
                      [0.1, 0.1, 1.1, 1.1],
                      [0.2, 0.2, 1.2, 1.2]]], dtype=np.float32)  # 1 batch, 3 boxes
    
    scores = np.array([[[0.9, 0.8, 0.7],
                       [0.6, 0.5, 0.4]]], dtype=np.float32)  # 1 batch, 2 classes, 3 boxes
    
    print("Test data:")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Scores shape: {scores.shape}")
    print()
    
    # Test with max_boxes=1
    max_boxes = 1
    print(f"=== Testing with max_boxes={max_boxes} ===")
    
    # Create Relax function
    bb = relax.BlockBuilder()
    
    # Create properly typed variables
    boxes_var = relax.Var("boxes", relax.TensorStructInfo(boxes.shape, "float32"))
    scores_var = relax.Var("scores", relax.TensorStructInfo(scores.shape, "float32"))
    
    with bb.function("main", [boxes_var, scores_var]):
        with bb.dataflow():
            # Call NMS directly without legalization
            nms_result = bb.emit(
                relax.op.vision.all_class_non_max_suppression(
                    boxes_var,
                    scores_var,
                    relax.const(max_boxes, dtype="int64"),
                    relax.const(0.5, dtype="float32"),
                    relax.const(0.1, dtype="float32"),
                    output_format="onnx"
                )
            )
            
            # Extract selected_indices
            selected_indices = bb.emit(relax.TupleGetItem(nms_result, 0))
            
            bb.emit_output(selected_indices)
        bb.emit_func_output(selected_indices)
    
    # Build the module
    mod = bb.get()
    print("Module created successfully")
    
    # Skip legalization for now
    print("Skipping legalization...")
    
    # Compile and run
    target = tvm.target.Target("llvm")
    print("Compiling...")
    with tvm.target.Target(target):
        mod = relax.transform.ToNonDataflow()(mod)
        mod = relax.transform.CallTIRRewrite()(mod)
        mod = relax.transform.VMShapeLower()(mod)
        mod = relax.transform.ToMixedPrecision()(mod)
        mod = relax.transform.FoldConstant()(mod)
        mod = relax.transform.DeadCodeElimination()(mod)
    
    # Build the module
    ex = relax.build(mod, target)
    print("Compilation completed")
    
    # Create VM
    vm = relax.VirtualMachine(ex, tvm.cpu())
    print("VM created")
    
    # Run the function
    print("Running...")
    result = vm["main"](boxes, scores)
    print("Run completed")
    
    print(f"Output shape: {result.shape}")
    print(f"Output:\n{result}")
    print(f"Expected max boxes per class: {max_boxes}")
    print(f"Expected total boxes: {max_boxes * 2}")  # 2 classes
    print(f"Actual total boxes: {result.shape[0]}")

if __name__ == "__main__":
    test_basic_nms()
