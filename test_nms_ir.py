#!/usr/bin/env python3

import numpy as np
import tvm
import tvm.relax as relax
from tvm import topi, te

def test_nms_ir():
    """Test NMS IR function directly"""
    
    # Create test data
    batch_class = 2  # 1 batch * 2 classes
    num_boxes = 3
    
    # Create selected_indices (simulated NMS output)
    selected_indices = te.placeholder((batch_class, num_boxes), name="selected_indices", dtype="int32")
    
    # Create num_detections (how many boxes were selected per class)
    num_detections = te.placeholder((batch_class,), name="num_detections", dtype="int32")
    
    # Create row_offsets
    row_offsets = te.placeholder((batch_class,), name="row_offsets", dtype="int64")
    
    # Create max_output_boxes_per_class as a constant tensor
    max_boxes = 1
    max_output_boxes_per_class = te.compute((), lambda: max_boxes, name="max_boxes")
    
    # Create output tensor
    out_rows = batch_class * num_boxes  # Conservative upper bound
    out = te.placeholder((out_rows, 3), name="out", dtype="int64")
    
    # Test the IR function
    from tvm.topi.vision.nms import _collect_selected_indices_ir
    
    ir_func = _collect_selected_indices_ir(
        num_class=2,  # 2 classes
        selected_indices=selected_indices,
        num_detections=num_detections,
        row_offsets=row_offsets,
        out=out,
        max_output_boxes_per_class=max_output_boxes_per_class
    )
    
    print("IR function created successfully")
    print(f"IR function: {ir_func}")
    
    # Create a simple test to verify the IR
    def test_ir(selected_indices, num_detections, row_offsets, out):
        return ir_func
    
    # Create extern call
    result = te.extern(
        [(out_rows, 3)],
        [selected_indices, num_detections, row_offsets],
        lambda ins, outs: test_ir(ins[0], ins[1], ins[2], outs[0]),
        dtype=["int64"],
        name="test_collect_indices"
    )
    
    print(f"Result tensor: {result}")
    print(f"Result shape: {result.shape}")

if __name__ == "__main__":
    test_nms_ir()
