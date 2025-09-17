#!/usr/bin/env python3

import numpy as np
import tvm
from tvm import te
from tvm.topi.vision.nms_util import binary_search

def test_valid_count():
    """Test valid_count calculation with score threshold."""
    
    # Test data: scores [0.9, 0.3, 0.1], score_threshold = 0.2
    # Expected: valid_count should be 2 (only scores 0.9 and 0.3 >= 0.2)
    
    batch_classes = 1
    num_boxes = 3
    score_threshold = 0.2
    
    # Create test scores (sorted in descending order)
    scores_data = np.array([[0.9, 0.3, 0.1]], dtype=np.float32)
    
    # Create TE tensors
    scores = te.placeholder((batch_classes, num_boxes), name="scores", dtype="float32")
    
    # Create TIR function
    def binary_search_ir(scores, valid_count):
        ib = tvm.tir.ir_builder.create()
        scores = ib.buffer_ptr(scores)
        valid_count = ib.buffer_ptr(valid_count)
        
        with ib.for_range(0, batch_classes, name="i", kind="parallel") as i:
            binary_search(ib, i, tvm.tir.IntImm("int32", num_boxes), scores, score_threshold, valid_count)
        
        return ib.get()
    
    # Create output tensor
    valid_count = te.extern(
        [(batch_classes,)],
        [scores],
        lambda ins, outs: binary_search_ir(ins[0], outs[0]),
        dtype=["int32"],
        name="valid_count",
        tag="valid_count",
    )
    
    # Create schedule - try different approaches
    try:
        s = tvm.te.create_schedule(valid_count.op)
    except AttributeError:
        try:
            s = tvm.create_schedule(valid_count.op)
        except AttributeError:
            # Try using the schedule from the operation
            s = te.create_schedule(valid_count.op)
    
    # Build and run
    func = tvm.build(s, [scores, valid_count], "llvm")
    
    # Create runtime arrays
    scores_nd = tvm.nd.array(scores_data)
    valid_count_nd = tvm.nd.array(np.zeros((batch_classes,), dtype=np.int32))
    
    # Run
    func(scores_nd, valid_count_nd)
    
    print(f"Input scores: {scores_data}")
    print(f"Score threshold: {score_threshold}")
    print(f"Valid count: {valid_count_nd.numpy()}")
    print(f"Expected valid count: 2")
    
    # Verify
    expected_valid_count = 2
    actual_valid_count = valid_count_nd.numpy()[0]
    
    if actual_valid_count == expected_valid_count:
        print("✅ Valid count calculation is correct!")
    else:
        print(f"❌ Valid count calculation is wrong! Expected {expected_valid_count}, got {actual_valid_count}")

if __name__ == "__main__":
    test_valid_count()
