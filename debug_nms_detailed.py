#!/usr/bin/env python3

import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax.transform import LegalizeOps
import onnx
from onnx import helper, TensorProto

def debug_nms_detailed():
    """Detailed debug of NMS score threshold issue."""
    
    print("=== Detailed NMS Debug ===")
    
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
    
    # Test with ONNX Runtime
    print("\n=== ONNX Runtime Test ===")
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        center_point_box=0
    )

    graph = helper.make_graph(
        [nms_node],
        "nms_test_debug",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 3, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 3]),
        ],
        initializer=[
            helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [3]),
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.1]),
            helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.2]),
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [3, 3])],
    )

    model = helper.make_model(graph, producer_name="nms_test_debug", opset_imports=[helper.make_opsetid("", 11)])
    
    import onnxruntime as ort
    ort_session = ort.InferenceSession(model.SerializeToString())
    ort_inputs = {
        "boxes": boxes_data,
        "scores": scores_data,
    }
    ort_output = ort_session.run(None, ort_inputs)
    print(f"ONNX Runtime output shape: {ort_output[0].shape}")
    print(f"ONNX Runtime output:\n{ort_output[0]}")
    
    # Test with TVM step by step
    print("\n=== TVM Step-by-Step Debug ===")
    
    # Step 1: Import ONNX model
    print("Step 1: Importing ONNX model...")
    mod = from_onnx(model, keep_params_in_input=True)
    
    # Step 2: Legalize
    print("Step 2: Legalizing operations...")
    mod = LegalizeOps()(mod)
    
    # Step 3: Build and run
    print("Step 3: Building and running...")
    target = tvm.target.Target("llvm")
    with tvm.target.Target(target):
        ex = relax.build(mod, target)
        vm = relax.VirtualMachine(ex, tvm.cpu())
        
        # Provide all 5 arguments as expected by the function
        tvm_output = vm["main"](
            tvm.runtime.Tensor(boxes_data),
            tvm.runtime.Tensor(scores_data),
            tvm.runtime.Tensor(np.array([3], dtype=np.int64)),  # max_output_boxes_per_class
            tvm.runtime.Tensor(np.array([0.1], dtype=np.float32)),  # iou_threshold
            tvm.runtime.Tensor(np.array([0.2], dtype=np.float32))   # score_threshold
        )
        print(f"TVM output shape: {tvm_output[0].shape}")
        print(f"TVM output:\n{tvm_output[0].numpy()}")
        
        # Analyze the results
        print(f"\n=== Analysis ===")
        print(f"ONNX Runtime selected {len(ort_output[0])} boxes")
        print(f"TVM selected {len(tvm_output[0].numpy())} boxes")
        
        # Check which boxes were selected
        ort_selected = ort_output[0]
        tvm_selected = tvm_output[0].numpy()
        
        print(f"\nONNX Runtime selected boxes:")
        for i, box_idx in enumerate(ort_selected):
            if box_idx[0] >= 0:  # Valid entry
                score = scores_data[0, box_idx[1], box_idx[2]]
                print(f"  {i}: batch={box_idx[0]}, class={box_idx[1]}, box={box_idx[2]} (score={score})")
        
        print(f"\nTVM selected boxes:")
        for i, box_idx in enumerate(tvm_selected):
            if box_idx[0] >= 0:  # Valid entry
                score = scores_data[0, box_idx[1], box_idx[2]]
                print(f"  {i}: batch={box_idx[0]}, class={box_idx[1]}, box={box_idx[2]} (score={score})")
        
        # Check if score threshold is being applied
        print(f"\nScore threshold analysis:")
        print(f"Scores: {scores_data[0, 0]}")
        print(f"Score threshold: 0.2")
        print(f"Expected valid boxes: {np.sum(scores_data[0, 0] >= 0.2)}")
        
        # Check if the issue is in valid_count calculation
        print(f"\nDebugging valid_count calculation...")
        
        # Let's manually test the binary search logic
        scores_sorted = np.sort(scores_data[0, 0])[::-1]  # Sort in descending order
        print(f"Sorted scores: {scores_sorted}")
        
        # Binary search for score threshold
        def binary_search_debug(scores, threshold):
            lo, hi = 0, len(scores)
            while lo < hi:
                mid = (lo + hi) // 2
                if scores[mid] > threshold:
                    lo = mid + 1
                else:
                    hi = mid
            return lo
        
        valid_count = binary_search_debug(scores_sorted, 0.2)
        print(f"Binary search result: {valid_count}")
        print(f"Expected: 2 (scores 0.9 and 0.3 >= 0.2)")
        
        # Check if the issue is in the NMS algorithm itself
        print(f"\nDebugging NMS algorithm...")
        print(f"TVM output has {len(tvm_selected)} boxes, but only {len(ort_selected)} should be selected")
        
        # Check if the issue is in the output shape
        print(f"\nOutput shape analysis:")
        print(f"TVM output shape: {tvm_output[0].shape}")
        print(f"ONNX Runtime output shape: {ort_output[0].shape}")
        print(f"Expected shape: [2, 3] (only 2 boxes should be selected)")

if __name__ == "__main__":
    debug_nms_detailed()