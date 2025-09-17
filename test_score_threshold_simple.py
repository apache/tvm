#!/usr/bin/env python3

import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax.transform import LegalizeOps
import onnx
from onnx import helper, TensorProto

def test_score_threshold_simple():
    """Simple test to verify score threshold is correctly extracted."""
    
    # Create ONNX model
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        center_point_box=0
    )

    boxes_data = np.array([[[0.0, 0.0, 1.0, 1.0],    # Box 0
                            [2.0, 0.0, 3.0, 1.0],    # Box 1
                            [0.0, 2.0, 1.0, 3.0]]],  # Box 2
                        dtype=np.float32)
    
    scores_data = np.array([[[0.9, 0.3, 0.1]]], dtype=np.float32)

    graph = helper.make_graph(
        [nms_node],
        "nms_test_simple",
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

    model = helper.make_model(graph, producer_name="nms_test_simple", opset_imports=[helper.make_opsetid("", 11)])
    
    # Import ONNX model
    mod = from_onnx(model, keep_params_in_input=True)
    print("Original model:")
    print(mod['main'])
    
    # Legalize
    mod = LegalizeOps()(mod)
    print("\nLegalized model:")
    print(mod['main'])
    
    # Check if score_threshold is correctly extracted
    # Look for the score_threshold value in the legalized model
    model_str = str(mod['main'])
    if "0.2" in model_str:
        print("\n✓ Score threshold 0.2 found in legalized model")
    else:
        print("\n✗ Score threshold 0.2 NOT found in legalized model")
        print("Looking for score threshold values in the model...")
        if "0.0" in model_str:
            print("Found 0.0 - this might be the default value")
        if "0.20000000298023224" in model_str:
            print("Found 0.20000000298023224 - this is the correct value")

if __name__ == "__main__":
    test_score_threshold_simple()
