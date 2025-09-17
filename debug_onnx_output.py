#!/usr/bin/env python3

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as rt

def test_onnx_nms_output():
    """Test ONNX NMS to see the exact expected output pattern."""
    
    # Create the same ONNX model as in the test
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        center_point_box=0
    )

    boxes_shape = [1, 5, 4]  # batch_size, num_boxes, 4
    scores_shape = [1, 2, 5]  # batch_size, num_classes, num_boxes

    graph = helper.make_graph(
        [nms_node],
        "nms_test",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        ],
        initializer=[
            helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [3]),
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.5]),
            helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.1]),
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [0, 3])],
    )

    model = helper.make_model(graph, producer_name="nms_test", opset_imports=[helper.make_opsetid("", 11)])

    # Use the same random input generation as the test
    import sys
    sys.path.append('/ssd1/tlopexh/tvm/tests/python/relax')
    from test_frontend_onnx import generate_random_inputs
    inputs = generate_random_inputs(model, {})

    # Run with ONNX Runtime
    try:
        ort_session = rt.InferenceSession(model.SerializeToString())
        ort_out = ort_session.run(None, inputs)
        print("ONNX Runtime output:")
        print("Shape:", ort_out[0].shape)
        print("Data:")
        print(ort_out[0])
        print("\nFull output array:")
        for i, row in enumerate(ort_out[0]):
            print(f"Row {i}: {row}")
    except Exception as e:
        print(f"ONNX Runtime error: {e}")

if __name__ == "__main__":
    test_onnx_nms_output()
