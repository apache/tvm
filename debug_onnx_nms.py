#!/usr/bin/env python3

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime

def test_onnx_nms_behavior():
    """Test ONNX Runtime NMS behavior with different max_boxes values"""
    
    # Create simple test data
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0],
                      [0.1, 0.1, 1.1, 1.1],
                      [0.2, 0.2, 1.2, 1.2]]], dtype=np.float32)  # 1 batch, 3 boxes
    
    scores = np.array([[[0.9, 0.8, 0.7],
                       [0.6, 0.5, 0.4]]], dtype=np.float32)  # 1 batch, 2 classes, 3 boxes
    
    print("Test data:")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Boxes:\n{boxes[0]}")
    print(f"Scores shape: {scores.shape}")
    print(f"Scores:\n{scores[0]}")
    print()
    
    # Test with different max_boxes values
    for max_boxes in [1, 2, 3]:
        print(f"=== Testing with max_boxes={max_boxes} ===")
        
        # Create ONNX model
        nms_node = helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
            outputs=['selected_indices'],
            name='nms'
        )
        
        graph = helper.make_graph(
            [nms_node],
            'nms_test',
            inputs=[
                helper.make_tensor_value_info('boxes', TensorProto.FLOAT, boxes.shape),
                helper.make_tensor_value_info('scores', TensorProto.FLOAT, scores.shape),
            ],
            initializer=[
                helper.make_tensor('max_output_boxes_per_class', TensorProto.INT64, [1], [max_boxes]),
                helper.make_tensor('iou_threshold', TensorProto.FLOAT, [1], [0.5]),
                helper.make_tensor('score_threshold', TensorProto.FLOAT, [1], [0.1]),
            ],
            outputs=[helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [0, 3])],
        )
        
        model = helper.make_model(graph, producer_name='nms_test')
        model.opset_import[0].version = 11
        
        # Run with ONNX Runtime
        ort_session = onnxruntime.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
        ort_output = ort_session.run([], {'boxes': boxes, 'scores': scores})
        
        print(f"ONNX Runtime output shape: {ort_output[0].shape}")
        print(f"ONNX Runtime output:\n{ort_output[0]}")
        print(f"Expected max boxes per class: {max_boxes}")
        print(f"Expected total boxes: {max_boxes * 2}")  # 2 classes
        print(f"Actual total boxes: {ort_output[0].shape[0]}")
        print()

if __name__ == "__main__":
    test_onnx_nms_behavior()

