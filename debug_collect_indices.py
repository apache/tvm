#!/usr/bin/env python3

import numpy as np
import tvm
from tvm import relax, te, topi
from tvm.relax.frontend.onnx import from_onnx
import onnx
from onnx import helper, TensorProto

def debug_collect_indices():
    # Create a simple ONNX model
    boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, 4, 4])
    scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, 2, 4])
    max_output_boxes_per_class = helper.make_tensor_value_info('max_output_boxes_per_class', TensorProto.INT64, [1])
    iou_threshold = helper.make_tensor_value_info('iou_threshold', TensorProto.FLOAT, [1])
    score_threshold = helper.make_tensor_value_info('score_threshold', TensorProto.FLOAT, [1])
    
    selected_indices = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [6, 3])
    
    nms_node = helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
        outputs=['selected_indices'],
        name='nms'
    )
    
    graph = helper.make_graph([nms_node], 'nms_graph', 
                             [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold],
                             [selected_indices])
    
    model = helper.make_model(graph, producer_name='test')
    model.opset_import[0].version = 11
    
    # Convert to TVM
    tvm_model = from_onnx(model)
    
    # Create some test data
    boxes_data = np.random.rand(1, 4, 4).astype(np.float32)
    scores_data = np.random.rand(1, 2, 4).astype(np.float32)
    max_boxes_data = np.array([3], dtype=np.int64)
    iou_thresh_data = np.array([0.5], dtype=np.float32)
    score_thresh_data = np.array([0.1], dtype=np.float32)
    
    # Test the TOPI function directly
    print("Testing TOPI function directly...")
    
    # Create TE tensors
    boxes_te = te.placeholder((1, 4, 4), name="boxes", dtype="float32")
    scores_te = te.placeholder((1, 2, 4), name="scores", dtype="float32")
    max_boxes_te = te.placeholder((1,), name="max_boxes", dtype="int64")
    iou_thresh_te = te.placeholder((1,), name="iou_thresh", dtype="float32")
    score_thresh_te = te.placeholder((1,), name="score_thresh", dtype="float32")
    
    print(f"max_boxes_te type: {type(max_boxes_te)}")
    print(f"max_boxes_te shape: {max_boxes_te.shape}")
    
    # Call TOPI function
    result = topi.vision.all_class_non_max_suppression(
        boxes_te,
        scores_te,
        max_boxes_te,  # This is a te.Tensor
        iou_thresh_te,
        score_thresh_te,
        output_format="onnx"
    )
    
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result)}")
    print(f"Selected indices shape: {result[0].shape}")
    print(f"Num detections shape: {result[1].shape}")
    
    # Let's also test with a constant int
    print("\nTesting with constant int...")
    result2 = topi.vision.all_class_non_max_suppression(
        boxes_te,
        scores_te,
        3,  # This is an int
        iou_thresh_te,
        score_thresh_te,
        output_format="onnx"
    )
    
    print(f"Result2 type: {type(result2)}")
    print(f"Result2 length: {len(result2)}")
    print(f"Selected indices2 shape: {result2[0].shape}")
    print(f"Num detections2 shape: {result2[1].shape}")

if __name__ == "__main__":
    debug_collect_indices()

