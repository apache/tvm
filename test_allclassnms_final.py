#!/usr/bin/env python3
"""
Test script for AllClassNMS operator implementation
"""

import numpy as np
import onnx
from onnx import helper, TensorProto

def create_test_onnx_model():
    """Create a simple ONNX model with AllClassNMS operator"""
    
    # Create input shapes
    batch_size = 1
    num_boxes = 3
    num_classes = 2
    
    # Create input nodes
    boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [batch_size, num_boxes, 4])
    scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [batch_size, num_classes, num_boxes])
    max_output_boxes_per_class = helper.make_tensor_value_info('max_output_boxes_per_class', TensorProto.INT64, [])
    iou_threshold = helper.make_tensor_value_info('iou_threshold', TensorProto.FLOAT, [])
    score_threshold = helper.make_tensor_value_info('score_threshold', TensorProto.FLOAT, [])
    
    # Create output node
    output = helper.make_tensor_value_info('output', TensorProto.INT64, ['N', 3])
    
    # Create AllClassNMS node
    allclassnms_node = helper.make_node(
        'AllClassNMS',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
        outputs=['output'],
        center_point_box=0,
        output_format='onnx'
    )
    
    # Create graph
    graph = helper.make_graph(
        [allclassnms_node],
        'test_allclassnms',
        [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold],
        [output]
    )
    
    # Create model
    model = helper.make_model(graph)
    model.opset_import[0].version = 11
    
    return model

def test_onnx_model():
    """Test the ONNX model creation"""
    try:
        model = create_test_onnx_model()
        print("✓ ONNX model created successfully")
        print(f"  - Model opset version: {model.opset_import[0].version}")
        print(f"  - Number of nodes: {len(model.graph.node)}")
        print(f"  - Node name: {model.graph.node[0].name}")
        print(f"  - Node op_type: {model.graph.node[0].op_type}")
        print(f"  - Node inputs: {model.graph.node[0].input}")
        print(f"  - Node outputs: {model.graph.node[0].output}")
        return True
    except Exception as e:
        print(f"✗ Failed to create ONNX model: {e}")
        return False

if __name__ == "__main__":
    print("Testing AllClassNMS ONNX model creation...")
    success = test_onnx_model()
    
    if success:
        print("\n✓ AllClassNMS ONNX model test passed!")
        print("\nNext steps:")
        print("1. Test with TVM Relax frontend")
        print("2. Run the actual inference")
    else:
        print("\n✗ AllClassNMS ONNX model test failed!")
