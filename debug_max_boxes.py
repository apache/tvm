#!/usr/bin/env python3

import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

def test_max_boxes_shape():
    # Create a simple ONNX model to see max_output_boxes_per_class shape
    import onnx
    from onnx import helper, TensorProto
    
    # Create a simple NMS model
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
    
    # Check the shape of max_output_boxes_per_class in the model
    print("TVM Model functions:")
    for name, func in tvm_model.functions.items():
        if name != "main":
            continue
        print(f"Function {name}:")
        print(func)
        print("\nStruct info:")
        print(func.struct_info)
        
        # Look for the NMS call
        def find_nms_call(expr):
            if hasattr(expr, 'op') and hasattr(expr.op, 'name'):
                if 'non_max_suppression' in expr.op.name:
                    print(f"Found NMS call: {expr}")
                    print(f"Args: {expr.args}")
                    for i, arg in enumerate(expr.args):
                        print(f"  Arg {i}: {arg}")
                        if hasattr(arg, 'struct_info'):
                            print(f"    Struct info: {arg.struct_info}")
            if hasattr(expr, 'body'):
                find_nms_call(expr.body)
            if hasattr(expr, 'blocks'):
                for block in expr.blocks:
                    for binding in block.bindings:
                        if hasattr(binding, 'value'):
                            find_nms_call(binding.value)
        
        find_nms_call(func.body)

if __name__ == "__main__":
    test_max_boxes_shape()

