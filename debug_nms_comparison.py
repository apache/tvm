#!/usr/bin/env python3

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

def create_nms_model(max_boxes=2, iou_thresh=0.3, score_thresh=0.2):
    """Create a simple NMS model for testing"""
    boxes_shape = [1, 3, 4]  # batch_size, num_boxes, 4
    scores_shape = [1, 2, 3]  # batch_size, num_classes, num_boxes

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
            helper.make_tensor_value_info('boxes', TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info('scores', TensorProto.FLOAT, scores_shape),
        ],
        initializer=[
            helper.make_tensor('max_output_boxes_per_class', TensorProto.INT64, [1], [max_boxes]),
            helper.make_tensor('iou_threshold', TensorProto.FLOAT, [1], [iou_thresh]),
            helper.make_tensor('score_threshold', TensorProto.FLOAT, [1], [score_thresh]),
        ],
        outputs=[helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [0, 3])],
    )

    model = helper.make_model(graph, producer_name='nms_test')
    model.opset_import[0].version = 11
    return model

def test_nms_comparison():
    """Compare TVM and ONNX Runtime NMS outputs"""
    # Create test data
    np.random.seed(42)
    boxes = np.random.rand(1, 3, 4).astype(np.float32)
    scores = np.random.rand(1, 2, 3).astype(np.float32)
    
    print("Test data:")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Boxes:\n{boxes[0]}")
    print(f"Scores shape: {scores.shape}")
    print(f"Scores:\n{scores[0]}")
    print()

    # Test with different max_boxes values
    for max_boxes in [2, 3, 4]:
        print(f"=== Testing with max_boxes={max_boxes} ===")
        
        # Create model
        model = create_nms_model(max_boxes=max_boxes, iou_thresh=0.3, score_thresh=0.2)
        
        # ONNX Runtime
        ort_session = onnxruntime.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
        ort_output = ort_session.run([], {'boxes': boxes, 'scores': scores})
        
        print(f"ONNX Runtime output shape: {ort_output[0].shape}")
        print(f"ONNX Runtime output:\n{ort_output[0]}")
        
        # TVM
        tvm_model = from_onnx(model, opset=11, keep_params_in_input=True)
        tvm_model = relax.transform.DecomposeOpsForInference()(tvm_model)
        tvm_model = relax.transform.LegalizeOps()(tvm_model)
        
        # Get the function
        func = tvm_model['main']
        print(f"TVM function ret_type: {func.ret_struct_info}")
        
        # Use the same compilation as in the test
        tvm_model = relax.transform.DecomposeOpsForInference()(tvm_model)
        tvm_model = relax.transform.LegalizeOps()(tvm_model)
        
        # Separate model from parameters
        tvm_model, params = relax.frontend.detach_params(tvm_model)
        
        # Compile the relax graph into a VM then run
        with tvm.transform.PassContext(opt_level=3):
            ex = tvm.compile(tvm_model, target="llvm")
            vm = relax.VirtualMachine(ex, tvm.cpu())
        
        # Prepare inputs
        input_list = [boxes, scores]
        if params:
            input_list += params["main"]
        
        # Run model
        vm.set_input("main", *input_list)
        vm.invoke_stateful("main")
        tvm_output = vm.get_outputs("main")
        
        print(f"TVM output shape: {tvm_output.shape}")
        print(f"TVM output:\n{tvm_output}")
        print(f"Shape match: {tvm_output.shape == ort_output[0].shape}")
        print()

if __name__ == "__main__":
    test_nms_comparison()
