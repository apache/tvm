#!/usr/bin/env python3
"""Test NMS with fixed data to verify correctness"""

import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
import onnx
from onnx import helper, TensorProto

def test_nms_with_fixed_data():
    """Test NMS with fixed data instead of random data"""
    
    # Create fixed test data
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0],    # Box 0: [0,0,1,1]
                       [0.5, 0.5, 1.5, 1.5],    # Box 1: [0.5,0.5,1.5,1.5] - overlaps with box 0
                       [2.0, 2.0, 3.0, 3.0]]],  # Box 2: [2,2,3,3] - no overlap
                   dtype=np.float32)
    
    scores = np.array([[[0.9, 0.8, 0.7],        # Class 0 scores: [0.9, 0.8, 0.7]
                        [0.6, 0.5, 0.4]]],       # Class 1 scores: [0.6, 0.5, 0.4]
                      dtype=np.float32)
    
    print("Fixed test data:")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Boxes:\n{boxes[0]}")
    print(f"Scores:\n{scores[0]}")
    
    # Create ONNX model
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        center_point_box=0
    )
    
    graph = helper.make_graph(
        [nms_node],
        "nms_test_fixed",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes.shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores.shape),
        ],
        initializer=[
            helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [2]),
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.5]),
            helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.1]),
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [4, 3])],
    )
    
    model = helper.make_model(graph, producer_name="nms_test_fixed")
    model.opset_import[0].version = 11  # Use opset 11 instead of default
    
    # Test with ONNX Runtime
    try:
        import onnxruntime as ort
        ort_session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        ort_output = ort_session.run([], {"boxes": boxes, "scores": scores})
        print(f"\nONNX Runtime output shape: {ort_output[0].shape}")
        print(f"ONNX Runtime output:\n{ort_output[0]}")
    except Exception as e:
        print(f"ONNX Runtime error: {e}")
        ort_output = None
    
    # Test with TVM
    try:
        tvm_model = from_onnx(model, opset=11, keep_params_in_input=True)
        tvm_model = relax.transform.DecomposeOpsForInference()(tvm_model)
        tvm_model = relax.transform.LegalizeOps()(tvm_model)
        tvm_model, params = relax.frontend.detach_params(tvm_model)
        
        with tvm.transform.PassContext(opt_level=3):
            ex = tvm.compile(tvm_model, target="llvm")
            vm = relax.VirtualMachine(ex, tvm.cpu())
        
        # Get the input parameters from the model
        input_params = [key for key in tvm_model["main"].params if key.name_hint in ["boxes", "scores"]]
        print(f"TVM model parameters: {[p.name_hint for p in tvm_model['main'].params]}")
        print(f"Number of parameters: {len(tvm_model['main'].params)}")
        
        # Prepare inputs in the correct order
        input_list = []
        for param in tvm_model["main"].params:
            if param.name_hint == "boxes":
                input_list.append(boxes)
            elif param.name_hint == "scores":
                input_list.append(scores)
            else:
                # For other parameters (like constants), we need to get them from params
                if param.name_hint in params["main"]:
                    input_list.append(params["main"][param.name_hint])
                else:
                    print(f"Warning: Parameter {param.name_hint} not found in params")
        
        # Add params if they exist
        if params:
            input_list += params["main"]
        
        vm.set_input("main", *input_list)
        vm.invoke_stateful("main")
        tvm_output = vm.get_outputs("main")
        
        print(f"\nTVM output shape: {tvm_output[0].numpy().shape}")
        print(f"TVM output:\n{tvm_output[0].numpy()}")
        
        # Compare outputs
        if ort_output is not None:
            tvm_np = tvm_output[0].numpy()
            ort_np = ort_output[0]
            
            # Handle shape mismatch
            if tvm_np.shape != ort_np.shape:
                if len(tvm_np.shape) == 2 and len(ort_np.shape) == 2 and tvm_np.shape[1] == ort_np.shape[1]:
                    if tvm_np.shape[0] > ort_np.shape[0]:
                        tvm_np = tvm_np[:ort_np.shape[0]]
                    elif ort_np.shape[0] > tvm_np.shape[0]:
                        padding = np.zeros((ort_np.shape[0] - tvm_np.shape[0], tvm_np.shape[1]), dtype=ort_np.dtype)
                        ort_np = np.concatenate([ort_np, padding], axis=0)
            
            print(f"\nComparison:")
            print(f"TVM (adjusted):\n{tvm_np}")
            print(f"ONNX Runtime (adjusted):\n{ort_np}")
            print(f"Shapes match: {tvm_np.shape == ort_np.shape}")
            print(f"Content match: {np.array_equal(tvm_np, ort_np)}")
            
    except Exception as e:
        print(f"TVM error: {e}")

if __name__ == "__main__":
    test_nms_with_fixed_data()
