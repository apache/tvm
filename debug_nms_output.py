#!/usr/bin/env python3

import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
import onnx
import onnxruntime as ort

def test_nms_output():
    # Create ONNX model
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0],
                       [0.0, 0.1, 1.0, 1.1],
                       [0.0, -0.1, 1.0, 0.9],
                       [0.0, 10.0, 1.0, 11.0],
                       [0.0, 10.1, 1.0, 11.1],
                       [0.0, 100.0, 1.0, 101.0]]], dtype=np.float32)
    
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                        [0.95, 0.75, 0.6, 0.80, 0.5, 0.3]]], dtype=np.float32)
    
    max_output_boxes_per_class = np.array([3], dtype=np.int64)
    iou_threshold = np.array([0.5], dtype=np.float32)
    score_threshold = np.array([0.0], dtype=np.float32)
    
    # Create ONNX model
    onnx_model = create_onnx_model()
    
    # Convert to TVM
    print("转换 ONNX 模型...")
    tvm_model = from_onnx(onnx_model, opset=11)
    
    # Apply legalization
    print("应用 legalization...")
    tvm_model = relax.transform.LegalizeOps()(tvm_model)
    
    # Compile
    print("编译模型...")
    target = tvm.target.Target("llvm")
    mod = relax.build(tvm_model, target=target)
    
    # Run TVM
    print("运行 TVM...")
    vm = relax.VirtualMachine(mod, tvm.cpu())
    
    tvm_out = vm["main"](
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold
    )
    
    print("TVM 输出:")
    print(f"形状: {tvm_out[0].shape}")
    print(f"内容: {tvm_out[0].numpy()}")
    print(f"num_total_detections: {tvm_out[1].numpy()}")
    
    # Run ONNX Runtime
    print("\n运行 ONNX Runtime...")
    ort_session = ort.InferenceSession(onnx_model.SerializeToString())
    ort_out = ort_session.run(
        None,
        {
            "boxes": boxes,
            "scores": scores,
            "max_output_boxes_per_class": max_output_boxes_per_class,
            "iou_threshold": iou_threshold,
            "score_threshold": score_threshold
        }
    )
    
    print("ONNX 输出:")
    print(f"形状: {ort_out[0].shape}")
    print(f"内容: {ort_out[0]}")
    print(f"num_total_detections: {ort_out[1]}")

def create_onnx_model():
    import onnx
    from onnx import helper, TensorProto
    
    # Create inputs
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 6, 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2, 6])
    max_output_boxes_per_class = helper.make_tensor_value_info("max_output_boxes_per_class", TensorProto.INT64, [1])
    iou_threshold = helper.make_tensor_value_info("iou_threshold", TensorProto.FLOAT, [1])
    score_threshold = helper.make_tensor_value_info("score_threshold", TensorProto.FLOAT, [1])
    
    # Create outputs
    selected_indices = helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [None, 3])
    num_total_detections = helper.make_tensor_value_info("num_total_detections", TensorProto.INT64, [1])
    
    # Create NMS node
    nms_node = helper.make_node(
        "NonMaxSuppression",
        inputs=["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        outputs=["selected_indices", "num_total_detections"],
        name="nms"
    )
    
    # Create graph
    graph = helper.make_graph(
        [nms_node],
        "nms_test",
        [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold],
        [selected_indices, num_total_detections]
    )
    
    # Create model
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 11
    
    return model

if __name__ == "__main__":
    test_nms_output()