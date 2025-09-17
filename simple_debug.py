#!/usr/bin/env python3

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

# 创建简单的测试数据
boxes = np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1], [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0], [0.0, 10.1, 1.0, 11.1]]], dtype=np.float32)
scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5], [0.9, 0.75, 0.6, 0.95, 0.5]]], dtype=np.float32)

print("Boxes:")
print(boxes)
print("Scores:")
print(scores)

# 创建 ONNX 模型
nms_node = helper.make_node(
    'NonMaxSuppression',
    inputs=['boxes', 'scores'],
    outputs=['selected_indices'],
    name='nms',
    center_point_box=0,
    max_output_boxes_per_class=3,
    iou_threshold=0.5,
    score_threshold=0.1
)

boxes_input = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, 5, 4])
scores_input = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, 2, 5])
selected_indices_output = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [None, 3])

graph = helper.make_graph([nms_node], 'nms_model', [boxes_input, scores_input], [selected_indices_output])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])

# 运行 ONNX Runtime
try:
    sess = ort.InferenceSession(model.SerializeToString())
    ort_out = sess.run(['selected_indices'], {'boxes': boxes, 'scores': scores})[0]
    print(f"\nONNX output shape: {ort_out.shape}")
    print("ONNX output:")
    print(ort_out)
except Exception as e:
    print(f"ONNX Runtime error: {e}")
    # 手动计算期望输出
    print("\nManual calculation:")
    print("Expected pattern based on scores:")
    print("Class 0: scores [0.9, 0.75, 0.6, 0.95, 0.5]")
    print("Sorted by score: [0.95, 0.9, 0.75, 0.6, 0.5] -> indices [3, 0, 1, 2, 4]")
    print("NMS selection: [3, 0, 1] (top 3)")
    print("Class 1: same pattern")
    print("Expected output: [[0, 0, 3], [0, 0, 0], [0, 0, 1], [0, 1, 3], [0, 1, 0], [0, 1, 1]]")

