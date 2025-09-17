#!/usr/bin/env python3

import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax.transform import LegalizeOps
from onnx import helper, TensorProto

def create_nms_model():
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

    model = helper.make_model(graph, producer_name="nms_test")
    return model

def generate_random_inputs(model):
    input_values = {}
    for i in model.graph.input:
        shape = []
        for dim in i.type.tensor_type.shape.dim:
            shape.append(dim.dim_value)
        input_values[i.name] = np.random.rand(*shape).astype(np.float32)
    return input_values

# 创建模型和输入
model = create_nms_model()
inputs = generate_random_inputs(model)

print("Input shapes:")
for name, value in inputs.items():
    print(f"  {name}: {value.shape}")

# 转换模型
tvm_model = from_onnx(model, opset=11, keep_params_in_input=True)

# 应用 legalization
tvm_model = LegalizeOps()(tvm_model)

# 编译和运行
target = tvm.target.Target("llvm")
with tvm.target.Target(target):
    mod = relax.build(tvm_model, target=target)

vm = relax.VirtualMachine(mod, tvm.cpu())

# 准备输入
boxes = tvm.nd.array(inputs["boxes"])
scores = tvm.nd.array(inputs["scores"])

# 运行
tvm_out = vm["main"](boxes, scores)

print(f"\nTVM output shape: {tvm_out[0].shape}")
print("TVM output:")
tvm_out_np = tvm_out[0].numpy()
print(tvm_out_np)

# 运行 ONNX Runtime 获取期望输出
import onnxruntime as ort
sess = ort.InferenceSession(model.SerializeToString())
ort_out = sess.run(['selected_indices'], inputs)[0]

print(f"\nONNX output shape: {ort_out.shape}")
print("ONNX output:")
print(ort_out)

# 比较差异
print(f"\nDetailed comparison:")
diff = np.abs(tvm_out_np - ort_out)
print(f"Max difference: {np.max(diff)}")
print(f"Number of different elements: {np.sum(diff > 0)}")

print(f"\nElement-by-element comparison:")
for i in range(len(tvm_out_np)):
    for j in range(len(tvm_out_np[i])):
        tvm_val = tvm_out_np[i, j]
        ort_val = ort_out[i, j]
        diff_val = abs(tvm_val - ort_val)
        if diff_val > 0:
            print(f"  [{i},{j}]: TVM={tvm_val}, ONNX={ort_val}, diff={diff_val}")
        else:
            print(f"  [{i},{j}]: TVM={tvm_val}, ONNX={ort_val} ✓")

print(f"\nFull comparison:")
print("TVM:  ", tvm_out_np.flatten())
print("ONNX: ", ort_out.flatten())
print("Diff: ", diff.flatten())

