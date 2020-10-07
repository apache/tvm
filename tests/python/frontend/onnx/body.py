import onnx
import numpy as np

y_in = onnx.helper.make_tensor_value_info('y_in', onnx.TensorProto.FLOAT, [1])
y_out = onnx.helper.make_tensor_value_info('y_out', onnx.TensorProto.FLOAT, [1])
scan_out = onnx.helper.make_tensor_value_info('scan_out', onnx.TensorProto.FLOAT, [1])
cond_in = onnx.helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
cond_out = onnx.helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
iter_count = onnx.helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])

x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
y = np.array([-2]).astype(np.float32)

x_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['x'],
    value=onnx.helper.make_tensor(
        name='const_tensor_x',
        data_type=onnx.TensorProto.FLOAT,
        dims=x.shape,
        vals=x.flatten().astype(float),
    )
)

one_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['one'],
    value=onnx.helper.make_tensor(
        name='const_tensor_one',
        data_type=onnx.TensorProto.INT64,
        dims=(),
        vals=[1]
    )
)

i_add_node = onnx.helper.make_node(
    'Add',
    inputs=['iter_count', 'one'],
    outputs=['end']
)

start_unsqueeze_node = onnx.helper.make_node(
    'Unsqueeze',
    inputs=['iter_count'],
    outputs=['slice_start'],
    axes=[0]
)

end_unsqueeze_node = onnx.helper.make_node(
    'Unsqueeze',
    inputs=['end'],
    outputs=['slice_end'],
    axes=[0]
)

slice_node = onnx.helper.make_node(
    'Slice',
    inputs=['x', 'slice_start', 'slice_end'],
    outputs=['slice_out']
)

y_add_node = onnx.helper.make_node(
    'Add',
    inputs=['y_in', 'slice_out'],
    outputs=['y_out']
)

identity_node = onnx.helper.make_node(
    'Identity',
    inputs=['cond_in'],
    outputs=['cond_out']
)

scan_identity_node = onnx.helper.make_node(
    'Identity',
    inputs=['y_out'],
    outputs=['scan_out']
)

loop_body = onnx.helper.make_graph(
    [identity_node, x_const_node, one_const_node, i_add_node,
     start_unsqueeze_node, end_unsqueeze_node, slice_node, y_add_node,
     scan_identity_node],
    'loop_body',
    [iter_count, cond_in, y_in],
    [cond_out, y_out, scan_out]
)
body_model = onnx.helper.make_model(loop_body)
onnx.save(body_model, "body.onnx")

node = onnx.helper.make_node(
    'Loop',
    inputs=['trip_count', 'cond', 'y'],
    outputs=['res_y', 'res_scan'],
    body=loop_body
)

trip_count = np.array(5).astype(np.int64)
res_y = np.array([13]).astype(np.float32)
cond = np.array(1).astype(np.bool)
node_graph = onnx.helper.make_graph(
    [node],
    "loop_outer",
    inputs=[onnx.helper.make_tensor_value_info('trip_count', onnx.TensorProto.INT64, [5]),
            onnx.helper.make_tensor_value_info('cond', onnx.TensorProto.FLOAT, [1]),
            onnx.helper.make_tensor_value_info('y', onnx.TensorProto.BOOL, [1])],
    outputs=[onnx.helper.make_tensor_value_info('res_y', onnx.TensorProto.FLOAT, [13]),
             onnx.helper.make_tensor_value_info('res_scan', onnx.TensorProto.FLOAT, [5, 1])]
)
node_model = onnx.helper.make_model(node_graph)
onnx.save(node_model, "outer.onnx")