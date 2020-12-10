# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import numpy as np
import math
import onnx
from onnx import helper, TensorProto, mapping, numpy_helper
import torch
import torchvision
import tvm.topi.testing
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
import scipy
import tvm.testing


def get_input_data_shape_dict(graph_def, input_data):
    if isinstance(input_data, list):
        input_names = {}
        shape_dict = {}
        for i, _ in enumerate(input_data):
            input_names[i] = graph_def.graph.input[i].name
            shape_dict[input_names[i]] = input_data[i].shape
    else:
        input_names = graph_def.graph.input[0].name
        shape_dict = {input_names: input_data.shape}

    return input_names, shape_dict


def get_tvm_output_with_vm(
    graph_def, input_data, target, ctx, opset=None, freeze_params=False, convert_to_static=False
):
    """ Generic function to execute and get tvm output with vm executor"""
    if not isinstance(input_data, list):
        input_data = [input_data]
    _, shape_dict = get_input_data_shape_dict(graph_def, input_data)

    mod, params = relay.frontend.from_onnx(
        graph_def, shape_dict, opset=opset, freeze_params=freeze_params
    )
    if convert_to_static:
        from tvm.relay import transform

        mod = transform.DynamicToStatic()(mod)

    ex = relay.create_executor("vm", mod=mod, ctx=ctx, target=target)
    result = ex.evaluate()(*input_data)
    if isinstance(result, tvm.runtime.NDArray):
        return result.asnumpy()
    return [r.asnumpy() for r in result]


def get_tvm_output(
    graph_def, input_data, target, ctx, output_shape=None, output_dtype="float32", opset=None
):
    """ Generic function to execute and get tvm output"""
    target = "llvm"

    input_names, shape_dict = get_input_data_shape_dict(graph_def, input_data)

    mod, params = relay.frontend.from_onnx(graph_def, shape_dict, opset=opset)

    with tvm.transform.PassContext(opt_level=1):
        graph, lib, params = relay.build(mod, target, params=params)

    ctx = tvm.cpu(0)
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    if isinstance(input_data, list):
        for i, e in enumerate(input_names):
            # Its possible for some onnx inputs to not be needed in the tvm
            # module, confirm its present before setting.
            try:
                m.set_input(input_names[i], tvm.nd.array(input_data[i].astype(input_data[i].dtype)))
            except:
                continue
    else:
        m.set_input(input_names, tvm.nd.array(input_data.astype(input_data.dtype)))

    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    if isinstance(output_shape, list) and isinstance(output_dtype, list):
        tvm_output_list = []
        for i, _ in enumerate(output_shape):
            tvm_output = m.get_output(i)
            tvm_output_list.append(tvm_output.asnumpy())
        return tvm_output_list
    else:
        tvm_output = m.get_output(0)
        return tvm_output.asnumpy()


def get_onnxruntime_output(model, inputs, dtype="float32"):
    import onnxruntime.backend

    rep = onnxruntime.backend.prepare(model, "CPU")
    if isinstance(inputs, list) and len(inputs) > 1:
        return rep.run(inputs)
    elif isinstance(inputs, list) and len(inputs) == 1:
        inp = inputs[0]
    else:
        inp = inputs
    return rep.run(inp.astype(dtype))[0]


def verify_with_ort_with_inputs(
    model,
    inputs,
    out_shape=None,
    targets=None,
    use_vm=False,
    opset=None,
    freeze_params=False,
    convert_to_static=False,
    dtype="float32",
    rtol=1e-5,
    atol=1e-5,
):
    def flatten(out):
        if isinstance(out, list) and len(out) == 1:
            out = out[0]
        if isinstance(out, np.ndarray):
            return out.flatten()
        return out

    ort_out = get_onnxruntime_output(model, inputs, dtype)

    if targets is None:
        targets = [tgt for (tgt, _) in tvm.testing.enabled_targets()]

    for target in targets:
        ctx = tvm.context(target, 0)
        if use_vm:
            tvm_out = get_tvm_output_with_vm(
                model,
                inputs,
                target,
                ctx,
                opset=opset,
                freeze_params=freeze_params,
                convert_to_static=convert_to_static,
            )
        else:
            tvm_out = get_tvm_output(model, inputs, target, ctx, out_shape, dtype, opset=opset)

        tvm.testing.assert_allclose(flatten(ort_out), flatten(tvm_out), rtol=rtol, atol=atol)


def verify_with_ort(
    model,
    input_shapes,
    out_shape=None,
    targets=None,
    use_vm=False,
    opset=None,
    freeze_params=False,
    convert_to_static=False,
    dtype="float32",
    rtol=1e-5,
    atol=1e-5,
):
    inputs = [np.random.uniform(size=ishape).astype(dtype) for ishape in input_shapes]
    verify_with_ort_with_inputs(
        model,
        inputs,
        out_shape=out_shape,
        targets=targets,
        use_vm=use_vm,
        opset=opset,
        freeze_params=freeze_params,
        convert_to_static=convert_to_static,
        dtype=dtype,
        rtol=rtol,
        atol=atol,
    )


def make_constant_node(name, data_type, dims, vals):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=helper.make_tensor(name=name, data_type=data_type, dims=dims, vals=vals),
    )


@tvm.testing.uses_gpu
def test_reshape():
    in_shape = (4, 3, 3, 4)
    ref_shape = (6, 2, 4, 3)

    ref_array = np.array(ref_shape)
    ref_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["ref_in"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.INT32,
            dims=ref_array.shape,
            vals=ref_array.flatten().astype(int),
        ),
    )
    reshape_node = helper.make_node("Reshape", ["in", "ref_in"], ["out"])

    graph = helper.make_graph(
        [ref_node, reshape_node],
        "reshape_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(ref_shape))],
    )

    model = helper.make_model(graph, producer_name="reshape_test")

    for target, ctx in tvm.testing.enabled_targets():
        x = np.random.uniform(size=in_shape).astype("int32")
        tvm_out = get_tvm_output(model, x, target, ctx, ref_shape, "float32")
        tvm.testing.assert_allclose(ref_shape, tvm_out.shape)


# TODO(mbrookhart): enable once VM supports heterogenous execution
# @tvm.testing.uses_gpu
def test_expand():
    def _test_expand(name, data, shape, ref_data, dtype="int32"):
        shape_array = np.array(shape)
        if dtype == "int32":
            shape_node = onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=["shape"],
                value=onnx.helper.make_tensor(
                    name="const_tensor",
                    data_type=onnx.TensorProto.INT32,
                    dims=shape_array.shape,
                    vals=shape_array.flatten().astype("int32"),
                ),
            )
        elif dtype == "int64":
            shape_node = onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=["shape"],
                value=onnx.helper.make_tensor(
                    name="const_tensor",
                    data_type=onnx.TensorProto.INT64,
                    dims=shape_array.shape,
                    vals=shape_array.flatten().astype("int64"),
                ),
            )
        else:
            raise "Invalid dtype"
        expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])

        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(data.shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(ref_data.shape))],
        )

        model = helper.make_model(graph, producer_name=name)

        for target, ctx in tvm.testing.enabled_targets():
            tvm_out = get_tvm_output_with_vm(model, data, target, ctx, freeze_params=True)
            tvm.testing.assert_allclose(ref_data, tvm_out)

    in_shape = (3, 1)
    shape = (3, 4)
    data = np.random.uniform(size=in_shape).astype(np.float32)
    ref_data = np.tile(data, 4)
    _test_expand("expand_with_dim_unchanged_test", data, shape, ref_data, "int32")
    _test_expand("expand_with_dim_unchanged_test", data, shape, ref_data, "int64")

    in_shape = (3, 1)
    shape = (2, 1, 6)
    data = np.random.uniform(size=in_shape).astype(np.float32)
    ref_data = data * np.ones(shape, dtype=np.float32)
    _test_expand("expand_with_dim_changed_test", data, shape, ref_data, "int32")
    _test_expand("expand_with_dim_changed_test", data, shape, ref_data, "int64")


def verify_depth_to_space(inshape, outshape, mode, blockSize):
    node = onnx.helper.make_node("DepthToSpace", inputs=["x"], outputs=["y"], blocksize=blockSize)

    graph = helper.make_graph(
        [node],
        "depth_to_space_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(inshape))],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(outshape))],
    )

    model = helper.make_model(graph, producer_name="depth_to_space_test")

    verify_with_ort(model, [inshape], outshape)


@tvm.testing.uses_gpu
def test_depth_to_space():
    # current onnx.checker use OpSet-1 version of DepthToSpace, which doesn't have a mode argument.
    # TO-DO, we can add mode arguement to test CRD mode and DCR mode
    # in the future when we update to a newer onnx version.
    verify_depth_to_space((1, 8, 2, 3), (1, 2, 4, 6), mode="CRD", blockSize=2)


def verify_space_to_depth(inshape, outshape, blockSize):
    node = onnx.helper.make_node("SpaceToDepth", inputs=["x"], outputs=["y"], blocksize=blockSize)

    graph = helper.make_graph(
        [node],
        "space_to_depth_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(inshape))],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(outshape))],
    )

    model = helper.make_model(graph, producer_name="space_to_depth_test")

    verify_with_ort(model, [inshape], outshape)


@tvm.testing.uses_gpu
def test_space_to_depth():
    verify_space_to_depth((1, 1, 4, 6), (1, 4, 2, 3), 2)


@tvm.testing.uses_gpu
def test_shape():
    in_shape = (4, 3, 3, 4)
    ref_shape = (6, 2, 4, 3)

    ref_array = np.array(ref_shape)
    ref_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["ref_in"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.INT32,
            dims=ref_array.shape,
            vals=ref_array.flatten().astype(int),
        ),
    )
    reshape_node = helper.make_node("Reshape", ["in", "ref_in"], ["out"])

    shape_node = helper.make_node("Shape", ["out"], ["final_out"])

    graph = helper.make_graph(
        [ref_node, reshape_node, shape_node],
        "shape_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("final_out", TensorProto.FLOAT, list(ref_shape))],
    )

    model = helper.make_model(graph, producer_name="shape_test")

    for target, ctx in tvm.testing.enabled_targets():
        x = np.random.uniform(size=in_shape).astype("int32")
        tvm_out = get_tvm_output(model, x, target, ctx, ref_shape, "int32")
        tvm.testing.assert_allclose(ref_shape, tvm_out)


def _test_power_iteration(x_shape, y_shape):
    if isinstance(y_shape, int):
        y_shape = [y_shape]

    x = np.random.uniform(size=x_shape).astype(np.float32)
    y = np.random.uniform(size=y_shape).astype(np.float32)

    np_res = np.power(x, y).astype(np.float32)

    res = helper.make_node("Pow", ["x", "y"], ["out"])

    graph = helper.make_graph(
        [res],
        "power_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape)),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(np_res.shape))],
    )

    model = helper.make_model(graph, producer_name="power_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [x, y], target, ctx, np_res.shape)
        tvm.testing.assert_allclose(np_res, tvm_out, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_power():
    _test_power_iteration((1, 3), (1))
    _test_power_iteration((2, 3), (2, 3))
    _test_power_iteration((2, 3), (1, 3))


def verify_range(start, limit, delta, dtype):
    dtype_map = {
        "float32": TensorProto.FLOAT,
        "int32": TensorProto.INT32,
        "int64": TensorProto.INT64,
    }
    dtype_onnx = dtype_map[dtype]
    y = helper.make_node("Range", ["start", "limit", "delta"], ["output"])
    graph = helper.make_graph(
        [y],
        "range_test",
        inputs=[
            helper.make_tensor_value_info("start", dtype_onnx, []),
            helper.make_tensor_value_info("limit", dtype_onnx, []),
            helper.make_tensor_value_info("delta", dtype_onnx, []),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "output", dtype_onnx, np.arange(start, limit, delta).shape
            )
        ],
    )
    model = helper.make_model(graph, producer_name="range_test")
    inputs = [np.array(x).astype(dtype) for x in [start, limit, delta]]
    verify_with_ort_with_inputs(model, inputs, use_vm=True)


@tvm.testing.uses_gpu
def test_range():
    for t in ["float32", "int32", "int64"]:
        verify_range(0, 10, 1, t)
        verify_range(2, 8, 2, t)
        verify_range(-3, 6, 4, t)
        verify_range(-2, -7, -1, t)


@tvm.testing.uses_gpu
def test_squeeze():
    in_shape = (1, 3, 1, 3, 1, 1)
    out_shape = (3, 3)
    y = helper.make_node("Squeeze", ["in"], ["out"], axes=[0, 2, 4, 5])

    graph = helper.make_graph(
        [y],
        "squeeze_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="squeeze_test")

    for target, ctx in tvm.testing.enabled_targets():
        x = np.random.uniform(size=in_shape).astype("float32")
        tvm_out = get_tvm_output(model, x, target, ctx, out_shape, "float32")
        tvm.testing.assert_allclose(out_shape, tvm_out.shape)


@tvm.testing.uses_gpu
def test_flatten():

    in_shape = (1, 3, 4, 4)
    axis = 1
    ref_shape = (1, 48)

    flatten_node = helper.make_node("Flatten", ["in"], ["out"], axis=axis)

    graph = helper.make_graph(
        [flatten_node],
        "flatten_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(ref_shape))],
    )

    model = helper.make_model(graph, producer_name="flatten_test")

    for target, ctx in tvm.testing.enabled_targets():
        x = np.random.uniform(size=in_shape).astype("int32")
        tvm_out = get_tvm_output(model, x, target, ctx, ref_shape, "float32")
        tvm.testing.assert_allclose(ref_shape, tvm_out.shape)


@tvm.testing.uses_gpu
def test_unsqueeze():
    in_shape = (3, 3)
    axis = (0, 3, 4)
    out_shape = (1, 3, 3, 1, 1)
    y = helper.make_node("Unsqueeze", ["in"], ["out"], axes=list(axis))

    graph = helper.make_graph(
        [y],
        "squeeze_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="squeeze_test")

    for target, ctx in tvm.testing.enabled_targets():
        x = np.random.uniform(size=in_shape).astype("float32")
        tvm_out = get_tvm_output(model, x, target, ctx, out_shape, "float32")
        tvm.testing.assert_allclose(out_shape, tvm_out.shape)


def verify_gather(in_shape, indices, axis, dtype):
    x = np.random.uniform(size=in_shape).astype(dtype)
    indices = np.array(indices, dtype="int32")
    out_np = np.take(x, indices, axis=axis)

    y = helper.make_node("Gather", ["in", "indices"], ["out"], axis=axis)

    graph = helper.make_graph(
        [y],
        "gather_test",
        inputs=[
            helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape)),
            helper.make_tensor_value_info("indices", TensorProto.INT32, list(indices.shape)),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_np.shape))],
    )
    model = helper.make_model(graph, producer_name="gather_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [x, indices], target, ctx, out_np.shape)
        tvm.testing.assert_allclose(out_np, tvm_out)


@tvm.testing.uses_gpu
def test_gather():
    verify_gather((4,), [1], 0, "int32")
    verify_gather((1, 4), [0], 0, "int32")
    verify_gather((4,), [[[1, 0], [0, 1]]], 0, "float32")
    verify_gather((2, 2), [[[1, 0], [0, 1]]], 1, "int32")
    verify_gather((3, 3, 3), [[[1, 0]]], -1, "int32")
    verify_gather((4, 3, 5, 6), [[2, 1, 0, 0]], 0, "float32")


def verify_gatherelements(in_shape, indices, axis):
    x = np.random.uniform(size=in_shape).astype("float32")
    indices = np.array(indices, dtype="int32")

    y = helper.make_node("GatherElements", ["data", "indices"], ["output"], axis=axis)
    graph = helper.make_graph(
        [y],
        "gather_elements_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, list(in_shape)),
            helper.make_tensor_value_info("indices", TensorProto.INT32, list(indices.shape)),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, list(in_shape))],
    )
    model = helper.make_model(graph, producer_name="gather_elements_test")

    verify_with_ort_with_inputs(model, [x, indices])


@tvm.testing.uses_gpu
def test_gatherelements():
    verify_gatherelements((4,), [3, 0, 2, 1], 0)
    verify_gatherelements((2, 2), [[1, 0], [0, 1]], 0)
    verify_gatherelements((2, 2), [[0, 0], [1, 0]], 1)
    verify_gatherelements((2, 2), [[1, 0], [0, 1]], 1)

    indices = [
        [[1, 0, 0], [1, 0, 1], [0, 1, 1]],
        [[1, 1, 1], [1, 2, 1], [1, 0, 1]],
        [[1, 2, 1], [1, 2, 1], [1, 2, 1]],
    ]

    verify_gatherelements((3, 3, 3), indices, 2)


def verify_scatter(in_shape, indices, axis):
    x = np.random.uniform(size=in_shape).astype("float32")
    indices = np.array(indices, dtype="int32")
    updates = np.random.uniform(size=indices.shape).astype("float32")

    y = helper.make_node("ScatterElements", ["data", "indices", "updates"], ["output"], axis=axis)

    graph = helper.make_graph(
        [y],
        "scatter_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, list(in_shape)),
            helper.make_tensor_value_info("indices", TensorProto.INT32, list(indices.shape)),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, list(indices.shape)),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, list(in_shape))],
    )
    model = helper.make_model(graph, producer_name="scatter_test")
    verify_with_ort_with_inputs(model, [x, indices, updates])


@tvm.testing.uses_gpu
def test_scatter():
    verify_scatter((4,), [1], 0)
    verify_scatter((1, 4), [[0]], 0)
    verify_scatter((4,), [2, 3], 0)
    verify_scatter((2, 2), [[1, 0], [0, 1]], 1)
    verify_scatter((3, 3, 3), [[[-1, -3]]], -1)
    verify_scatter((4, 3, 5, 6), [[[[2, 1, 0, 0]]]], 0)


def _test_slice_iteration_v1(indata, outdata, starts, ends, axes=None):
    if axes:
        y = helper.make_node("Slice", ["in"], ["out"], axes=axes, starts=starts, ends=ends)
    else:
        y = helper.make_node("Slice", ["in"], ["out"], starts=starts, ends=ends)

    graph = helper.make_graph(
        [y],
        "slice_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))],
    )

    model = helper.make_model(graph, producer_name="slice_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, indata, target, ctx, outdata.shape, "float32", opset=1)
        tvm.testing.assert_allclose(outdata, tvm_out)


def _test_slice_iteration_v10(indata, outdata, **attrs):
    starts = attrs["starts"]
    ends = attrs["ends"]
    axes = None if "axes" not in attrs else attrs["axes"]
    steps = None if "steps" not in attrs else attrs["steps"]
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    inputs = [
        helper.make_tensor_value_info("data", TensorProto.FLOAT, list(indata.shape)),
        helper.make_tensor_value_info("starts", TensorProto.INT64, list(starts.shape)),
        helper.make_tensor_value_info("ends", TensorProto.INT64, list(ends.shape)),
    ]
    initializer = [
        helper.make_tensor("starts", TensorProto.INT64, list(starts.shape), starts),
        helper.make_tensor("ends", TensorProto.INT64, list(ends.shape), ends),
    ]
    nodes = []

    if "add_noop_to_input_attrs" in attrs:

        def add_noop_to_input_attr(attr_name, attr):
            output_name = attr_name + "_output"

            ref_shape = list(np.array(attr).shape)
            ref_shape.insert(0, 1)
            ref_shape = tuple(ref_shape)
            ref_array = np.array(ref_shape)
            ref_node = onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=["ref_in_" + attr_name],
                value=onnx.helper.make_tensor(
                    name="const_tensor__1_" + attr_name,
                    data_type=onnx.TensorProto.INT64,
                    dims=ref_array.shape,
                    vals=ref_array.flatten().astype(int),
                ),
            )
            in_shape = np.array(attr).shape
            in_array = np.array(in_shape)
            ref_node2 = onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=["input_shape_" + attr_name],
                value=onnx.helper.make_tensor(
                    name="const_tensor__2_" + attr_name,
                    data_type=onnx.TensorProto.INT64,
                    dims=in_array.shape,
                    vals=in_array.flatten().astype(int),
                ),
            )

            reshape1_node = helper.make_node(
                "Reshape", [attr_name, "ref_in_" + attr_name], ["reshape_" + attr_name]
            )
            reshape2_node = helper.make_node(
                "Reshape", ["reshape_" + attr_name, "input_shape_" + attr_name], [output_name]
            )
            return [ref_node, ref_node2, reshape1_node, reshape2_node]

    slice_inputs = []
    for attr_name in ["starts", "ends", "axes", "steps"]:
        if attr_name not in attrs:
            continue
        if "add_noop_to_input_attrs" in attrs and attr_name in attrs["add_noop_to_input_attrs"]:
            nodes.extend(add_noop_to_input_attr(attr_name, attrs[attr_name]))
            slice_inputs.append(attr_name + "_output")
        else:
            slice_inputs.append(attr_name)

    if axes:
        axes = np.asarray(axes)
        inputs.append(helper.make_tensor_value_info("axes", TensorProto.INT32, list(axes.shape)))
        initializer.append(helper.make_tensor("axes", TensorProto.INT32, list(axes.shape), axes))

    if steps:
        assert axes is not None and len(axes) == len(steps)
        steps = np.asarray(steps)
        inputs.append(helper.make_tensor_value_info("steps", TensorProto.INT32, list(axes.shape)))
        initializer.append(helper.make_tensor("steps", TensorProto.INT32, list(steps.shape), steps))

    y = helper.make_node("Slice", ["data", *slice_inputs], ["out"])

    nodes.append(y)
    graph = helper.make_graph(
        nodes,
        "slice_test",
        inputs=inputs,
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))],
        initializer=initializer,
    )
    model = helper.make_model(graph, producer_name="slice_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output_with_vm(model, indata, target, ctx, opset=10, freeze_params=True)
        tvm.testing.assert_allclose(outdata, tvm_out)


# TODO(mbrookhart): enable once VM supports heterogenous execution
# @tvm.testing.uses_gpu
def test_slice():
    x = np.random.randn(20, 10, 5).astype(np.float32)
    _test_slice_iteration_v1(x, x[0:3, 0:10], starts=(0, 0), ends=(3, 10), axes=(0, 1))
    _test_slice_iteration_v1(x, x[:, :, 3:4], starts=(0, 0, 3), ends=(20, 10, 4))
    _test_slice_iteration_v1(x, x[:, 1:1000], starts=(1,), ends=(1000,), axes=(1,))
    _test_slice_iteration_v1(x, x[:, 0:-1], starts=(0,), ends=(-1,), axes=(1,))
    _test_slice_iteration_v10(x, x[0:3, 0:10], starts=(0, 0), ends=(3, 10), axes=(0, 1))
    _test_slice_iteration_v10(x, x[:, :, 3:4], starts=(0, 0, 3), ends=(20, 10, 4))
    _test_slice_iteration_v10(x, x[:, 1:1000], starts=(1,), ends=(1000,), axes=(1,))
    _test_slice_iteration_v10(x, x[:, 0:-1], starts=(0,), ends=(-1,), axes=(1,))
    _test_slice_iteration_v10(
        x,
        x[0:3, 0:10],
        starts=(0, 0),
        ends=(3, 10),
        axes=(0, 1),
        add_noop_to_input_attrs=["starts"],
    )
    _test_slice_iteration_v10(
        x, x[:, :, 3:4], starts=(0, 0, 3), ends=(20, 10, 4), add_noop_to_input_attrs=["ends"]
    )
    _test_slice_iteration_v10(
        x, x[:, 1:1000], starts=(1,), ends=(1000,), axes=(1,), add_noop_to_input_attrs=["axes"]
    )
    _test_slice_iteration_v10(
        x,
        x[:, 0:-1],
        starts=(0,),
        ends=(-1,),
        axes=(1,),
        add_noop_to_input_attrs=["starts", "ends"],
    )
    _test_slice_iteration_v10(
        x,
        x[0:3, 0:10],
        starts=(0, 0),
        ends=(3, 10),
        axes=(0, 1),
        add_noop_to_input_attrs=["ends", "axes"],
    )
    _test_slice_iteration_v10(
        x,
        x[:, :, 3:4],
        starts=(0, 0, 3),
        ends=(20, 10, 4),
        add_noop_to_input_attrs=["starts", "axes"],
    )
    _test_slice_iteration_v10(
        x,
        x[:, 1:1000],
        starts=(1,),
        ends=(1000,),
        axes=(1,),
        add_noop_to_input_attrs=["starts", "ends", "axes"],
    )
    x = np.random.randn(1, 1, 1, 128).astype(np.float32)
    _test_slice_iteration_v10(
        x, x, starts=(0, 0), ends=(9223372036854775807, 9223372036854775807), axes=(0, 3)
    )

    x = np.random.randn(4, 4).astype(np.float32)
    _test_slice_iteration_v10(
        x, x[:, 1::2], starts=(1,), ends=(9223372036854775807,), axes=(1,), steps=(2,)
    )
    _test_slice_iteration_v10(
        x,
        x[0::1, 1::2],
        starts=(0, 1),
        ends=(4, 4),
        axes=(0, 1),
        steps=(1, 2),
    )


def _test_onnx_op_elementwise(inshape, outfunc, npargs, dtype, opname, kwargs):
    indata = np.random.uniform(-1, 1, size=inshape).astype(dtype)
    outdata = outfunc(indata, **npargs)

    y = helper.make_node(opname, ["in"], ["out"], **kwargs)

    graph = helper.make_graph(
        [y],
        opname + "_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))],
    )

    model = helper.make_model(graph, producer_name=opname + "_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, indata, target, ctx, outdata.shape, dtype)
        tvm.testing.assert_allclose(outdata, tvm_out)


@tvm.testing.uses_gpu
def test_floor():
    _test_onnx_op_elementwise((2, 4, 5, 6), np.floor, {}, "float32", "Floor", {})


@tvm.testing.uses_gpu
def test_ceil():
    _test_onnx_op_elementwise((2, 4, 5, 6), np.ceil, {}, "float32", "Ceil", {})


@tvm.testing.uses_gpu
def test_clip():
    _test_onnx_op_elementwise(
        (2, 4, 5, 6),
        np.clip,
        {"a_min": -1.0, "a_max": 1.0},
        "float32",
        "Clip",
        {"min": -1.0, "max": 1.0},
    )


@tvm.testing.uses_gpu
def test_clip_min_max_as_inputs():
    input_shape = (2, 4, 5, 6)
    nodes = [
        make_constant_node("min", onnx.TensorProto.FLOAT, (), [0.0]),
        make_constant_node("max", onnx.TensorProto.FLOAT, (), [6.0]),
    ]
    input_names = ["in", "min", "max"]
    nodes.append(helper.make_node("Clip", inputs=input_names, outputs=["out"]))
    graph = helper.make_graph(
        nodes,
        "clip_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(input_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_shape))],
    )
    model = helper.make_model(graph, producer_name="clip_test")

    verify_with_ort(model, [input_shape], input_shape)


@tvm.testing.uses_gpu
def test_round():
    _test_onnx_op_elementwise((2, 4, 5, 6), np.round, {}, "float32", "Round", {})


def _test_finite_ops(inshape, outfunc, npargs, dtype, opname, kwargs):
    indata = np.random.choice(a=[np.nan, np.inf, -np.inf, 0.5, 1.0, 0], size=inshape).astype(dtype)

    outdata = outfunc(indata, **npargs)
    y = helper.make_node(opname, ["in"], ["out"], **kwargs)

    graph = helper.make_graph(
        [y],
        opname + "_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.BOOL, list(outdata.shape))],
    )

    model = helper.make_model(graph, producer_name=opname + "_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, indata, target, ctx, outdata.shape, dtype)
        tvm.testing.assert_allclose(outdata, tvm_out)


@tvm.testing.uses_gpu
def test_isinf():
    _test_finite_ops((2, 4, 5, 6), np.isinf, {}, "float32", "IsInf", {})


@tvm.testing.uses_gpu
def test_isnan():
    _test_finite_ops((2, 4, 5, 6), np.isnan, {}, "float32", "IsNaN", {})


def verify_gather_nd(in_shape, indices, dtype):
    x = np.random.uniform(size=in_shape).astype(dtype)
    indices = np.array(indices, dtype="int32")
    out_np = tvm.topi.testing.gather_nd_python(x, indices)

    y = helper.make_node("GatherND", ["in", "indices"], ["out"])

    graph = helper.make_graph(
        [y],
        "gather_test",
        inputs=[
            helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape)),
            helper.make_tensor_value_info("indices", TensorProto.INT32, list(indices.shape)),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_np.shape))],
    )
    model = helper.make_model(graph, producer_name="gather_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [x, indices], target, ctx, out_np.shape)
        tvm.testing.assert_allclose(out_np, tvm_out)


@tvm.testing.uses_gpu
def test_gather_nd():
    verify_gather_nd((2, 2), [[0, 0], [1, 1]], "int32")
    verify_gather_nd((3, 3, 3), [[0, 1], [1, 0]], "float32")
    verify_gather_nd((4, 3, 5, 6), [[2, 1, 0, 0]], "float32")


# TODO(mbrookhart): enable once VM supports heterogenous execution
# @tvm.testing.uses_gpu
def test_onehot():
    indices_shape = [10]
    indices_array = np.random.randint(low=0, high=9, size=indices_shape, dtype="int32")
    depth = 10
    values = np.asarray([0, 1]).astype("int32")
    out_np = np.eye(depth)[indices_array.reshape(-1)]

    onehot_node = helper.make_node("OneHot", ["indices", "depth", "values"], ["out"])

    graph = helper.make_graph(
        [onehot_node],
        "onehot_test",
        inputs=[
            helper.make_tensor_value_info("indices", TensorProto.INT32, indices_shape),
            helper.make_tensor_value_info("depth", TensorProto.INT32, [1]),
            helper.make_tensor_value_info("values", TensorProto.INT32, values.shape),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.INT32, out_np.shape)],
    )

    model = helper.make_model(graph, producer_name="onehot_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output_with_vm(
            model, [indices_array, np.array([depth]).astype("int32"), values], target, ctx
        )
        tvm.testing.assert_allclose(out_np, tvm_out, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_matmul():
    a_shape = (4, 3)
    b_shape = (3, 4)

    a_array = np.random.uniform(size=a_shape).astype("float32")
    b_array = np.random.uniform(size=b_shape).astype("float32")
    out_np = np.matmul(a_array, b_array)

    mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])

    graph = helper.make_graph(
        [mul_node],
        "matmul_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_np.shape))],
    )

    model = helper.make_model(graph, producer_name="matmul_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [a_array, b_array], target, ctx, out_np.shape)
        tvm.testing.assert_allclose(out_np, tvm_out, rtol=1e-5, atol=1e-5)


def verify_batch_matmul(a_shape, b_shape, out_shape, target, ctx):
    a_array = np.random.uniform(size=a_shape).astype("float32")
    b_array = np.random.uniform(size=b_shape).astype("float32")

    mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])

    graph = helper.make_graph(
        [mul_node],
        "matmul_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)],
    )

    model = helper.make_model(graph, producer_name="matmul_test")
    onnx_out = get_onnxruntime_output(model, [a_array, b_array], "float32")[0]

    tvm_out = get_tvm_output_with_vm(model, [a_array, b_array], target, ctx)
    tvm.testing.assert_allclose(onnx_out, tvm_out, rtol=1e-5, atol=1e-5)


# TODO(mbrookhart): enable cuda once VM supports heterogenous execution
@tvm.testing.parametrize_targets("llvm")
def test_batch_matmul(target, ctx):
    verify_batch_matmul((2, 3, 4, 3), (2, 3, 3, 4), (2, 3, 4, 4), target, ctx)
    verify_batch_matmul((2, 4, 3), (3, 4), (2, 4, 4), target, ctx)
    verify_batch_matmul((2, 3, 4, 3), (3, 4), (2, 3, 4, 4), target, ctx)
    # Test implicit broadcasting.
    verify_batch_matmul((4, 3), (2, 3, 4), (2, 4, 4), target, ctx)
    verify_batch_matmul((2, 4, 3), (1, 3, 4), (2, 4, 4), target, ctx)
    verify_batch_matmul((1, 4, 3), (2, 3, 4), (2, 4, 4), target, ctx)


def verify_simple_dynamic_model(a_shape, b_shape, target, ctx):
    def verify_model(ex, a_shape, b_shape):
        a_array = np.random.uniform(size=a_shape).astype("float32")
        b_array = np.random.uniform(size=b_shape).astype("float32")
        # matmul
        out_np = np.matmul(a_array, b_array)
        # relu
        out_np[out_np < 0] = 0

        tvm_out = ex.evaluate()(a_array, b_array).asnumpy()
        tvm.testing.assert_allclose(out_np, tvm_out, rtol=1e-5, atol=1e-5)

    mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])
    relu_node = helper.make_node("Relu", ["out"], ["relu"])

    a_array = np.random.uniform(size=a_shape).astype("float32")
    b_array = np.random.uniform(size=b_shape).astype("float32")
    # matmul
    out_np = np.matmul(a_array, b_array)

    graph = helper.make_graph(
        [mul_node, relu_node],
        "matmul_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
        ],
        outputs=[helper.make_tensor_value_info("relu", TensorProto.FLOAT, list(out_np.shape))],
    )

    model = helper.make_model(graph, producer_name="matmul_test")

    a_anys = [relay.Any()] * len(a_shape)
    b_anys = [relay.Any()] * len(b_shape)

    mod, params = relay.frontend.from_onnx(model, {"a": a_anys, "b": b_anys})

    ex = relay.create_executor("vm", mod=mod, ctx=ctx, target=target)
    verify_model(ex, a_shape, b_shape)
    verify_model(ex, [a * 2 for a in a_shape], [b * 2 for b in b_shape])
    verify_model(ex, [a * 3 for a in a_shape], [b * 3 for b in b_shape])


# TODO(mbrookhart): enable cuda once VM supports heterogenous execution
@tvm.testing.parametrize_targets("llvm")
def test_batch_matmul_dynamic_model(target, ctx):
    verify_simple_dynamic_model((2, 3, 4, 3), (2, 3, 3, 4), target, ctx)
    verify_simple_dynamic_model((2, 4, 3), (3, 4), target, ctx)
    verify_simple_dynamic_model((2, 3, 4, 3), (3, 4), target, ctx)


def verify_lrn(shape, nsize, dtype, alpha=None, beta=None, bias=None):
    in_array = np.random.uniform(size=shape).astype(dtype)

    if alpha == None and beta == None and bias == None:
        alpha = 0.0001
        beta = 0.75
        bias = 1.0
        node = onnx.helper.make_node("LRN", inputs=["in"], outputs=["out"], size=nsize)
    else:
        node = onnx.helper.make_node(
            "LRN", inputs=["in"], outputs=["out"], alpha=alpha, beta=beta, bias=bias, size=nsize
        )

    graph = helper.make_graph(
        [node],
        "lrn_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(shape))],
    )
    model = helper.make_model(graph, producer_name="lrn_test")

    def _get_python_lrn():
        square_sum = np.zeros(shape).astype(dtype)
        for n, c, h, w in np.ndindex(in_array.shape):
            square_sum[n, c, h, w] = sum(
                in_array[
                    n,
                    max(0, c - int(math.floor((nsize - 1) / 2))) : min(
                        5, c + int(math.ceil((nsize - 1) / 2)) + 1
                    ),
                    h,
                    w,
                ]
                ** 2
            )
        py_out = in_array / ((bias + (alpha / nsize) * square_sum) ** beta)
        return py_out

    for target, ctx in tvm.testing.enabled_targets():
        input_name = model.graph.input[0].name
        py_out = _get_python_lrn()
        tvm_out = get_tvm_output(model, in_array, target, ctx, py_out.shape, "float32")
        tvm.testing.assert_allclose(py_out, tvm_out, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_lrn():
    verify_lrn((5, 5, 5, 5), 3, "float32")
    verify_lrn((5, 5, 5, 5), 3, "float32", alpha=0.0002, beta=0.5, bias=2.0)


def verify_instance_norm(shape, axis=1):
    def _get_python_instance_norm(x, gamma, beta, epsilon=1e-5):
        dims_x = len(x.shape)
        axis = tuple(range(2, dims_x))
        mean = np.mean(x, axis=axis, keepdims=True)
        var = np.var(x, axis=axis, keepdims=True)
        dim_ones = (1,) * (dims_x - 2)
        gamma = gamma.reshape(-1, *dim_ones)
        beta = beta.reshape(-1, *dim_ones)
        return gamma * (x - mean) / np.sqrt(var + epsilon) + beta

    x = np.random.randn(*shape).astype(np.float32)
    gamma = np.random.randn(shape[1]).astype(np.float32)
    beta = np.random.randn(shape[1]).astype(np.float32)
    epsilon = 1e-5
    y = _get_python_instance_norm(x, gamma, beta, epsilon).astype(np.float32)

    node = onnx.helper.make_node(
        "InstanceNormalization",
        inputs=["x", "gamma", "beta"],
        outputs=["y"],
        epsilon=epsilon,
    )
    graph = helper.make_graph(
        [node],
        "instance_norm_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape)),
            helper.make_tensor_value_info("gamma", TensorProto.FLOAT, (shape[1],)),
            helper.make_tensor_value_info("beta", TensorProto.FLOAT, (shape[1],)),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))],
    )
    model = helper.make_model(graph, producer_name="instance_norm_test")
    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [x, gamma, beta], target, ctx, shape, "float32")
        tvm.testing.assert_allclose(y, tvm_out, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_instance_norm():
    verify_instance_norm((2, 3, 4, 5))
    verify_instance_norm((32, 64, 80, 64))
    verify_instance_norm((8, 6, 5))
    verify_instance_norm((8, 7, 6, 5, 4))


def _test_upsample_nearest():
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale)
    y = helper.make_node("Upsample", ["in"], ["out"], mode="nearest", scales=[1.0, 1.0, 2.0, 2.0])

    in_array = np.random.uniform(size=in_shape).astype(np.float32)
    out_array = tvm.topi.testing.upsampling_python(in_array, (scale, scale), "NCHW")

    graph = helper.make_graph(
        [y],
        "upsample_nearest_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_nearest_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, in_array, target, ctx, out_shape, "float32")
        tvm.testing.assert_allclose(out_array, tvm_out)


def _test_upsample3d_nearest():
    scale = 2
    in_shape = (1, 1, 3, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale, 3 * scale)
    y = helper.make_node(
        "Upsample", ["in"], ["out"], mode="nearest", scales=[1.0, 1.0, 2.0, 2.0, 2.0]
    )

    in_array = np.random.uniform(size=in_shape).astype(np.float32)
    out_array = tvm.topi.testing.upsampling3d_python(in_array, (scale, scale, scale), "NCDHW")

    graph = helper.make_graph(
        [y],
        "upsample_nearest_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_nearest_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, in_array, target, ctx, out_shape, "float32")
        tvm.testing.assert_allclose(out_array, tvm_out)


def _test_upsample_bilinear():
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale)
    y = helper.make_node("Upsample", ["in"], ["out"], mode="linear", scales=[1.0, 1.0, 2.0, 2.0])

    in_array = np.random.uniform(size=in_shape).astype(np.float32)
    out_array = tvm.topi.testing.bilinear_resize_python(in_array, (3 * scale, 3 * scale), "NCHW")

    graph = helper.make_graph(
        [y],
        "upsample_bilinear_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_bilinear_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, in_array, target, ctx, out_shape, "float32")
        tvm.testing.assert_allclose(out_array, tvm_out, rtol=1e-5, atol=1e-5)


def _test_upsample_bilinear_opset9():
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale)
    y = helper.make_node("Upsample", ["in", "scales"], ["out"], mode="linear")
    scales = [1, 1, 2, 2]
    in_array = np.random.uniform(size=in_shape).astype(np.float32)
    out_array = tvm.topi.testing.bilinear_resize_python(in_array, (3 * scale, 3 * scale), "NCHW")

    ref_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=scales,
            vals=np.random.random(scales).flatten().astype(float),
        ),
    )

    shape_node = helper.make_node("Shape", ["const"], ["scales"])

    graph = helper.make_graph(
        [ref_node, shape_node, y],
        "upsample_bilinear_opset9_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_bilinear_opset9_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output_with_vm(
            model, [in_array], target, ctx, opset=9, freeze_params=True
        )


def _test_upsample3d_trilinear():
    scale = 2
    in_shape = (1, 1, 3, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale, 3 * scale)
    y = helper.make_node("Upsample", ["in", "scales"], ["out"], mode="linear")
    scales = [1.0, 1.0, 2.0, 2.0, 2.0]
    in_array = np.random.uniform(size=in_shape).astype(np.float32)
    out_array = tvm.topi.testing.trilinear_resize3d_python(
        in_array,
        (3 * scale, 3 * scale, 3 * scale),
        "NCDHW",
        coordinate_transformation_mode="half_pixel",
    )

    ref_array = np.array(scales)
    ref_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["scales"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=ref_array.shape,
            vals=ref_array.flatten().astype(float),
        ),
    )

    graph = helper.make_graph(
        [ref_node, y],
        "upsample_trilinear_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_trilinear_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, in_array, target, ctx, out_shape, "float32")
        tvm.testing.assert_allclose(out_array, tvm_out, rtol=1e-5, atol=1e-5)


# TODO(mbrookhart): enable once VM supports heterogenous execution
# @tvm.testing.uses_gpu
def test_upsample():
    _test_upsample_nearest()
    _test_upsample_bilinear()
    _test_upsample_bilinear_opset9()
    _test_upsample3d_nearest()
    _test_upsample3d_trilinear()


def _test_softmax(inshape, axis):
    opname = "Softmax"
    indata = np.random.uniform(size=inshape).astype(np.float32)
    outshape = inshape
    outdata = tvm.topi.testing.softmax_python(indata)
    if isinstance(axis, int):
        y = helper.make_node(opname, ["in"], ["out"], axis=axis)
    elif axis is None:
        y = helper.make_node(opname, ["in"], ["out"])

    graph = helper.make_graph(
        [y],
        opname + "_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))],
    )

    model = helper.make_model(graph, producer_name=opname + "_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, indata, target, ctx, outshape, "float32")
        tvm.testing.assert_allclose(outdata, tvm_out, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_softmax():
    _test_softmax((1, 10), None)
    _test_softmax((1, 10), 1)


def verify_min(input_dim):
    dtype = "float32"

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.min((a_np1, a_np2, a_np3), axis=0)

    min_node = helper.make_node("Min", ["a_np1", "a_np2", "a_np3"], ["out"])

    graph = helper.make_graph(
        [min_node],
        "Min_test",
        inputs=[
            helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim)),
            helper.make_tensor_value_info("a_np2", TensorProto.FLOAT, list(input_dim)),
            helper.make_tensor_value_info("a_np3", TensorProto.FLOAT, list(input_dim)),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(b_np.shape))],
    )

    model = helper.make_model(graph, producer_name="Min_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [a_np1, a_np2, a_np3], target, ctx, b_np.shape)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_forward_min():
    verify_min((1, 3, 20, 20))
    verify_min((20, 20))


def verify_max(input_dim):
    dtype = "float32"

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.max((a_np1, a_np2, a_np3), axis=0)

    max_node = helper.make_node("Max", ["a_np1", "a_np2", "a_np3"], ["out"])

    graph = helper.make_graph(
        [max_node],
        "Max_test",
        inputs=[
            helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim)),
            helper.make_tensor_value_info("a_np2", TensorProto.FLOAT, list(input_dim)),
            helper.make_tensor_value_info("a_np3", TensorProto.FLOAT, list(input_dim)),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(b_np.shape))],
    )

    model = helper.make_model(graph, producer_name="Max_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [a_np1, a_np2, a_np3], target, ctx, b_np.shape)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_forward_max():
    verify_max((1, 3, 20, 20))
    verify_max((20, 20))


def verify_mean(input_dim):
    dtype = "float32"

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.mean((a_np1, a_np2, a_np3), axis=0)

    mean_node = helper.make_node("Mean", ["a_np1", "a_np2", "a_np3"], ["out"])

    graph = helper.make_graph(
        [mean_node],
        "Mean_test",
        inputs=[
            helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim)),
            helper.make_tensor_value_info("a_np2", TensorProto.FLOAT, list(input_dim)),
            helper.make_tensor_value_info("a_np3", TensorProto.FLOAT, list(input_dim)),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(b_np.shape))],
    )

    model = helper.make_model(graph, producer_name="Mean_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [a_np1, a_np2, a_np3], target, ctx, b_np.shape)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_forward_mean():
    verify_mean((1, 3, 20, 20))
    verify_mean((20, 20))


def verify_hardsigmoid(input_dim, alpha, beta):
    dtype = "float32"

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.clip(a_np1 * alpha + beta, 0, 1)

    hardsigmoid_node = helper.make_node("HardSigmoid", ["a_np1"], ["out"], alpha=alpha, beta=beta)

    graph = helper.make_graph(
        [hardsigmoid_node],
        "HardSigmoid_test",
        inputs=[helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(b_np.shape))],
    )

    model = helper.make_model(graph, producer_name="HardSigmoid_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [a_np1], target, ctx, b_np.shape)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_forward_hardsigmoid():
    verify_hardsigmoid((1, 3, 20, 20), 0.5, 0.6)
    verify_hardsigmoid((20, 20), 0.3, 0.4)


def verify_argmin(input_dim, axis=None, keepdims=None):
    def _argmin_numpy(data, axis=0, keepdims=True):
        result = np.argmin(data, axis=axis)
        if keepdims == 1:
            result = np.expand_dims(result, axis)
        return result.astype(data.dtype)

    a_np1 = np.random.uniform(-10, 10, input_dim).astype(np.int32)
    if keepdims is None and axis is None:
        b_np = _argmin_numpy(a_np1)
        node = onnx.helper.make_node("ArgMin", inputs=["a_np1"], outputs=["out"])
    elif axis is None:
        b_np = _argmin_numpy(a_np1, keepdims=keepdims)
        node = onnx.helper.make_node("ArgMin", inputs=["a_np1"], outputs=["out"], keepdims=keepdims)
    elif keepdims is None:
        b_np = _argmin_numpy(a_np1, axis=axis)
        node = onnx.helper.make_node("ArgMin", inputs=["a_np1"], outputs=["out"], axis=axis)
    else:
        b_np = _argmin_numpy(a_np1, axis=axis, keepdims=keepdims)
        node = onnx.helper.make_node(
            "ArgMin", inputs=["a_np1"], outputs=["out"], axis=axis, keepdims=keepdims
        )
    graph = helper.make_graph(
        [node],
        "argmin_test",
        inputs=[helper.make_tensor_value_info("a_np1", TensorProto.INT32, list(a_np1.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.INT32, list(b_np.shape))],
    )

    model = helper.make_model(graph, producer_name="argmin_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [a_np1], target, ctx, b_np.shape, b_np.dtype)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)


def verify_argmax(input_dim, axis=None, keepdims=None):
    def _argmax_numpy(data, axis=0, keepdims=True):
        result = np.argmax(data, axis=axis)
        if keepdims == 1:
            result = np.expand_dims(result, axis)
        return result.astype(data.dtype)

    a_np1 = np.random.uniform(-10, 10, input_dim).astype(np.int32)
    if keepdims is None and axis is None:
        b_np = _argmax_numpy(a_np1)
        node = onnx.helper.make_node("ArgMax", inputs=["a_np1"], outputs=["out"])
    elif axis is None:
        b_np = _argmax_numpy(a_np1, keepdims=keepdims)
        node = onnx.helper.make_node("ArgMax", inputs=["a_np1"], outputs=["out"], keepdims=keepdims)
    elif keepdims is None:
        b_np = _argmax_numpy(a_np1, axis=axis)
        node = onnx.helper.make_node("ArgMax", inputs=["a_np1"], outputs=["out"], axis=axis)
    else:
        b_np = _argmax_numpy(a_np1, axis=axis, keepdims=keepdims)
        node = onnx.helper.make_node(
            "ArgMax", inputs=["a_np1"], outputs=["out"], axis=axis, keepdims=keepdims
        )

    graph = helper.make_graph(
        [node],
        "argmax_test",
        inputs=[helper.make_tensor_value_info("a_np1", TensorProto.INT32, list(a_np1.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.INT32, list(b_np.shape))],
    )

    model = helper.make_model(graph, producer_name="argmax_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [a_np1], target, ctx, b_np.shape, b_np.dtype)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_forward_arg_min_max():
    """Verify argmin and argmax"""
    verify_argmin([3, 4, 4])
    verify_argmax([3, 4, 4])
    verify_argmin([3, 4, 4], axis=1)
    verify_argmax([3, 4, 4], axis=0)
    verify_argmin([3, 4, 4], keepdims=0)
    verify_argmax([3, 4, 4], keepdims=1)
    for axis in [None, 0, 1, 2]:
        for keepdims in [None, True, False]:
            verify_argmin([3, 4, 4], axis, keepdims)
            verify_argmax([3, 4, 4], axis, keepdims)


def verify_constantofshape(input_dim, value, dtype):
    out = np.empty(shape=input_dim, dtype=dtype)
    out.fill(value)

    fill_node = helper.make_node(
        "ConstantOfShape",
        ["input"],
        ["output"],
        value=helper.make_tensor(
            "value", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], (1,), (value,)
        ),
    )

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_dim)]

    graph = helper.make_graph(
        [fill_node],
        "fill_test",
        inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, list(out.shape))],
    )

    model = helper.make_model(graph, producer_name="fill_test")

    for target, ctx in tvm.testing.enabled_targets():
        input_np = np.array(input_dim).astype("float32")
        tvm_out = get_tvm_output_with_vm(model, [input_np], target, ctx)

        tvm.testing.assert_allclose(out, tvm_out, rtol=1e-5, atol=1e-5)


# TODO(mbrookhart): enable once VM supports heterogenous execution
# @tvm.testing.uses_gpu
def test_constantofshape():
    verify_constantofshape((2, 3, 4, 5), 10, "float32")
    verify_constantofshape((3, 3), 0, "int32")
    verify_constantofshape((1, 2, 3), -1, "float32")


def verify_pad(indata, pads, mode="constant", value=0.0):
    indata = np.array(indata).astype(np.float32)
    #  numpy expect result
    len_dim = len(pads) // 2
    np_pads = [(pads[i], pads[i + len_dim]) for i in range(len_dim)]
    #  onnx graph
    if mode in ["edge", "reflect"]:
        outdata = np.pad(indata, pad_width=np_pads, mode=mode)
        node = helper.make_node(
            "Pad",
            inputs=["input"],
            outputs=["output"],
            mode=mode,
            pads=pads,
        )
    else:
        outdata = np.pad(indata, pad_width=np_pads, mode="constant", constant_values=value)
        node = helper.make_node(
            "Pad", inputs=["input"], outputs=["output"], mode="constant", pads=pads, value=value
        )
    graph = helper.make_graph(
        [node],
        "pad_test",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape))],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))],
    )
    model = helper.make_model(graph, producer_name="pad_test")
    #  tvm result
    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, indata, target, ctx, outdata.shape, "float32", opset=2)
    tvm.testing.assert_allclose(outdata, tvm_out, rtol=1e-5, atol=1e-5)


def verify_pad_v11(indata, pads, mode="constant", value=0.0):
    indata = np.array(indata).astype(np.float32)
    #  numpy expect result
    len_dim = len(pads) // 2
    np_pads = [(pads[i], pads[i + len_dim]) for i in range(len_dim)]
    pads = np.array(pads)
    #  onnx graph
    if mode in ["edge", "reflect"]:
        inputs = [indata, pads]
        outdata = np.pad(indata, pad_width=np_pads, mode=mode)
        node = helper.make_node("Pad", inputs=["input", "pads"], outputs=["output"], mode=mode)
        graph = helper.make_graph(
            [node],
            "pad_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape)),
                helper.make_tensor_value_info("pads", TensorProto.INT64, (len(pads),)),
            ],
            initializer=[helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads)],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
            ],
        )
    else:
        inputs = [indata, pads, np.array([value]).astype("float32")]
        outdata = np.pad(indata, pad_width=np_pads, mode="constant", constant_values=value)
        node = helper.make_node(
            "Pad", inputs=["input", "pads", "constant_value"], outputs=["output"], mode="constant"
        )
        graph = helper.make_graph(
            [node],
            "pad_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape)),
                helper.make_tensor_value_info("pads", TensorProto.INT64, (len(pads),)),
                helper.make_tensor_value_info("constant_value", TensorProto.FLOAT, (1,)),
            ],
            initializer=[
                helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads),
                helper.make_tensor("constant_value", TensorProto.FLOAT, (1,), [value]),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
            ],
        )
    model = helper.make_model(graph, producer_name="pad_test")
    #  tvm result
    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output_with_vm(model, inputs, target, ctx, opset=11, freeze_params=False)
    tvm.testing.assert_allclose(outdata, tvm_out, rtol=1e-5, atol=1e-5)


# TODO(mbrookhart): enable once VM supports heterogenous execution
# @tvm.testing.uses_gpu
def test_pad():
    verify_pad(np.random.randn(2, 2).astype(np.float32), [0, 1, 0, 0], "constant", 0.0)
    verify_pad(np.random.randn(2, 3).astype(np.float32), [1, 0, 0, 1], "constant", 0.0)
    verify_pad(np.random.randn(3, 2).astype(np.float32), [0, 0, 1, 0], "constant", 5.0)
    verify_pad(np.random.randn(1, 3, 4, 5).astype(np.float32), [0, 0, 1, 1, 0, 0, 1, 1], "edge")
    verify_pad(np.random.randn(1, 3, 4, 5).astype(np.float32), [0, 0, 1, 1, 0, 0, 1, 1], "reflect")

    verify_pad_v11(np.random.randn(2, 2).astype(np.float32), [0, 1, 0, 0], "constant", 0.0)
    verify_pad_v11(np.random.randn(2, 3).astype(np.float32), [1, 0, 0, 1], "constant", 0.0)
    verify_pad_v11(np.random.randn(3, 2).astype(np.float32), [0, 0, 1, 0], "constant", 5.0)
    verify_pad_v11(np.random.randn(1, 3, 4, 5).astype(np.float32), [0, 0, 1, 1, 0, 0, 1, 1], "edge")
    verify_pad_v11(
        np.random.randn(1, 3, 4, 5).astype(np.float32), [0, 0, 1, 1, 0, 0, 1, 1], "reflect"
    )


def verify_reduce_func(func, data, axis, keepdims):
    inshape = data.shape
    outshape = np.sum(data, axis=axis, keepdims=keepdims == 1).shape

    if axis:
        node = onnx.helper.make_node(
            func, inputs=["x"], outputs=["y"], axes=axis, keepdims=keepdims
        )
    else:
        node = onnx.helper.make_node(func, inputs=["x"], outputs=["y"], keepdims=keepdims)

    graph = helper.make_graph(
        [node],
        "reduce_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(inshape))],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(outshape))],
    )

    model = helper.make_model(graph, producer_name="reduce_test")

    verify_with_ort_with_inputs(model, [data], outshape)


@tvm.testing.uses_gpu
def test_all_reduce_funcs():
    funcs = [
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceL1",
        "ReduceL2",
    ]

    for func in funcs:
        for keepdims in [True, False]:
            verify_reduce_func(
                func, np.random.randn(3, 2, 2).astype(np.float32), axis=None, keepdims=keepdims
            )

            verify_reduce_func(
                func, np.random.randn(3, 2, 3).astype(np.float32), axis=None, keepdims=keepdims
            )

            verify_reduce_func(
                func, np.random.randn(3, 3, 3).astype(np.float32), axis=(1,), keepdims=keepdims
            )

            verify_reduce_func(
                func, np.random.randn(3, 3, 3, 1).astype(np.float32), axis=(1, 2), keepdims=keepdims
            )

            verify_reduce_func(
                func, np.random.randn(3, 3, 3, 1).astype(np.float32), axis=(1,), keepdims=keepdims
            )

            verify_reduce_func(
                func, np.random.randn(1, 3, 4, 1).astype(np.float32), axis=(1,), keepdims=keepdims
            )


def verify_split(indata, outdatas, split, axis=0, pass_split=True):
    indata = np.array(indata).astype(np.float32)
    outdatas = [np.array(o).astype(np.float32) for o in outdatas]
    if split:
        split_index = range(len(split))
    else:
        split_index = range(len(outdatas))
    if pass_split:
        node = helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["output_{}".format(i) for i in range(len(split_index))],
            axis=axis,
            split=split,
        )
    else:
        node = helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["output_{}".format(i) for i in range(len(split_index))],
            axis=axis,
        )
    graph = helper.make_graph(
        [node],
        "split_test",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape))],
        outputs=[
            helper.make_tensor_value_info(
                "output_{}".format(i), TensorProto.FLOAT, list(outdatas[i].shape)
            )
            for i in range(len(split_index))
        ],
    )
    model = helper.make_model(graph, producer_name="split_test")

    import onnxruntime.backend

    rep = onnxruntime.backend.prepare(model, "CPU")
    onnx_out = rep.run(indata)

    for target, ctx in tvm.testing.enabled_targets():
        output_shape = [o.shape for o in outdatas]
        output_type = ["float32", "float32", "float32"]
        tvm_out = get_tvm_output(model, indata, target, ctx, output_shape, output_type)
        for o, t in zip(onnx_out, tvm_out):
            tvm.testing.assert_allclose(o, t)


@tvm.testing.uses_gpu
def test_split():
    # 1D
    verify_split([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [2, 2, 2], 0)
    verify_split(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [2, 2, 2], 0, False
    )
    verify_split([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]], [2, 1, 3], 0)
    # 2D
    verify_split(
        [[1.0, 2.0, 3.0, 4.0], [7.0, 8.0, 9.0, 10.0]],
        [[[1.0, 2.0], [7.0, 8.0]], [[3.0, 4.0], [9.0, 10.0]]],
        [2, 2],
        1,
    )
    # Split evenly (unstack)
    verify_split([1, 2, 3], [[1], [2], [3]], False, 0, False)


@tvm.testing.uses_gpu
def test_binary_ops():
    in_shape = (1, 2, 3, 3)
    dtype = "float32"
    out_shape = in_shape

    def verify_binary_ops(op, x, y, out_np, x_name="in1", y_name="in2", broadcast=None):
        if broadcast is None:
            z = helper.make_node(op, [x_name, y_name], ["out"])
        else:
            z = helper.make_node(op, [x_name, y_name], ["out"], broadcast=1)
        graph = helper.make_graph(
            [z],
            "_test",
            inputs=[
                helper.make_tensor_value_info(x_name, TensorProto.FLOAT, list(in_shape)),
                helper.make_tensor_value_info(y_name, TensorProto.FLOAT, list(in_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
        )
        model = helper.make_model(graph, producer_name="_test")
        for target, ctx in tvm.testing.enabled_targets():
            tvm_out = get_tvm_output(model, [x, y], target, ctx)
            tvm.testing.assert_allclose(out_np, tvm_out, rtol=1e-5, atol=1e-5)

    x = np.random.uniform(size=in_shape).astype(dtype)
    y = np.random.uniform(size=in_shape).astype(dtype)
    z = np.random.uniform(size=(3,)).astype(dtype)
    verify_binary_ops("Add", x, y, x + y, broadcast=None)
    verify_binary_ops("Add", x, z, x + z, broadcast=True)
    verify_binary_ops("Sub", x, y, x - y, broadcast=None)
    verify_binary_ops("Sub", x, z, x - z, broadcast=True)
    verify_binary_ops("Mul", x, y, x * y, broadcast=None)
    verify_binary_ops("Mul", x, z, x * z, broadcast=True)
    verify_binary_ops("Mul", x, x, x * x, x_name="in1", y_name="in1", broadcast=None)
    verify_binary_ops("Div", x, y, x / y, broadcast=None)
    verify_binary_ops("Div", x, z, x / z, broadcast=True)
    verify_binary_ops("Sum", x, y, x + y, broadcast=None)
    verify_binary_ops("Greater", x, y, x > y, broadcast=True)
    verify_binary_ops("Less", x, y, x < y, broadcast=True)
    verify_binary_ops("Equal", x, y, x == y, broadcast=True)


@tvm.testing.uses_gpu
def test_single_ops():
    in_shape = (1, 2, 3, 3)
    dtype = "float32"
    out_shape = in_shape

    def verify_single_ops(op, x, out_np, rtol=1e-5, atol=1e-5):
        z = helper.make_node(op, ["in1"], ["out"])
        graph = helper.make_graph(
            [z],
            "_test",
            inputs=[
                helper.make_tensor_value_info("in1", TensorProto.FLOAT, list(in_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
        )
        model = helper.make_model(graph, producer_name="_test")
        for target, ctx in tvm.testing.enabled_targets():
            tvm_out = get_tvm_output(model, [x], target, ctx)
            tvm.testing.assert_allclose(out_np, tvm_out, rtol=rtol, atol=atol)

    x = np.random.uniform(size=in_shape).astype(dtype)
    verify_single_ops("Neg", x, -x)
    verify_single_ops("Abs", x, np.abs(x))
    verify_single_ops("Reciprocal", x, 1 / x)
    verify_single_ops("Sqrt", x, np.sqrt(x))
    verify_single_ops("Relu", x, np.maximum(x, 0))
    verify_single_ops("Exp", x, np.exp(x))
    verify_single_ops("Log", x, np.log(x))
    verify_single_ops("Log", x, np.log(x))
    verify_single_ops("ACos", x, np.arccos(x))
    verify_single_ops("ACosh", x, np.arccosh(x))
    verify_single_ops("ASin", x, np.arcsin(x))
    verify_single_ops("ASinh", x, np.arcsinh(x))
    verify_single_ops("ATan", x, np.arctan(x))
    verify_single_ops("ATanh", x, np.arctanh(x))
    verify_single_ops("Cos", x, np.cos(x))
    verify_single_ops("Cosh", x, np.cosh(x))
    verify_single_ops("Sin", x, np.sin(x))
    verify_single_ops("Sinh", x, np.sinh(x))
    verify_single_ops("Tan", x, np.tan(x))
    verify_single_ops("Tanh", x, np.tanh(x))
    verify_single_ops("Sigmoid", x, 1 / (1 + np.exp(-x)))
    verify_single_ops("Softsign", x, x / (1 + np.abs(x)))
    verify_single_ops("SoftPlus", x, np.log(1 + np.exp(x)))


@tvm.testing.uses_gpu
def test_leaky_relu():
    def leaky_relu_x(x, alpha):
        return np.where(x >= 0, x, x * alpha)

    _test_onnx_op_elementwise(
        (2, 4, 5, 6), leaky_relu_x, {"alpha": 0.25}, "float32", "LeakyRelu", {"alpha": 0.25}
    )


@tvm.testing.uses_gpu
def test_elu():
    def elu_x(x, alpha):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1.0))

    _test_onnx_op_elementwise(
        (2, 4, 5, 6), elu_x, {"alpha": 0.25}, "float32", "Elu", {"alpha": 0.25}
    )


@tvm.testing.uses_gpu
def test_selu():
    def selu_x(x, alpha, gamma):
        return gamma * np.where(x > 0, x, alpha * (np.exp(x) - 1.0))

    _test_onnx_op_elementwise(
        (2, 4, 5, 6),
        selu_x,
        {"alpha": 0.25, "gamma": 0.3},
        "float32",
        "Selu",
        {"alpha": 0.25, "gamma": 0.3},
    )


@tvm.testing.uses_gpu
def test_prelu():
    def verify_prelu(x_shape, a_shape):
        node = helper.make_node("PRelu", inputs=["X", "slope"], outputs=["Y"])

        graph = helper.make_graph(
            [node],
            "prelu_test",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x_shape)),
                helper.make_tensor_value_info("slope", TensorProto.FLOAT, list(a_shape)),
            ],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(x_shape))],
        )

        model = helper.make_model(graph, producer_name="prelu_test")

        verify_with_ort(model, [x_shape, a_shape], list(x_shape))

    verify_prelu([3, 4, 5, 6], [1, 4, 1, 1])
    verify_prelu([1, 8, 5, 6], [1, 8, 1, 1])
    verify_prelu([2, 12, 16, 16], [1, 12, 1, 1])
    verify_prelu([2, 12, 16, 16], [1])  # Test alpha broadcasting.


@tvm.testing.uses_gpu
def test_ThresholdedRelu():
    def ThresholdedRelu_x(x, alpha):
        out_np = np.clip(x, alpha, np.inf)
        out_np[out_np == alpha] = 0
        return out_np

    _test_onnx_op_elementwise(
        (2, 4, 5, 6),
        ThresholdedRelu_x,
        {"alpha": 0.25},
        "float32",
        "ThresholdedRelu",
        {"alpha": 0.25},
    )


@tvm.testing.uses_gpu
def test_ScaledTanh():
    def ScaledTanh_x(x, alpha, beta):
        return alpha * np.tanh(beta * x)

    _test_onnx_op_elementwise(
        (2, 4, 5, 6),
        ScaledTanh_x,
        {"alpha": 0.25, "beta": 0.3},
        "float32",
        "ScaledTanh",
        {"alpha": 0.25, "beta": 0.3},
    )


@tvm.testing.uses_gpu
def test_ParametricSoftplus():
    def ParametricSoftplus_x(x, alpha, beta):
        return alpha * np.log(np.exp(beta * x) + 1)

    _test_onnx_op_elementwise(
        (2, 4, 5, 6),
        ParametricSoftplus_x,
        {"alpha": 0.25, "beta": 0.3},
        "float32",
        "ParametricSoftplus",
        {"alpha": 0.25, "beta": 0.3},
    )


@tvm.testing.uses_gpu
def test_Scale():
    def Scale_x(x, scale):
        return scale * x

    _test_onnx_op_elementwise(
        (2, 4, 5, 6), Scale_x, {"scale": 0.25}, "float32", "Scale", {"scale": 0.25}
    )


@tvm.testing.uses_gpu
def test_LogSoftmax():
    _test_onnx_op_elementwise(
        (1, 4), tvm.topi.testing.log_softmax_python, {}, "float32", "LogSoftmax", {"axis": 1}
    )


def check_torch_conversion(model, input_size):
    dummy_input = torch.randn(*input_size)
    file_name = "{}.onnx".format(model.__name__)
    # Set verbose=True for more output
    torch.onnx.export(model(), dummy_input, file_name, export_params=True, verbose=False)
    onnx_model = onnx.load(file_name)
    input_data = np.random.uniform(size=input_size).astype("int32")
    verify_with_ort_with_inputs(onnx_model, [input_data])


@tvm.testing.uses_gpu
def test_resnet():
    check_torch_conversion(torchvision.models.resnet18, (1, 3, 224, 224))
    # check_torch_conversion(torchvision.models.resnet101, (1,3,224,224))


# def test_alexnet():
# Torch's ONNX export does not support the adaptive pooling used by AlexNet?
# check_torch_conversion(torchvision.models.alexnet, (1,3,224,224))

# Torch's ONNX export does not support the adaptive pooling used by vgg16?
# def test_vgg16():
#     check_torch_conversion(torchvision.models.vgg16, (1,3,224,224))

# TODO(@jroesch): Update Torch + ONNX to support this import.
# def test_squeezenet():
#     # Torch's ONNX export does not support the max pooling used by Squezenet
#     check_torch_conversion(torchvision.models.squeezenet1_0, (1,3,224,224))


@tvm.testing.uses_gpu
def test_densenet():
    check_torch_conversion(torchvision.models.densenet161, (1, 3, 224, 224))


@tvm.testing.uses_gpu
def test_inception():
    check_torch_conversion(torchvision.models.inception_v3, (1, 3, 224, 224))


# TODO(@jroesch): Update Torch + ONNX to support this import.
# def test_googlenet():
#     check_torch_conversion(torchvision.models.googlenet, (1,3,224,224))

# TODO(@jroesch): Update Torch + ONNX to support this import.
# def test_shufflenetv2():
#     check_torch_conversion(torchvision.models.shufflenetv2, (1,3,224,224))


@tvm.testing.uses_gpu
def test_sign():
    def Sign_x(x):
        return np.sign(x)

    _test_onnx_op_elementwise((3, 4, 5, 6), Sign_x, {}, "float32", "Sign", {})


def verify_not(indata, dtype):
    x = indata.astype(dtype)
    outdata = np.logical_not(x)

    node = helper.make_node(
        "Not",
        inputs=["in"],
        outputs=["out"],
    )

    graph = helper.make_graph(
        [node],
        "not_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.BOOL, list(x.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.BOOL, list(outdata.shape))],
    )

    model = helper.make_model(graph, producer_name="not_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [x], target, ctx, outdata.shape)
        tvm.testing.assert_allclose(outdata, tvm_out)


@tvm.testing.uses_gpu
def test_not():
    # 2d
    verify_not(indata=(np.random.randn(3, 4) > 0), dtype=bool)
    # 3d
    verify_not(indata=(np.random.randn(3, 4, 5) > 0), dtype=bool)
    # 4d
    verify_not(indata=(np.random.randn(3, 4, 5, 6) > 0), dtype=bool)


def verify_and(indata, dtype):
    x = indata[0].astype(dtype)
    y = indata[1].astype(dtype)
    outdata = np.logical_and(x, y)

    node = helper.make_node(
        "And",
        inputs=["in1", "in2"],
        outputs=["out"],
    )

    graph = helper.make_graph(
        [node],
        "and_test",
        inputs=[
            helper.make_tensor_value_info("in1", TensorProto.BOOL, list(x.shape)),
            helper.make_tensor_value_info("in2", TensorProto.BOOL, list(y.shape)),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.BOOL, list(outdata.shape))],
    )

    model = helper.make_model(graph, producer_name="and_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [x, y], target, ctx, outdata.shape)
        tvm.testing.assert_allclose(outdata, tvm_out)


@tvm.testing.uses_gpu
def test_and():
    # 2d
    x = np.random.randn(3, 4) > 0
    y = np.random.randn(3, 4) > 0
    verify_and(indata=[x, y], dtype=bool)

    # 3d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(3, 4, 5) > 0
    verify_and(indata=[x, y], dtype=bool)

    # 4d
    x = np.random.randn(3, 4, 5, 6) > 0
    y = np.random.randn(3, 4, 5, 6) > 0
    verify_and(indata=[x, y], dtype=bool)

    # 3d vs 1d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(5) > 0
    verify_and(indata=[x, y], dtype=bool)

    # 3d vs 2d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(4, 5) > 0
    verify_and(indata=[x, y], dtype=bool)


def verify_tile_v1(indata, outdata, **kwargs):
    node = helper.make_node("Tile", inputs=["in"], outputs=["out"], **kwargs)
    graph = helper.make_graph(
        [node],
        "tile_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))],
    )

    model = helper.make_model(graph, producer_name="tile_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [indata], target, ctx, outdata.shape, opset=1)
        tvm.testing.assert_allclose(outdata, tvm_out)


def verify_tile_v6(indata, repeats, outdata):
    node = helper.make_node("Tile", inputs=["input", "repeats"], outputs=["out"])
    graph = helper.make_graph(
        [node],
        "tile_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape)),
            helper.make_tensor_value_info("repeats", TensorProto.INT64, list(repeats.shape)),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))],
    )

    model = helper.make_model(graph, producer_name="tile_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output_with_vm(model, [indata, repeats], target, ctx, opset=6)
        tvm.testing.assert_allclose(outdata, tvm_out)


# TODO(mbrookhart): enable once VM supports heterogenous execution
# @tvm.testing.uses_gpu
def test_tile():
    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    repeats = np.random.randint(low=1, high=10, size=(np.ndim(x),)).astype(np.int64)
    z = np.tile(x, repeats)
    verify_tile_v1(x, z, repeats=repeats)
    verify_tile_v6(x, repeats, z)


def verify_erf(indata, outdata):
    node = helper.make_node("Erf", inputs=["in"], outputs=["out"])
    graph = helper.make_graph(
        [node],
        "erf_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))],
    )
    model = helper.make_model(graph, producer_name="erf_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [indata], target, ctx, outdata.shape)
        tvm.testing.assert_allclose(outdata, tvm_out)


@tvm.testing.uses_gpu
def test_erf():
    x = np.random.rand(2, 3, 4, 6).astype(np.float32)
    z = scipy.special.erf(x)
    verify_erf(x, z)


def verify_where(condition, x, y, dtype, outdata):
    node = helper.make_node("Where", inputs=["condition", "x", "y"], outputs=["out"])
    graph = helper.make_graph(
        [node],
        "where_test",
        inputs=[
            helper.make_tensor_value_info("condition", TensorProto.BOOL, list(condition.shape)),
            helper.make_tensor_value_info("x", dtype, list(x.shape)),
            helper.make_tensor_value_info("y", dtype, list(y.shape)),
        ],
        outputs=[helper.make_tensor_value_info("out", dtype, list(outdata.shape))],
    )
    model = helper.make_model(graph, producer_name="where_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [condition, x, y], target, ctx, outdata.shape)
        tvm.testing.assert_allclose(outdata, tvm_out)


@tvm.testing.uses_gpu
def test_where():
    condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[9, 8], [7, 6]], dtype=np.int64)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.INT64, outdata)

    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[9, 8], [7, 6]], dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)

    x = np.array(1, dtype=np.float32)
    y = np.array([2], dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)

    x = np.array([2], dtype=np.float32)
    y = np.array(1, dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)

    condition = np.array(1, dtype=np.bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[5, 6], [7, 8]], dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)

    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[1], [7]], dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)


def verify_or(indata, dtype):
    x = indata[0].astype(dtype)
    y = indata[1].astype(dtype)
    outdata = np.logical_or(x, y)

    node = helper.make_node(
        "Or",
        inputs=["in1", "in2"],
        outputs=["out"],
    )

    graph = helper.make_graph(
        [node],
        "or_test",
        inputs=[
            helper.make_tensor_value_info("in1", TensorProto.BOOL, list(x.shape)),
            helper.make_tensor_value_info("in2", TensorProto.BOOL, list(y.shape)),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.BOOL, list(outdata.shape))],
    )

    model = helper.make_model(graph, producer_name="or_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [x, y], target, ctx, outdata.shape)
        tvm.testing.assert_allclose(outdata, tvm_out)


@tvm.testing.uses_gpu
def test_or():
    # 2d
    x = np.random.randn(3, 4) > 0
    y = np.random.randn(3, 4) > 0
    verify_or(indata=[x, y], dtype=bool)

    # 3d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(3, 4, 5) > 0
    verify_or(indata=[x, y], dtype=bool)

    # 4d
    x = np.random.randn(3, 4, 5, 6) > 0
    y = np.random.randn(3, 4, 5, 6) > 0
    verify_or(indata=[x, y], dtype=bool)

    # 3d vs 1d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(5) > 0
    verify_or(indata=[x, y], dtype=bool)

    # 3d vs 2d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(4, 5) > 0
    verify_or(indata=[x, y], dtype=bool)


@tvm.testing.uses_gpu
def test_batch_norm():
    def verify_batch_norm(in_shape):
        batchnorm = onnx.helper.make_node(
            "BatchNormalization", inputs=["x", "scale", "B", "mean", "var"], outputs=["Y"]
        )

        graph = helper.make_graph(
            [batchnorm],
            "batchnorm_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(in_shape)),
                helper.make_tensor_value_info("scale", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("mean", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("var", TensorProto.FLOAT, [in_shape[1]]),
            ],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(in_shape))],
        )

        model = helper.make_model(graph, producer_name="batchnorm_test")
        # X, scale, b, mean, var
        inshapes = [in_shape, in_shape[1], in_shape[1], in_shape[1], in_shape[1]]
        verify_with_ort(model, inshapes, in_shape)

    verify_batch_norm([1, 3, 224, 224])
    verify_batch_norm([1, 3, 24, 24])
    verify_batch_norm([16, 3, 24, 24])
    verify_batch_norm([16, 16, 24, 24])
    verify_batch_norm([16, 16, 10, 10])


# TODO(mbrookhart): enable once VM supports heterogenous execution
# @tvm.testing.uses_gpu
def test_batch_norm_dynamic_subgraph():
    def verify_batch_norm_dynamic_subgraph(in_shape, o_shape):

        batchnorm = onnx.helper.make_node(
            "BatchNormalization", inputs=["x", "scale", "B", "mean", "var"], outputs=["Y"]
        )

        shape_node = helper.make_node("Shape", ["Y"], ["shape"])
        reshape_node = helper.make_node("Reshape", ["in", "shape"], ["out"])
        graph = helper.make_graph(
            [batchnorm, shape_node, reshape_node],
            "batchnorm_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(in_shape)),
                helper.make_tensor_value_info("in", TensorProto.FLOAT, list(o_shape)),
                helper.make_tensor_value_info("scale", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("mean", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("var", TensorProto.FLOAT, [in_shape[1]]),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(in_shape))],
        )

        model = helper.make_model(graph, producer_name="batchnorm_test")

        # X, inp, scale, b, mean, var
        inshapes = [in_shape, o_shape, in_shape[1], in_shape[1], in_shape[1], in_shape[1]]
        verify_with_ort(model, inshapes, in_shape, use_vm=True)

    verify_batch_norm_dynamic_subgraph([16, 16, 10, 10], [160, 160])


def verify_conv(
    x_shape,
    w_shape,
    y_shape,
    padding,
    kernel_shape,
    strides,
    dilations,
    auto_pad="NOTSET",
    unset_pad=False,
):
    if unset_pad:
        node = helper.make_node(
            "Conv",
            inputs=["x", "W"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            # Default values for other attributes:
            strides=strides,
            dilations=dilations,
            # groups=1
        )
    elif padding is None:
        node = helper.make_node(
            "Conv",
            inputs=["x", "W"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            # Default values for other attributes:
            strides=strides,
            dilations=dilations,
            # groups=1
            auto_pad=auto_pad,
        )
    else:
        node = helper.make_node(
            "Conv",
            inputs=["x", "W"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            # Default values for other attributes:
            strides=strides,
            dilations=dilations,
            # groups=1
            pads=padding,
        )

    graph = helper.make_graph(
        [node],
        "conv_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_shape)),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))],
    )

    model = helper.make_model(graph, producer_name="conv_test")

    verify_with_ort(model, [x_shape, w_shape], y_shape, use_vm=True, convert_to_static=True)


@tvm.testing.uses_gpu
def test_conv():
    def repeat(N, D):
        return tuple([N for _ in range(D)])

    for D in [1, 2, 3]:
        # Convolution with padding
        verify_conv(
            (1, 1) + repeat(5, D),
            (1, 1) + repeat(3, D),
            (1, 1) + repeat(5, D),
            2 * repeat(1, D),
            repeat(3, D),
            repeat(1, D),
            repeat(1, D),
        )
        # Convolution with assymetric padding
        verify_conv(
            (1, 1) + repeat(5, D),
            (1, 1) + repeat(3, D),
            (1, 1) + repeat(4, D),
            repeat(0, D) + repeat(1, D),
            repeat(3, D),
            repeat(1, D),
            repeat(1, D),
        )
        # Convolution without padding
        verify_conv(
            (1, 1) + repeat(5, D),
            (1, 1) + repeat(3, D),
            (1, 1) + repeat(3, D),
            2 * repeat(0, D),
            repeat(3, D),
            repeat(1, D),
            repeat(1, D),
        )
        # Convolution with autopadding
        verify_conv(
            (1, 1) + repeat(5, D),
            (1, 1) + repeat(3, D),
            (1, 1) + repeat(5, D),
            None,
            repeat(3, D),
            repeat(1, D),
            repeat(1, D),
            auto_pad="SAME_UPPER",
        )
        # Convolution with valid autopadding
        verify_conv(
            (1, 1) + repeat(5, D),
            (1, 1) + repeat(3, D),
            (1, 1) + repeat(3, D),
            None,
            repeat(3, D),
            repeat(1, D),
            repeat(1, D),
            auto_pad="VALID",
        )
        # Convolution with unset padding
        verify_conv(
            (1, 1) + repeat(5, D),
            (1, 1) + repeat(3, D),
            (1, 1) + repeat(3, D),
            2 * repeat(0, D),
            repeat(3, D),
            repeat(1, D),
            repeat(1, D),
            True,
        )
        # Convolution with non uniform stride
        verify_conv(
            (1, 1) + repeat(5, D),
            (1, 1) + repeat(3, D),
            (1, 1) + repeat(3, D),
            None,
            repeat(3, D),
            repeat(2, D),
            repeat(1, D),
            auto_pad="SAME_UPPER",
        )
        # Convolution with dilation
        verify_conv(
            (1, 1) + repeat(5, D),
            (1, 1) + repeat(3, D),
            (1, 1) + repeat(5, D),
            2 * repeat(2, D),
            repeat(3, D),
            repeat(1, D),
            repeat(2, D),
        )


def verify_convtranspose_with_padding(
    x_shape,
    w_shape,
    y_shape,
    padding,
    kernel_shape,
    strides,
    dilations,
    auto_pad="NOTSET",
    unset_pad=False,
):
    if unset_pad:
        node = helper.make_node(
            "ConvTranspose",
            inputs=["x", "W"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            # Default values for other attributes:
            strides=strides,
            dilations=dilations,
            group=1,
        )
    elif padding is None:
        node = helper.make_node(
            "ConvTranspose",
            inputs=["x", "W"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            # Default values for other attributes:
            strides=strides,
            dilations=dilations,
            group=1,
            auto_pad=auto_pad,
        )
    else:
        node = helper.make_node(
            "ConvTranspose",
            inputs=["x", "W"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            # Default values for other attributes:
            strides=strides,
            dilations=dilations,
            group=1,
            pads=padding,
        )

    graph = helper.make_graph(
        [node],
        "convtranspose_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_shape)),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))],
    )

    model = helper.make_model(graph, producer_name="conv_test")

    verify_with_ort(model, [x_shape, w_shape], y_shape, use_vm=True, convert_to_static=True)


def verify_convtranspose(x_shape, w_shape, y_shape, p):
    node = onnx.helper.make_node(
        "ConvTranspose",
        inputs=["x", "W"],
        outputs=["y"],
        strides=[3, 2],
        group=1,
        kernel_shape=[3, 3],
        pads=p,
    )

    graph = helper.make_graph(
        [node],
        "verify_convtranspose_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_shape)),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))],
    )

    model = helper.make_model(graph, producer_name="convtranspose_trest")
    verify_with_ort(model, [x_shape, w_shape], y_shape)


@tvm.testing.uses_gpu
def test_convtranspose():
    # Convolution Transpose with padding
    # (1, 1, 3, 3) input tensor
    # (1, 2, 3, 3) tensor for convolution weights
    # (1, 2, 7, 3) output tensor
    # [1, 2, 1, 2] list for pads
    verify_convtranspose((1, 1, 3, 3), (1, 2, 3, 3), (1, 2, 7, 3), [1, 2, 1, 2])

    def repeat(N, D):
        return tuple([N for _ in range(D)])

    # TODO(mbrookhart): onnxruntime in CI only supports 2D,
    # find something else to test 1D and 3D against
    for D in [2]:
        # Convolution with padding
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, D),
            (1, 1) + repeat(3, D),
            (1, 1) + repeat(5, D),
            2 * repeat(1, D),
            repeat(3, D),
            repeat(1, D),
            repeat(1, D),
        )
        # Convolution without padding
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, D),
            (1, 1) + repeat(3, D),
            (1, 1) + repeat(7, D),
            2 * repeat(0, D),
            repeat(3, D),
            repeat(1, D),
            repeat(1, D),
        )
        # Convolution with autopadding
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, D),
            (1, 1) + repeat(3, D),
            (1, 1) + repeat(5, D),
            None,
            repeat(3, D),
            repeat(1, D),
            repeat(1, D),
            auto_pad="SAME_UPPER",
        )
        # Convolution with valid autopadding
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, D),
            (1, 1) + repeat(3, D),
            (1, 1) + repeat(7, D),
            None,
            repeat(3, D),
            repeat(1, D),
            repeat(1, D),
            auto_pad="VALID",
        )
        # Convolution with unset padding
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, D),
            (1, 1) + repeat(3, D),
            (1, 1) + repeat(7, D),
            2 * repeat(0, D),
            repeat(3, D),
            repeat(1, D),
            repeat(1, D),
            True,
        )
        # Convolution with non uniform stride
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, D),
            (1, 1) + repeat(3, D),
            (1, 1) + repeat(9, D),
            None,
            repeat(3, D),
            repeat(2, D),
            repeat(1, D),
            auto_pad="SAME_UPPER",
        )
        # Convolution with dilation
        # TODO(mbrookhart): Relay doesn't currently support convtranspose with dilation
        # verify_convtranspose_with_padding(
        #     (1, 1) + repeat(5, D),
        #     (1, 1) + repeat(3, D),
        #     (1, 1) + repeat(5, D),
        #     2 * repeat(2, D),
        #     repeat(3, D),
        #     repeat(1, D),
        #     repeat(2, D),
        # )


@tvm.testing.uses_gpu
def test_unsqueeze_constant():
    from torch.nn import Linear, Sequential, Module

    class Flatten(Module):
        def forward(self, input):
            return input.view(input.size(0), -1)

    import tempfile

    with tempfile.NamedTemporaryFile() as fp:
        file_name = fp.name
        input_size = (1, 16, 32, 32)
        dummy_input = torch.randn(*input_size)
        layer = Sequential(Flatten(), Linear(16 * 32 * 32, 64))
        torch.onnx.export(layer, dummy_input, file_name, export_params=True)

        onnx_model = onnx.load(file_name)
        relay.frontend.from_onnx(onnx_model, {"0": input_size})


def verify_pooling(x_shape, kernel_shape, strides, pads, out_shape, mode, auto_pad="NOTSET"):
    print(x_shape, kernel_shape, strides, mode, pads, auto_pad)
    x_np = np.random.uniform(size=x_shape).astype("float32")

    if mode == "max":
        node_type = "MaxPool"
    elif mode == "average":
        node_type = "AveragePool"
    else:
        raise ValueError("Pool method {} is not supported.".format(mode))

    pool_node = helper.make_node(
        node_type, inputs=["x"], outputs=["y"], kernel_shape=kernel_shape, strides=strides
    )

    if pads is None:
        pad_attr = helper.make_attribute("auto_pad", auto_pad)
    else:
        pad_attr = helper.make_attribute("pads", pads)
    pool_node.attribute.append(pad_attr)

    if mode == "max":
        storage_attr = helper.make_attribute("storage_order", 0)
        pool_node.attribute.append(storage_attr)

    graph = helper.make_graph(
        [pool_node],
        "pooling_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="pooling_test")
    verify_with_ort(model, [x_shape], out_shape, use_vm=True, convert_to_static=True)


@tvm.testing.uses_gpu
def test_pooling():
    for mode in ["max", "average"]:
        # Pool1D
        verify_pooling(
            x_shape=[1, 1, 32],
            kernel_shape=[3],
            strides=[1],
            pads=[1, 1],
            out_shape=[1, 1, 32],
            mode=mode,
        )
        # Pool2D
        verify_pooling(
            x_shape=[1, 1, 32, 32],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[1, 1, 1, 1],
            out_shape=[1, 1, 32, 32],
            mode=mode,
        )

        # Pool1D with stride
        verify_pooling(
            x_shape=[1, 1, 32],
            kernel_shape=[3],
            strides=[2],
            pads=[1, 1],
            out_shape=[1, 1, 16],
            mode=mode,
        )
        # Pool2D with stride
        verify_pooling(
            x_shape=[1, 1, 32, 32],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
            out_shape=[1, 1, 16, 16],
            mode=mode,
        )

        # Pool1D with stride and autopadding
        verify_pooling(
            x_shape=[1, 1, 32],
            kernel_shape=[3],
            strides=[2],
            pads=None,
            out_shape=[1, 1, 16],
            mode=mode,
            auto_pad="SAME_UPPER",
        )
        # Pool2D with stride and autopadding
        verify_pooling(
            x_shape=[1, 1, 32, 32],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=None,
            out_shape=[1, 1, 16, 16],
            mode=mode,
            auto_pad="SAME_UPPER",
        )

        # Pool3D with stride
        verify_pooling(
            x_shape=[1, 1, 32, 32, 32],
            kernel_shape=[3, 3, 3],
            strides=[2, 2, 2],
            pads=[1, 1, 1, 1, 1, 1],
            out_shape=[1, 1, 16, 16, 16],
            mode=mode,
        )

        # Pool3D with stride and autopadding
        verify_pooling(
            x_shape=[1, 1, 32, 32, 32],
            kernel_shape=[3, 3, 3],
            strides=[2, 2, 2],
            pads=None,
            out_shape=[1, 1, 16, 16, 16],
            mode=mode,
            auto_pad="SAME_UPPER",
        )


def verify_mod(x_shape, y_shape, fmod, out_shape, dtype="float32"):
    x_np = np.random.uniform(-100.0, 100.0, x_shape).astype(dtype)
    y_np = np.random.uniform(-100.0, 100.0, y_shape).astype(dtype)
    y_np = np.where(y_np == 0, 1, y_np)  # remove 0's to avoid division by zero error

    mod_node = helper.make_node("Mod", inputs=["x", "y"], outputs=["z"], fmod=fmod)

    onnx_dtype = TensorProto.FLOAT if dtype == "float32" else TensorProto.INT32
    graph = helper.make_graph(
        [mod_node],
        "mod_test",
        inputs=[
            helper.make_tensor_value_info("x", onnx_dtype, list(x_shape)),
            helper.make_tensor_value_info("y", onnx_dtype, list(y_shape)),
        ],
        outputs=[helper.make_tensor_value_info("z", onnx_dtype, list(out_shape))],
    )
    model = helper.make_model(graph, producer_name="mod_test")
    verify_with_ort_with_inputs(model, [x_np, y_np], out_shape)


@tvm.testing.uses_gpu
def test_mod():
    # Mod
    verify_mod(
        x_shape=[1, 32, 32], y_shape=[1, 1, 32], fmod=0, out_shape=(1, 32, 32), dtype="int32"
    )
    verify_mod(
        x_shape=[1, 32, 32, 32],
        y_shape=[1, 32, 32, 32],
        fmod=0,
        out_shape=(1, 32, 32, 32),
        dtype="int32",
    )

    # fmod
    verify_mod(
        x_shape=[1, 32, 32], y_shape=[1, 32, 32], fmod=1, out_shape=(1, 32, 32), dtype="int32"
    )
    verify_mod(x_shape=[1, 1, 32, 32], y_shape=[1, 32, 32, 32], fmod=1, out_shape=(1, 32, 32, 32))
    verify_mod(x_shape=[1, 32, 32, 32], y_shape=[1, 1, 32, 32], fmod=1, out_shape=(1, 32, 32, 32))
    verify_mod(
        x_shape=[1, 32, 32, 32],
        y_shape=[1, 32, 32, 32],
        fmod=1,
        out_shape=(1, 32, 32, 32),
        dtype="int32",
    )
    verify_mod(x_shape=[1, 32, 32, 32], y_shape=[1, 32, 32, 32], fmod=1, out_shape=(1, 32, 32, 32))


def verify_xor(x_shape, y_shape):
    x_np = np.random.choice(a=[False, True], size=x_shape).astype("bool")
    y_np = np.random.choice(a=[False, True], size=y_shape).astype("bool")

    np_out = np.logical_xor(x_np, y_np)
    out_shape = np_out.shape

    xor_node = helper.make_node("Xor", inputs=["x", "y"], outputs=["z"])

    onnx_dtype = TensorProto.BOOL
    graph = helper.make_graph(
        [xor_node],
        "xor_test",
        inputs=[
            helper.make_tensor_value_info("x", onnx_dtype, list(x_shape)),
            helper.make_tensor_value_info("y", onnx_dtype, list(y_shape)),
        ],
        outputs=[helper.make_tensor_value_info("z", onnx_dtype, list(out_shape))],
    )
    model = helper.make_model(graph, producer_name="xor_test")

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [x_np, y_np], target, ctx, out_shape)
        tvm.testing.assert_allclose(np_out, tvm_out, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_xor():
    # XOR
    verify_xor(x_shape=[1, 32, 32], y_shape=[1, 32, 32])

    # Xor broadcast
    verify_xor(x_shape=[1, 32, 32], y_shape=[1, 1, 32])


def verify_max_roi_pool(x_shape, rois_shape, pooled_shape, spatial_scale, out_shape):
    if spatial_scale is None:
        pool_node = helper.make_node(
            "MaxRoiPool", inputs=["x", "rois"], outputs=["y"], pooled_shape=pooled_shape
        )
    else:
        pool_node = helper.make_node(
            "MaxRoiPool",
            inputs=["x", "rois"],
            outputs=["y"],
            pooled_shape=pooled_shape,
            spatial_scale=spatial_scale,
        )

    graph = helper.make_graph(
        [pool_node],
        "pool_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
            helper.make_tensor_value_info("rois", TensorProto.FLOAT, list(rois_shape)),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="pool_test")
    verify_with_ort(model, [x_shape, rois_shape], out_shape)


@tvm.testing.uses_gpu
def test_max_roi_pool():
    verify_max_roi_pool(
        x_shape=[1, 3, 6, 6],
        rois_shape=[3, 5],
        pooled_shape=[1, 1],
        spatial_scale=None,
        out_shape=[3, 3, 1, 1],
    )

    verify_max_roi_pool(
        x_shape=[1, 3, 10, 10],
        rois_shape=[4, 5],
        pooled_shape=[2, 2],
        spatial_scale=2.0,
        out_shape=[4, 3, 2, 2],
    )


def verify_lppool(x_shape, kernel_shape, p, strides, pads, out_shape, auto_pad="NOTSET"):
    if pads is None:
        pool_node = helper.make_node(
            "LpPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            p=p,
            auto_pad=auto_pad,
            strides=strides,
        )
    else:
        pool_node = helper.make_node(
            "LpPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            p=p,
            pads=pads,
            strides=strides,
        )

    graph = helper.make_graph(
        [pool_node],
        "lppool_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="lppool_test")
    verify_with_ort(model, [x_shape], out_shape, use_vm=True, convert_to_static=True)


@tvm.testing.uses_gpu
def test_lppool():
    # Pool1D
    verify_lppool(
        x_shape=[1, 1, 32], kernel_shape=[3], p=2, strides=[1], pads=[1, 1], out_shape=[1, 1, 32]
    )

    # Pool2D
    verify_lppool(
        x_shape=[1, 1, 32, 32],
        kernel_shape=[3, 3],
        p=2,
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        out_shape=[1, 1, 32, 32],
    )

    # Pool1D with stride
    verify_lppool(
        x_shape=[1, 1, 32], kernel_shape=[3], p=2, strides=[2], pads=[1, 1], out_shape=[1, 1, 16]
    )

    # Pool2D with stride
    verify_lppool(
        x_shape=[1, 1, 32, 32],
        kernel_shape=[3, 3],
        p=2,
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        out_shape=[1, 1, 16, 16],
    )

    # Pool1D with stride and autopadding
    verify_lppool(
        x_shape=[1, 1, 32],
        kernel_shape=[3],
        p=2,
        strides=[2],
        pads=None,
        out_shape=[1, 1, 16],
        auto_pad="SAME_UPPER",
    )

    # Pool2D with stride and autopadding
    verify_lppool(
        x_shape=[1, 1, 32, 32],
        kernel_shape=[3, 3],
        p=2,
        strides=[2, 2],
        pads=None,
        out_shape=[1, 1, 16, 16],
        auto_pad="SAME_UPPER",
    )

    # Pool3D with stride
    verify_lppool(
        x_shape=[1, 1, 32, 32, 32],
        kernel_shape=[3, 3, 3],
        p=2,
        strides=[2, 2, 2],
        pads=[1, 1, 1, 1, 1, 1],
        out_shape=[1, 1, 16, 16, 16],
    )

    # Pool3D with stride and autopadding
    verify_lppool(
        x_shape=[1, 1, 32, 32, 32],
        kernel_shape=[3, 3, 3],
        p=2,
        strides=[2, 2, 2],
        pads=None,
        out_shape=[1, 1, 16, 16, 16],
        auto_pad="SAME_UPPER",
    )


def verify_rnn(
    seq_length,
    batch_size,
    input_size,
    hidden_size,
    rnn_type="LSTM",
    use_bias=False,
    activations=None,
    alphas=None,
    betas=None,
    use_initial_state=False,
    use_peep=False,
    linear_before_reset=False,
):
    if rnn_type == "LSTM":
        multiplier = 4
    elif rnn_type == "GRU":
        multiplier = 3
    else:
        raise NotImplementedError("%s RNNs not yet supported." % rnn_type)
    x_np = np.random.uniform(size=(seq_length, batch_size, input_size)).astype("float32")
    w_np = np.random.uniform(size=(1, multiplier * hidden_size, input_size)).astype("float32")
    r_np = np.random.uniform(size=(1, multiplier * hidden_size, hidden_size)).astype("float32")
    input_names = ["X", "W", "R"]
    input_tensors = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x_np.shape)),
        helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_np.shape)),
        helper.make_tensor_value_info("R", TensorProto.FLOAT, list(r_np.shape)),
    ]
    input_values = [x_np, w_np, r_np]

    if use_bias:
        b_np = np.random.uniform(size=(1, multiplier * 2 * hidden_size)).astype("float32")
        input_names.append("B")
        input_tensors.append(
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, multiplier * 2 * hidden_size])
        )
        input_values.append(b_np)

    if use_initial_state:
        assert use_bias == True, "Initial states must have bias specified."
        sequence_np = np.repeat(seq_length, batch_size).astype("int32")
        input_names.append("sequence_lens")
        input_tensors.append(
            helper.make_tensor_value_info("sequence_lens", TensorProto.INT32, [batch_size])
        )
        input_values.append(sequence_np)

        initial_h_np = np.random.uniform(size=(1, batch_size, hidden_size)).astype("float32")
        input_names.append("initial_h")
        input_tensors.append(
            helper.make_tensor_value_info(
                "initial_h", TensorProto.FLOAT, [1, batch_size, hidden_size]
            )
        )
        input_values.append(initial_h_np)

        if rnn_type == "LSTM":
            initial_c_np = np.random.uniform(size=(1, batch_size, hidden_size)).astype("float32")
            input_names.append("initial_c")
            input_tensors.append(
                helper.make_tensor_value_info(
                    "initial_c", TensorProto.FLOAT, [1, batch_size, hidden_size]
                )
            )
            input_values.append(initial_c_np)

    if use_peep and rnn_type == "LSTM":
        assert use_initial_state == True, "Peepholes require initial state to be specified."
        p_np = np.random.uniform(size=(1, 3 * hidden_size)).astype("float32")
        input_names.append("P")
        input_tensors.append(
            helper.make_tensor_value_info("P", TensorProto.FLOAT, [1, 3 * hidden_size])
        )
        input_values.append(p_np)

    Y_shape = [seq_length, 1, batch_size, hidden_size]
    Y_h_shape = [1, batch_size, hidden_size]
    outputs = ["Y", "Y_h"]
    graph_outputs = [
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(Y_shape)),
        helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, list(Y_h_shape)),
    ]
    output_shapes = [Y_shape, Y_h_shape]

    if rnn_type == "LSTM":
        Y_c_shape = [1, batch_size, hidden_size]
        outputs.append("Y_c")
        graph_outputs.append(
            helper.make_tensor_value_info("Y_c", TensorProto.FLOAT, list(Y_c_shape))
        )
        output_shapes.append(Y_c_shape)

    rnn_node = helper.make_node(
        rnn_type, inputs=input_names, outputs=outputs, hidden_size=hidden_size
    )
    if activations is not None:
        activations_attr = helper.make_attribute("activations", activations)
        rnn_node.attribute.append(activations_attr)
    if alphas is not None:
        alphas_attr = helper.make_attribute("activation_alpha", alphas)
        rnn_node.attribute.append(alphas_attr)
    if betas is not None:
        betas_attr = helper.make_attribute("activation_beta", betas)
        rnn_node.attribute.append(betas_attr)
    if linear_before_reset and rnn_type == "GRU":
        lbr_attr = helper.make_attribute("linear_before_reset", 1)
        rnn_node.attribute.append(lbr_attr)

    graph = helper.make_graph([rnn_node], "rnn_test", inputs=input_tensors, outputs=graph_outputs)

    model = helper.make_model(graph, producer_name="rnn_test")

    for target, ctx in tvm.testing.enabled_targets():
        onnx_out = get_onnxruntime_output(model, input_values, "float32")
        tvm_out = get_tvm_output(
            model,
            input_values,
            target,
            ctx,
            output_shapes,
            output_dtype=["float32"] * len(output_shapes),
        )
        for o_out, t_out in zip(onnx_out, tvm_out):
            tvm.testing.assert_allclose(o_out, t_out, rtol=5e-3, atol=5e-3)


@tvm.testing.uses_gpu
def test_lstm():
    # No bias.
    verify_rnn(
        seq_length=2, batch_size=1, input_size=16, hidden_size=32, use_bias=False, rnn_type="LSTM"
    )
    # large batch.
    verify_rnn(
        seq_length=4, batch_size=8, input_size=16, hidden_size=32, use_bias=True, rnn_type="LSTM"
    )
    # Non power of two.
    verify_rnn(
        seq_length=3, batch_size=3, input_size=16, hidden_size=40, use_bias=True, rnn_type="LSTM"
    )
    # Long sequence.
    verify_rnn(
        seq_length=8, batch_size=1, input_size=16, hidden_size=32, use_bias=True, rnn_type="LSTM"
    )
    # Large hidden.
    verify_rnn(
        seq_length=2, batch_size=1, input_size=16, hidden_size=128, use_bias=True, rnn_type="LSTM"
    )
    # Large input.
    verify_rnn(
        seq_length=2, batch_size=1, input_size=64, hidden_size=32, use_bias=True, rnn_type="LSTM"
    )

    # Different activation testing.
    # Default value hardsigmoid.
    verify_rnn(
        seq_length=2,
        batch_size=1,
        input_size=16,
        hidden_size=32,
        use_bias=False,
        activations=["HardSigmoid", "Tanh", "Tanh"],
        rnn_type="LSTM",
    )
    # Multiple parameterized activations.
    verify_rnn(
        seq_length=2,
        batch_size=1,
        input_size=16,
        hidden_size=32,
        use_bias=False,
        activations=["HardSigmoid", "LeakyRelu", "Tanh"],
        alphas=[2.0, 0.5],
        betas=[0.3],
        rnn_type="LSTM",
    )
    # All parameterized with new Affine activation.
    verify_rnn(
        seq_length=2,
        batch_size=1,
        input_size=16,
        hidden_size=32,
        use_bias=False,
        activations=["HardSigmoid", "LeakyRelu", "Affine"],
        alphas=[2.0, 0.5, 0.8],
        betas=[0.3, 0.1],
        rnn_type="LSTM",
    )

    # Testing with initial state and peepholes
    verify_rnn(
        seq_length=2,
        batch_size=1,
        input_size=16,
        hidden_size=32,
        use_bias=True,
        use_initial_state=True,
        rnn_type="LSTM",
    )

    verify_rnn(
        seq_length=2,
        batch_size=1,
        input_size=16,
        hidden_size=32,
        use_bias=True,
        use_initial_state=True,
        use_peep=True,
        rnn_type="LSTM",
    )


@tvm.testing.uses_gpu
def test_gru():
    # No bias.
    verify_rnn(
        seq_length=2, batch_size=1, input_size=16, hidden_size=32, use_bias=False, rnn_type="GRU"
    )
    # large batch.
    verify_rnn(
        seq_length=4,
        batch_size=8,
        input_size=16,
        hidden_size=32,
        use_bias=True,
        rnn_type="GRU",
        linear_before_reset=True,
    )
    # Non power of two.
    verify_rnn(
        seq_length=3, batch_size=3, input_size=16, hidden_size=40, use_bias=True, rnn_type="GRU"
    )
    # Long sequence.
    verify_rnn(
        seq_length=8, batch_size=1, input_size=16, hidden_size=32, use_bias=True, rnn_type="GRU"
    )
    # Large hidden.
    verify_rnn(
        seq_length=2, batch_size=1, input_size=16, hidden_size=128, use_bias=True, rnn_type="GRU"
    )
    # Large input.
    verify_rnn(
        seq_length=2, batch_size=1, input_size=64, hidden_size=32, use_bias=True, rnn_type="GRU"
    )

    # Different activation testing.
    # Default value hardsigmoid.
    verify_rnn(
        seq_length=2,
        batch_size=1,
        input_size=16,
        hidden_size=32,
        use_bias=False,
        activations=["HardSigmoid", "Softsign"],
        rnn_type="GRU",
    )
    # Multiple parameterized activations.
    verify_rnn(
        seq_length=2,
        batch_size=1,
        input_size=16,
        hidden_size=32,
        use_bias=False,
        activations=["HardSigmoid", "LeakyRelu"],
        alphas=[2.0, 0.5],
        betas=[0.3],
        rnn_type="GRU",
    )
    # All parameterized with new Affine activation.
    verify_rnn(
        seq_length=2,
        batch_size=1,
        input_size=16,
        hidden_size=32,
        use_bias=False,
        activations=["HardSigmoid", "Affine"],
        alphas=[2.0, 0.8],
        betas=[0.3, 0.1],
        rnn_type="GRU",
    )

    # Testing with initial state
    verify_rnn(
        seq_length=2,
        batch_size=1,
        input_size=16,
        hidden_size=32,
        use_bias=True,
        use_initial_state=True,
        rnn_type="GRU",
    )


# TODO(mbrookhart): enable once VM supports heterogenous execution
# @tvm.testing.uses_gpu
def test_resize():
    def verify(ishape, oshape, scales, mode, coord_trans):
        nodes = [
            make_constant_node("roi", onnx.TensorProto.FLOAT, (0,), []),
            make_constant_node("scales", onnx.TensorProto.FLOAT, (len(scales),), scales),
        ]
        input_names = ["X", "roi", "scales"]
        if oshape != []:
            nodes.append(
                make_constant_node("sizes", onnx.TensorProto.INT64, (len(oshape),), oshape)
            )
            input_names.append("sizes")
        nodes.append(
            helper.make_node(
                "Resize",
                inputs=input_names,
                outputs=["Y"],
                mode=mode,
                coordinate_transformation_mode=coord_trans,
            )
        )

        if oshape == []:
            oshape = [round(dim * scale) for (dim, scale) in zip(ishape, scales)]
        graph = helper.make_graph(
            nodes,
            "resize_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, ishape)],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, oshape)],
        )

        model = helper.make_model(graph, producer_name="resize_test")

        verify_with_ort(model, [ishape], oshape, use_vm=True, opset=11, freeze_params=True)

    # upsampling
    verify([1, 16, 32, 32], [1, 16, 64, 64], [], "nearest", "asymmetric")
    verify([1, 16, 32, 32], [1, 16, 64, 64], [], "linear", "align_corners")
    verify([1, 16, 32, 32], [1, 16, 64, 64], [], "linear", "half_pixel")
    # downsampling
    verify([1, 16, 32, 32], [1, 16, 16, 16], [], "nearest", "asymmetric")
    verify([1, 16, 32, 32], [1, 16, 16, 16], [], "linear", "align_corners")
    verify([1, 16, 32, 32], [1, 16, 16, 16], [], "linear", "half_pixel")
    # scales are specified instead of sizes
    verify([1, 16, 32, 32], [], [1, 1, 2, 2], "nearest", "asymmetric")
    verify([1, 16, 32, 32], [], [1, 1, 0.5, 0.5], "linear", "half_pixel")

    def verify_opset_10(ishape, scales, mode):
        nodes = [
            make_constant_node("scales", onnx.TensorProto.FLOAT, (len(scales),), scales),
        ]
        input_names = ["X", "scales"]
        nodes.append(
            helper.make_node(
                "Resize",
                inputs=input_names,
                outputs=["Y"],
                mode=mode,
            )
        )

        oshape = [round(dim * scale) for (dim, scale) in zip(ishape, scales)]
        graph = helper.make_graph(
            nodes,
            "resize_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, ishape)],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, oshape)],
        )

        model = helper.make_model(graph, producer_name="resize_test")
        model.opset_import[0].version = 10

        verify_with_ort(model, [ishape], oshape, use_vm=True, freeze_params=True)

    verify_opset_10([1, 16, 32, 32], [1, 1, 2, 2], "nearest")
    verify_opset_10([1, 16, 32, 32], [1, 1, 0.5, 0.5], "linear")


@tvm.testing.uses_gpu
def test_nonzero():
    def verify_nonzero(indata, outdata, dtype):
        node = helper.make_node(
            "NonZero",
            inputs=["X"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [node],
            "nonzero_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.INT64, list(indata.shape))],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.INT64, list(outdata.shape))],
        )

        model = helper.make_model(graph, producer_name="nonzero_test")

        verify_with_ort_with_inputs(
            model, [indata], targets=["llvm"], dtype="int64", use_vm=True, opset=9
        )

    input_data = np.array([[1, 0], [1, 1]], dtype=np.int64)
    result = np.array((np.nonzero(input_data)))  # expected output [[0, 1, 1], [0, 0, 1]]
    verify_nonzero(input_data, result, dtype=np.int64)

    input_data = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]], dtype=np.int64)
    result = np.array((np.nonzero(input_data)))  # expected output [[0, 1, 2, 2], [0, 1, 0, 1]]
    verify_nonzero(input_data, result, dtype=np.int64)


@tvm.testing.uses_gpu
def test_topk():
    def verify_topk(input_dims, K, axis=-1):
        output_dims = list(input_dims)
        output_dims[axis] = K

        node = helper.make_node(
            "TopK", inputs=["X", "K"], outputs=["Values", "Indicies"], axis=axis
        )

        graph = helper.make_graph(
            [node],
            "topk_test",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, list(input_dims)),
                helper.make_tensor_value_info(
                    "K",
                    TensorProto.INT64,
                    [
                        1,
                    ],
                ),
            ],
            outputs=[
                helper.make_tensor_value_info("Values", TensorProto.FLOAT, output_dims),
                helper.make_tensor_value_info("Indicies", TensorProto.INT64, output_dims),
            ],
        )

        model = helper.make_model(graph, producer_name="topk_test")

        indata = np.random.uniform(-10, 10, input_dims).astype(np.float32)
        onnx_out = get_onnxruntime_output(model, [indata, np.array([K])])

        for target, ctx in [("llvm", tvm.cpu())]:
            tvm_out = get_tvm_output_with_vm(model, [indata, np.array(K)], target, ctx)
            tvm.testing.assert_allclose(onnx_out, tvm_out, rtol=1e-05, atol=1e-05)

    for n in [12, 32]:
        for shape in [[n], [n, n], [n, n, n]]:
            for k in [1, 5, 10]:
                verify_topk(shape, k)

        verify_topk([n, n, n], 5, 0)
        verify_topk([n, n, n], 5, 1)
        verify_topk([n, n, n], 5, 2)


@tvm.testing.uses_gpu
def test_roi_align():
    def verify_roi_align(
        input_dims, num_roi, output_height, output_width, sampling_ratio=0, spatial_scale=1.0
    ):
        output_dims = [num_roi, input_dims[1], output_height, output_width]

        node = helper.make_node(
            "RoiAlign",
            inputs=["X", "rois", "batch_indicies"],
            outputs=["Y"],
            mode="avg",
            output_height=output_height,
            output_width=output_width,
            sampling_ratio=sampling_ratio,
            spatial_scale=spatial_scale,
        )

        graph = helper.make_graph(
            [node],
            "roialign_test",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, list(input_dims)),
                helper.make_tensor_value_info("rois", TensorProto.FLOAT, [num_roi, 4]),
                helper.make_tensor_value_info(
                    "batch_indicies",
                    TensorProto.INT64,
                    [
                        num_roi,
                    ],
                ),
            ],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_dims)],
        )

        model = helper.make_model(graph, producer_name="roialign_test")

        np_data = np.random.uniform(size=input_dims).astype("float32")
        np_rois = np.random.uniform(size=[num_roi, 4]).astype("float32") * input_dims[2]
        np_batch_indicies = np.random.randint(low=0, high=input_dims[0], size=num_roi)

        verify_with_ort_with_inputs(model, [np_data, np_rois, np_batch_indicies], output_dims)

    verify_roi_align((1, 4, 16, 16), 32, 7, 7, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((4, 4, 16, 32), 32, 7, 7, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((1, 8, 16, 16), 32, 7, 7, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((1, 4, 8, 8), 32, 7, 7, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((1, 4, 16, 16), 16, 5, 7, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((1, 4, 16, 12), 8, 7, 3, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((1, 4, 16, 16), 32, 7, 7, sampling_ratio=0, spatial_scale=0.5)
    verify_roi_align((3, 4, 12, 16), 32, 7, 7, sampling_ratio=0, spatial_scale=1.5)
    verify_roi_align((5, 4, 16, 14), 32, 7, 7, sampling_ratio=1, spatial_scale=1.0)
    verify_roi_align((1, 4, 16, 16), 32, 7, 7, sampling_ratio=2, spatial_scale=1.0)


def verify_cond_loop():
    y_in = helper.make_tensor_value_info("y_in", TensorProto.FLOAT, [1])
    y_out = helper.make_tensor_value_info("y_out", TensorProto.FLOAT, [1])
    scan_out = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [1])
    cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
    cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
    iter_count = helper.make_tensor_value_info("iter_count", TensorProto.INT64, [])

    y = np.array([-2]).astype(np.float32)

    five_const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["five"],
        value=helper.make_tensor(
            name="const_tensor_five", data_type=TensorProto.FLOAT, dims=(), vals=[5]
        ),
    )

    iter_cast_node = helper.make_node(
        "Cast", inputs=["iter_count"], outputs=["iter_cast"], to=onnx.TensorProto.FLOAT
    )

    y_add_node = helper.make_node("Add", inputs=["y_in", "iter_cast"], outputs=["y_out"])

    less_node = helper.make_node("Less", inputs=["y_out", "five"], outputs=["cond_less"])

    squeeze_node = helper.make_node("Squeeze", inputs=["cond_less"], outputs=["cond_squeeze"])

    cond_cast_node = helper.make_node(
        "Cast", inputs=["cond_squeeze"], outputs=["cond_out"], to=onnx.TensorProto.BOOL
    )

    scan_identity_node = helper.make_node("Identity", inputs=["y_out"], outputs=["scan_out"])

    loop_body = helper.make_graph(
        [
            five_const_node,
            iter_cast_node,
            y_add_node,
            less_node,
            squeeze_node,
            cond_cast_node,
            scan_identity_node,
        ],
        "loop_body",
        [iter_count, cond_in, y_in],
        [cond_out, y_out, scan_out],
    )

    loop_node = helper.make_node(
        "Loop", inputs=["trip_count", "cond", "y"], outputs=["res_y", "res_scan"], body=loop_body
    )

    trip_count = np.array(5).astype(np.int64)
    res_y = np.array([13]).astype(np.float32)
    cond = np.array(1).astype(np.bool)
    loop_graph = onnx.helper.make_graph(
        [loop_node],
        "loop_outer",
        inputs=[
            onnx.helper.make_tensor_value_info("trip_count", onnx.TensorProto.INT64, []),
            onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
            onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1]),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("res_y", onnx.TensorProto.FLOAT, [1]),
            onnx.helper.make_tensor_value_info("res_scan", onnx.TensorProto.FLOAT, [5, 1]),
        ],
    )
    loop_model = onnx.helper.make_model(loop_graph)

    # Set a high trip count so that condition trips first.
    trip_count = np.array(40).astype(np.int64)
    cond = np.array(1).astype(np.bool)
    input_vals = [trip_count, cond, y]
    onnx_out = get_onnxruntime_output(loop_model, input_vals)

    for target, ctx in [("llvm", tvm.cpu())]:
        tvm_out = get_tvm_output_with_vm(loop_model, input_vals, target, ctx, freeze_params=True)
        for i in range(len(tvm_out)):
            tvm.testing.assert_allclose(onnx_out[i], tvm_out[i], rtol=1e-05, atol=1e-05)


def verify_count_loop():
    y_in = helper.make_tensor_value_info("y_in", TensorProto.FLOAT, [1])
    y_out = helper.make_tensor_value_info("y_out", TensorProto.FLOAT, [1])
    scan_out = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [1])
    cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
    cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
    iter_count = helper.make_tensor_value_info("iter_count", TensorProto.INT64, [])

    y = np.array([-2]).astype(np.float32)

    iter_cast_node = helper.make_node(
        "Cast", inputs=["iter_count"], outputs=["iter_cast"], to=onnx.TensorProto.FLOAT
    )

    y_add_node = helper.make_node("Add", inputs=["y_in", "iter_cast"], outputs=["y_out"])

    identity_node = helper.make_node("Identity", inputs=["cond_in"], outputs=["cond_out"])

    scan_identity_node = helper.make_node("Identity", inputs=["y_out"], outputs=["scan_out"])

    loop_body = helper.make_graph(
        [identity_node, iter_cast_node, y_add_node, scan_identity_node],
        "loop_body",
        [iter_count, cond_in, y_in],
        [cond_out, y_out, scan_out],
    )

    loop_node = helper.make_node(
        "Loop", inputs=["trip_count", "cond", "y"], outputs=["res_y", "res_scan"], body=loop_body
    )

    trip_count = np.array(5).astype(np.int64)
    res_y = np.array([13]).astype(np.float32)
    cond = np.array(1).astype(np.bool)
    loop_graph = onnx.helper.make_graph(
        [loop_node],
        "loop_outer",
        inputs=[
            onnx.helper.make_tensor_value_info("trip_count", onnx.TensorProto.INT64, []),
            onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
            onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1]),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("res_y", onnx.TensorProto.FLOAT, [1]),
            onnx.helper.make_tensor_value_info("res_scan", onnx.TensorProto.FLOAT, [5, 1]),
        ],
    )
    loop_model = onnx.helper.make_model(loop_graph)

    trip_count = np.array(5).astype(np.int64)
    cond = np.array(1).astype(np.bool)
    input_vals = [trip_count, cond, y]
    onnx_out = get_onnxruntime_output(loop_model, input_vals)

    for target, ctx in [("llvm", tvm.cpu())]:
        tvm_out = get_tvm_output_with_vm(loop_model, input_vals, target, ctx, freeze_params=True)
        for i in range(len(tvm_out)):
            tvm.testing.assert_allclose(onnx_out[i], tvm_out[i], rtol=1e-05, atol=1e-05)


def test_loop():
    # Test a loop that exits once a condition is met.
    verify_cond_loop()
    # Test a loop that exits after a fixed number of iterations.
    verify_count_loop()


@tvm.testing.uses_gpu
def test_if():
    # Given a bool scalar input cond.
    # return constant tensor x if cond is True, otherwise return constant tensor y.
    then_out = onnx.helper.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, [5])
    else_out = onnx.helper.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, [5])

    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

    then_const_node = onnx.helper.make_node(
        "Constant", inputs=[], outputs=["then_out"], value=onnx.numpy_helper.from_array(x)
    )

    else_const_node = onnx.helper.make_node(
        "Constant", inputs=[], outputs=["else_out"], value=onnx.numpy_helper.from_array(y)
    )

    then_body = onnx.helper.make_graph([then_const_node], "then_body", [], [then_out])

    else_body = onnx.helper.make_graph([else_const_node], "else_body", [], [else_out])

    if_node = onnx.helper.make_node(
        "If", inputs=["cond"], outputs=["res"], then_branch=then_body, else_branch=else_body
    )

    if_graph = onnx.helper.make_graph(
        [if_node],
        "if_outer",
        inputs=[
            onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("res", onnx.TensorProto.FLOAT, [5]),
        ],
    )

    if_model = onnx.helper.make_model(if_graph)
    cond = np.array(1).astype("bool")
    correct_out = x if cond else y

    for target, ctx in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output_with_vm(if_model, [cond], target, ctx, freeze_params=True)
        for i in range(len(tvm_out)):
            tvm.testing.assert_allclose(correct_out[i], tvm_out[i], rtol=1e-05, atol=1e-05)


@tvm.testing.uses_gpu
def test_size():
    def verify_size(indata):
        node = helper.make_node(
            "Size",
            inputs=["X"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [node],
            "size_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.INT64, list(indata.shape))],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.INT64, [])],
        )

        model = helper.make_model(graph, producer_name="size_test")

        verify_with_ort_with_inputs(model, [indata], dtype="int64", use_vm=True, opset=11)

    input_data = np.array([[1, 0], [1, 1]], dtype=np.int64)
    verify_size(input_data)

    input_data = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]], dtype=np.int64)
    verify_size(input_data)


@tvm.testing.uses_gpu
def test_maxunpool():
    def verify_maxunpool(data, indices, kernel_shape, strides, output_shape=None, pads=None):
        input_names = ["xT", "xI"]
        input_info = [
            helper.make_tensor_value_info("xT", TensorProto.FLOAT, list(data.shape)),
            helper.make_tensor_value_info("xI", TensorProto.INT64, list(indices.shape)),
        ]
        input_values = [data, indices]
        if output_shape is not None:
            input_names.append("output_shape")
            input_info.append(
                helper.make_tensor_value_info(
                    "output_shape", TensorProto.INT64, list(output_shape.shape)
                )
            )
            input_values.append(output_shape)
        else:
            # Compute expected output shape
            output_shape = np.asarray(([1, 1] + list(strides))) * np.asarray(list(data.shape))
            output_shape += np.asarray(([0, 0] + list(kernel_shape))) - np.asarray(
                ([0, 0] + list(strides))
            )
            if pads is not None:
                output_shape -= np.asarray(
                    [0, 0] + list(np.sum(np.reshape(list(pads), [-1, 2]), axis=-1))
                )
        output_shape = [int(i) for i in output_shape]

        node = helper.make_node(
            "MaxUnpool", inputs=input_names, outputs=["y"], kernel_shape=kernel_shape
        )

        if pads is not None:
            pad_attr = helper.make_attribute("pads", pads)
            node.attribute.append(pad_attr)

        if strides is not None:
            strides_attr = helper.make_attribute("strides", strides)
            node.attribute.append(strides_attr)

        graph = helper.make_graph(
            [node],
            "maxunpool_test",
            inputs=input_info,
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )

        model = helper.make_model(graph, producer_name="size_test")

        verify_with_ort_with_inputs(model, input_values, use_vm=True, opset=11)

    # Basic test
    xT = np.array([[[[5, 6], [7, 8]]]], dtype=np.float32)
    xI = np.array([[[[0, 7], [13, 15]]]], dtype=np.int64)
    verify_maxunpool(xT, xI, [2, 2], strides=[2, 2])
    # Small stride
    verify_maxunpool(xT, xI, [2, 2], strides=[1, 1])
    # Big kernel
    verify_maxunpool(xT, xI, [3, 3], strides=[2, 2])
    # With output shape
    output_shape = np.array((1, 1, 5, 5), dtype=np.int64)
    verify_maxunpool(xT, xI, [2, 2], strides=[2, 2], output_shape=output_shape)
    # With explicit reverse padding
    pads = np.asarray([1, 1, 1, 1]).astype(np.int64)
    verify_maxunpool(xT, xI, [2, 2], strides=[2, 2], pads=pads)


@tvm.testing.uses_gpu
def test_softplus():
    def verify_softplus(indata):
        node = helper.make_node(
            "Softplus",
            inputs=["X"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [node],
            "softplus_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, list(indata.shape))],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(indata.shape))],
        )

        model = helper.make_model(graph, producer_name="softplus_test")

        verify_with_ort_with_inputs(model, [indata], dtype="float32", use_vm=True, opset=11)

    # Simple case with all signs.
    input_data = np.array([[-1, 0, 1]], dtype=np.float32)
    verify_softplus(input_data)
    # More fancy case.
    input_data = np.random.randn(1, 32, 32, 3).astype("float32")
    verify_softplus(input_data)


if __name__ == "__main__":
    test_flatten()
    test_reshape()
    test_shape()
    test_expand()
    test_power()
    test_squeeze()
    test_unsqueeze()
    test_slice()
    test_floor()
    test_ceil()
    test_round()
    test_isinf()
    test_isnan()
    test_clip()
    test_clip_min_max_as_inputs()
    test_onehot()
    test_matmul()
    test_gather()
    test_gatherelements()
    test_gather_nd()
    test_scatter()
    test_lrn()
    test_instance_norm()
    test_upsample()
    test_forward_min()
    test_forward_max()
    test_forward_mean()
    test_forward_hardsigmoid()
    test_forward_arg_min_max()
    test_softmax()
    test_constantofshape()
    test_all_reduce_funcs()
    test_pad()
    test_split()
    test_binary_ops()
    test_single_ops()
    test_leaky_relu()
    test_elu()
    test_selu()
    test_prelu()
    test_ThresholdedRelu()
    test_ScaledTanh()
    test_ParametricSoftplus()
    test_Scale()
    test_LogSoftmax()
    test_resnet()
    test_inception()
    test_densenet()
    test_sign()
    test_not()
    test_and()
    test_tile()
    test_erf()
    test_where()
    test_or()
    test_depth_to_space()
    test_space_to_depth()
    test_batch_norm()
    test_batch_norm_dynamic_subgraph()
    test_conv()
    test_convtranspose()
    test_unsqueeze_constant()
    test_pooling()
    test_lppool()
    test_lstm()
    test_gru()
    test_resize()
    test_nonzero()
    test_topk()
    test_mod()
    test_xor()
    test_max_roi_pool()
    test_roi_align()
    test_range()
    test_loop()
    test_size()
    test_maxunpool()
    test_softplus()
