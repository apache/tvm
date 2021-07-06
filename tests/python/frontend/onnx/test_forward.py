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
import os
import re

import numpy as np
import pytest
import scipy
import torch
import torchvision
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relay
from tvm.contrib import graph_executor

import onnx
from onnx import TensorProto, helper, mapping, numpy_helper


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
    graph_def, input_data, target, device, opset=None, freeze_params=False, convert_to_static=False
):
    """Generic function to execute and get tvm output with vm executor"""
    if not isinstance(input_data, list):
        input_data = [input_data]
    _, shape_dict = get_input_data_shape_dict(graph_def, input_data)

    mod, params = relay.frontend.from_onnx(
        graph_def, shape_dict, opset=opset, freeze_params=freeze_params
    )

    if convert_to_static:
        mod = relay.transform.DynamicToStatic()(mod)

    ex = relay.create_executor("vm", mod=mod, device=device, target=target)
    result = ex.evaluate()(*input_data, **params)
    if isinstance(result, tvm.runtime.NDArray):
        return result.numpy()
    return [r.numpy() for r in result]


def get_tvm_output(
    graph_def,
    input_data,
    target,
    device,
    output_shape=None,
    output_dtype="float32",
    opset=None,
    opt_level=1,
):
    """Generic function to execute and get tvm output"""
    # TODO: Resolve the issues and remove the following lines
    target = "llvm"
    device = tvm.cpu(0)

    input_names, shape_dict = get_input_data_shape_dict(graph_def, input_data)

    mod, params = relay.frontend.from_onnx(graph_def, shape_dict, opset=opset)

    with tvm.transform.PassContext(opt_level=opt_level):
        graph, lib, params = relay.build(mod, target, params=params)

    m = graph_executor.create(graph, lib, device)
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
    if isinstance(output_shape, list):
        tvm_output_list = []
        for i, _ in enumerate(output_shape):
            tvm_output = m.get_output(i)
            tvm_output_list.append(tvm_output.numpy())
        return tvm_output_list
    else:
        tvm_output = m.get_output(0)
        return tvm_output.numpy()


def get_onnxruntime_output(model, inputs):
    import onnxruntime.backend

    rep = onnxruntime.backend.prepare(model.SerializeToString(), "CPU")
    if isinstance(inputs, list) and len(inputs) == 1:
        inp = inputs[0]
    else:
        inp = inputs
    output = rep.run(inp)
    # Unpack output if there's only a single value.
    if len(output) == 1:
        output = output[0]
    return output


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
    apply_softmax=False,
    opt_level=1,
):
    if opset is not None:
        model.opset_import[0].version = opset

    ort_out = get_onnxruntime_output(model, inputs)

    if targets is None:
        targets = [tgt for (tgt, _) in tvm.testing.enabled_targets()]

    for target in targets:
        dev = tvm.device(target, 0)
        if use_vm:
            tvm_out = get_tvm_output_with_vm(
                model,
                inputs,
                target,
                dev,
                opset=opset,
                freeze_params=freeze_params,
                convert_to_static=convert_to_static,
            )
        else:
            tvm_out = get_tvm_output(
                model, inputs, target, dev, out_shape, dtype, opset=opset, opt_level=opt_level
            )
        if not isinstance(tvm_out, list):
            tvm_out = [tvm_out]
        if not isinstance(ort_out, list):
            ort_out = [ort_out]
        for tvm_val, ort_val in zip(tvm_out, ort_out):
            if apply_softmax:
                ort_val = scipy.special.softmax(ort_val)
                tvm_val = scipy.special.softmax(tvm_val)
            tvm.testing.assert_allclose(ort_val, tvm_val, rtol=rtol, atol=atol)
            assert ort_val.dtype == tvm_val.dtype


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


def is_version_greater_than(ver):
    return "".join(re.findall(r"(\d+\.)(\d+\.)(\d)", onnx.__version__)[0]) > "".join(
        re.findall(r"(\d+\.)(\d+\.)(\d)", ver)[0]
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

    for target, dev in tvm.testing.enabled_targets():
        x = np.random.uniform(size=in_shape).astype("int32")
        tvm_out = get_tvm_output(model, x, target, dev, ref_shape, "float32")
        tvm.testing.assert_allclose(ref_shape, tvm_out.shape)


@tvm.testing.uses_gpu
def test_double_reshape():
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
    reshape_node1 = helper.make_node("Reshape", ["in", "ref_in"], ["out1"])
    reshape_node2 = helper.make_node("Reshape", ["in", "ref_in"], ["out2"])
    add_node = helper.make_node("Add", ["out1", "out2"], ["out"])

    graph = helper.make_graph(
        [ref_node, reshape_node1, reshape_node2, add_node],
        "reshape_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(ref_shape))],
    )

    model = helper.make_model(graph, producer_name="reshape_test")

    for target, dev in tvm.testing.enabled_targets():
        x = np.random.uniform(size=in_shape).astype("int32")
        tvm_out = get_tvm_output(model, x, target, dev, ref_shape, "float32")
        tvm.testing.assert_allclose(ref_shape, tvm_out.shape)


@tvm.testing.uses_gpu
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

        for target, dev in tvm.testing.enabled_targets():
            tvm_out = get_tvm_output_with_vm(model, data, target, dev, freeze_params=True)
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

    verify_with_ort(model, [inshape], [outshape])


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

    verify_with_ort(model, [inshape], [outshape])


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

    for target, dev in tvm.testing.enabled_targets():
        x = np.random.uniform(size=in_shape).astype("int32")
        tvm_out = get_tvm_output(model, x, target, dev, ref_shape, "int32")
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

    for target, dev in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, [x, y], target, dev, np_res.shape)
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
    x = np.random.uniform(size=in_shape).astype("float32")
    verify_with_ort_with_inputs(model, [x], [out_shape], opset=11)


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
    verify_with_ort(model, [in_shape])


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
    verify_with_ort(model, [in_shape], opset=11)


def verify_gather(in_shape, indices, axis, dtype):
    x = np.random.uniform(size=in_shape).astype(dtype)
    indices = np.array(indices, dtype="int64")
    out_np = np.take(x, indices, axis=axis)

    y = helper.make_node("Gather", ["in", "indices"], ["out"], axis=axis)

    graph = helper.make_graph(
        [y],
        "gather_test",
        inputs=[
            helper.make_tensor_value_info(
                "in", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], list(in_shape)
            ),
            helper.make_tensor_value_info("indices", TensorProto.INT64, list(indices.shape)),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "out", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], list(out_np.shape)
            )
        ],
    )
    model = helper.make_model(graph, producer_name="gather_test")
    verify_with_ort_with_inputs(model, [x, indices], dtype=dtype)


@tvm.testing.uses_gpu
def test_gather():
    verify_gather((4,), [1], 0, "int32")
    verify_gather((1, 4), [0], 0, "int32")
    verify_gather((4,), [[[1, 0], [0, 1]]], 0, "float32")
    verify_gather((2, 2), [[[1, 0], [0, 1]]], 1, "int32")
    verify_gather((3, 3, 3), [[[1, 0]]], -1, "int32")
    verify_gather((4, 3, 5, 6), [[2, 1, 0, 0]], 0, "float32")


@tvm.testing.uses_gpu
def test_dynamic_gather():
    dtype = "float32"
    in_shape = [2, 2]
    indices = 1
    axis = 1
    x = np.random.uniform(size=in_shape).astype(dtype)
    indices = np.array(indices, dtype="int64")
    out_np = np.take(x, indices, axis=axis)

    indices = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["indices"],
        value=onnx.helper.make_tensor(
            name="const_indices",
            data_type=onnx.TensorProto.INT64,
            dims=[],
            vals=[1],
        ),
    )
    y = helper.make_node("Gather", ["in", "indices"], ["out"], axis=axis)

    graph = helper.make_graph(
        [indices, y],
        "gather_test",
        inputs=[
            helper.make_tensor_value_info(
                "in", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], ["?", "?"]
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "out", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], ["?"] * len(out_np.shape)
            )
        ],
    )
    model = helper.make_model(graph, producer_name="dynamic_gather_test")

    mod, params = relay.frontend.from_onnx(model)

    for target, device in tvm.testing.enabled_targets():
        ex = relay.create_executor("vm", mod=mod, device=device, target=target)
        result = ex.evaluate()(x, **params)
        tvm.testing.assert_allclose(out_np, result.numpy(), rtol=1e-5, atol=1e-5)


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
    verify_with_ort_with_inputs(model, [indata], [outdata.shape], opset=1)


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
        inputs.append(helper.make_tensor_value_info("axes", TensorProto.INT64, list(axes.shape)))
        initializer.append(helper.make_tensor("axes", TensorProto.INT64, list(axes.shape), axes))

    if steps:
        assert axes is not None and len(axes) == len(steps)
        steps = np.asarray(steps)
        inputs.append(helper.make_tensor_value_info("steps", TensorProto.INT64, list(axes.shape)))
        initializer.append(helper.make_tensor("steps", TensorProto.INT64, list(steps.shape), steps))

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
    verify_with_ort_with_inputs(model, [indata], opset=10, freeze_params=True, use_vm=True)


@tvm.testing.uses_gpu
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


def _test_onnx_op_elementwise(inshape, outfunc, npargs, dtype, opname, kwargs, opset=None):
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
    verify_with_ort_with_inputs(model, [indata], [outdata.shape], opset=opset, dtype=dtype)


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
        opset=6,
    )

    _test_onnx_op_elementwise(
        (2, 4, 5, 6),
        np.clip,
        {"a_min": -np.inf, "a_max": 1.0},
        "float32",
        "Clip",
        {"max": 1.0},
        opset=6,
    )

    _test_onnx_op_elementwise(
        (2, 4, 5, 6),
        np.clip,
        {"a_min": -1.0, "a_max": np.inf},
        "float32",
        "Clip",
        {"min": -1.0},
        opset=6,
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

    verify_with_ort(model, [input_shape], out_shape=[input_shape])


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
    verify_with_ort_with_inputs(model, [indata], [outdata.shape], dtype=dtype)


@tvm.testing.uses_gpu
def test_isinf():
    _test_finite_ops((2, 4, 5, 6), np.isinf, {}, "float32", "IsInf", {})


@tvm.testing.uses_gpu
def test_isnan():
    _test_finite_ops((2, 4, 5, 6), np.isnan, {}, "float32", "IsNaN", {})


def verify_gather_nd(in_shape, indices, out_shape, dtype="float32", batch_dims=0, opset=11):
    x = np.random.uniform(size=in_shape).astype(dtype)
    indices = np.array(indices, dtype="int64")

    y = helper.make_node("GatherND", ["in", "indices"], ["out"])

    if opset >= 12:
        batch_dims_attr = helper.make_attribute("batch_dims", batch_dims)
        y.attribute.append(batch_dims_attr)

    graph = helper.make_graph(
        [y],
        "gather_test",
        inputs=[
            helper.make_tensor_value_info(
                "in", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], list(in_shape)
            ),
            helper.make_tensor_value_info("indices", TensorProto.INT64, list(indices.shape)),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "out", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], list(out_shape)
            )
        ],
    )
    model = helper.make_model(graph, producer_name="gather_test")
    verify_with_ort_with_inputs(model, [x, indices], [out_shape], opset=opset)


@tvm.testing.uses_gpu
def test_gather_nd():
    verify_gather_nd([2, 2], [[0, 0], [1, 1]], [2], "int32")
    verify_gather_nd([2, 2], [[1], [0]], [2, 2])
    verify_gather_nd([2, 2, 2], [[0, 1], [1, 0]], [2, 2])
    verify_gather_nd([2, 2, 2], [[[0, 1]], [[1, 0]]], [2, 1, 2])

    if is_version_greater_than("1.6.0"):
        verify_gather_nd([2, 2, 2], [[1], [0]], [2, 2], batch_dims=1, opset=12)
        verify_gather_nd(
            (3, 2, 2, 3, 4),
            np.random.randint(low=0, high=2, size=(3, 2, 3), dtype="int64"),
            (3, 2),
            batch_dims=2,
            opset=12,
        )


@tvm.testing.uses_gpu
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

    # TODO(jwfromm): Replace test against np with test against onnxrt once we update versions.
    for target, dev in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output_with_vm(
            model, [indices_array, np.array([depth]).astype("int32"), values], target, dev
        )
        tvm.testing.assert_allclose(out_np, tvm_out, rtol=1e-5, atol=1e-5)


def verify_gemm(a_shape, b_shape, c_shape=None, freeze_params=False, dtype="float32"):
    out_shape = [a_shape[0], b_shape[1]]
    a_array = np.random.uniform(size=a_shape).astype(dtype)
    b_array = np.random.uniform(size=b_shape).astype(dtype)
    input_names = ["a", "b"]
    ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
    input_nodes = [
        helper.make_tensor_value_info("a", ONNX_DTYPE, list(a_shape)),
        helper.make_tensor_value_info("b", ONNX_DTYPE, list(b_shape)),
    ]
    input_values = [a_array, b_array]
    if c_shape is not None:
        c_array = np.random.uniform(size=c_shape).astype(dtype)
        input_names.append("c")
        input_nodes.append(helper.make_tensor_value_info("c", ONNX_DTYPE, list(c_shape)))
        input_values.append(c_array)

    gemm_node = helper.make_node("Gemm", input_names, ["out"])

    graph = helper.make_graph(
        [gemm_node],
        "gemm_test",
        inputs=input_nodes,
        outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="gemm_test")
    verify_with_ort_with_inputs(model, input_values, freeze_params=freeze_params, dtype=dtype)


@tvm.testing.uses_gpu
def test_gemm():
    verify_gemm(a_shape=(4, 3), b_shape=(3, 4))
    verify_gemm(a_shape=(4, 3), b_shape=(3, 4), c_shape=(4,))
    verify_gemm(a_shape=(4, 3), b_shape=(3, 4), c_shape=(4,), freeze_params=True)
    verify_gemm(a_shape=(4, 3), b_shape=(3, 4), c_shape=(4,), freeze_params=True, dtype="float16")


@tvm.testing.uses_gpu
def test_matmul():
    a_shape = (4, 3)
    b_shape = (3, 4)
    out_shape = [a_shape[0], b_shape[1]]

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
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="matmul_test")
    verify_with_ort_with_inputs(model, [a_array, b_array])


def verify_batch_matmul(a_shape, b_shape, out_shape, target, dev):
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
    verify_with_ort_with_inputs(model, [a_array, b_array], use_vm=True, targets=[target])


# TODO(mbrookhart, electriclilies): Add CUDA as a target once batch matmul is fixed
@tvm.testing.parametrize_targets("llvm")
def test_batch_matmul(target, dev):
    verify_batch_matmul((2, 3, 4, 3), (2, 3, 3, 4), (2, 3, 4, 4), target, dev)
    verify_batch_matmul((2, 4, 3), (3, 4), (2, 4, 4), target, dev)
    verify_batch_matmul((2, 3, 4, 3), (3, 4), (2, 3, 4, 4), target, dev)
    # Test implicit broadcasting.
    verify_batch_matmul((4, 3), (2, 3, 4), (2, 4, 4), target, dev)
    verify_batch_matmul((2, 4, 3), (1, 3, 4), (2, 4, 4), target, dev)
    verify_batch_matmul((1, 4, 3), (2, 3, 4), (2, 4, 4), target, dev)


def verify_simple_dynamic_model(a_shape, b_shape, target, dev):
    def verify_model(ex, a_shape, b_shape):
        a_array = np.random.uniform(size=a_shape).astype("float32")
        b_array = np.random.uniform(size=b_shape).astype("float32")
        # matmul
        out_np = np.matmul(a_array, b_array)
        # relu
        out_np[out_np < 0] = 0

        tvm_out = ex.evaluate()(a_array, b_array).numpy()
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

    ex = relay.create_executor("vm", mod=mod, device=dev, target=target)
    verify_model(ex, a_shape, b_shape)
    verify_model(ex, [a * 2 for a in a_shape], [b * 2 for b in b_shape])
    verify_model(ex, [a * 3 for a in a_shape], [b * 3 for b in b_shape])


# TODO(mbrookhart, electriclilies): Add CUDA as a target once batch matmul is fixed
@tvm.testing.parametrize_targets("llvm")
def test_batch_matmul_dynamic_model(target, dev):
    verify_simple_dynamic_model((2, 3, 4, 3), (2, 3, 3, 4), target, dev)
    verify_simple_dynamic_model((2, 4, 3), (3, 4), target, dev)
    verify_simple_dynamic_model((2, 3, 4, 3), (3, 4), target, dev)


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
    verify_with_ort_with_inputs(model, [in_array])


@tvm.testing.uses_gpu
def test_lrn():
    verify_lrn((5, 5, 5, 5), 3, "float32")
    verify_lrn((5, 5, 5, 5), 3, "float32", alpha=0.0002, beta=0.5, bias=2.0)


def verify_instance_norm(shape, axis=1):
    x = np.random.randn(*shape).astype(np.float32)
    gamma = np.random.randn(shape[1]).astype(np.float32)
    beta = np.random.randn(shape[1]).astype(np.float32)
    epsilon = 1e-5

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
    verify_with_ort_with_inputs(model, [x, gamma, beta], out_shape=[shape])


@tvm.testing.uses_gpu
def test_instance_norm():
    verify_instance_norm((2, 3, 4, 5))
    verify_instance_norm((32, 64, 80, 64))
    verify_instance_norm((8, 6, 5))
    verify_instance_norm((8, 7, 6, 5, 4))


def verify_upsample_nearest():
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale)
    y = helper.make_node("Upsample", ["in"], ["out"], mode="nearest", scales=[1.0, 1.0, 2.0, 2.0])

    in_array = np.random.uniform(size=in_shape).astype(np.float32)

    graph = helper.make_graph(
        [y],
        "upsample_nearest_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_nearest_test")
    verify_with_ort_with_inputs(model, [in_array], [out_shape], opset=7)


def verify_upsample3d_nearest():
    scale = 2
    in_shape = (1, 1, 3, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale, 3 * scale)
    y = helper.make_node(
        "Upsample", ["in"], ["out"], mode="nearest", scales=[1.0, 1.0, 2.0, 2.0, 2.0]
    )

    in_array = np.random.uniform(size=in_shape).astype(np.float32)

    graph = helper.make_graph(
        [y],
        "upsample_nearest_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_nearest_test")
    # Upsample is deprecated after opset 9
    verify_with_ort_with_inputs(model, [in_array], [out_shape], opset=7)


def verify_upsample_bilinear():
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale)
    y = helper.make_node("Upsample", ["in"], ["out"], mode="linear", scales=[1.0, 1.0, 2.0, 2.0])

    in_array = np.random.uniform(size=in_shape).astype(np.float32)

    graph = helper.make_graph(
        [y],
        "upsample_bilinear_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_bilinear_test")
    verify_with_ort_with_inputs(model, [in_array], [out_shape], opset=7)


def verify_upsample3d_trilinear():
    scale = 2
    in_shape = (1, 1, 3, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale, 3 * scale)
    y = helper.make_node("Upsample", ["in", "scales"], ["out"], mode="linear")
    scales = [1.0, 1.0, 2.0, 2.0, 2.0]
    in_array = np.random.uniform(size=in_shape).astype(np.float32)
    out_array = tvm.topi.testing.resize3d_python(
        in_array,
        (scale, scale, scale),
        "NCDHW",
        "linear",
        coordinate_transformation_mode="asymmetric",
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
    # TODO(jwfromm): Trilinear upsampling not supported in 1.0.0 onnxruntime.
    # Replace topi comparison with verify_with_ort once we update.
    for target, dev in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output(model, in_array, target, dev, out_shape, "float32")
        tvm.testing.assert_allclose(out_array, tvm_out, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_upsample():
    verify_upsample_nearest()
    verify_upsample_bilinear()
    verify_upsample3d_nearest()
    verify_upsample3d_trilinear()


def verify_softmax(inshape, axis):
    opname = "Softmax"
    indata = np.random.uniform(size=inshape).astype(np.float32)
    outshape = inshape
    y = helper.make_node(opname, ["in"], ["out"])
    if axis is not None:
        axis_attr = helper.make_attribute("axis", axis)
        y.attribute.append(axis_attr)

    graph = helper.make_graph(
        [y],
        opname + "_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outshape))],
    )

    model = helper.make_model(graph, producer_name=opname + "_test")
    verify_with_ort_with_inputs(model, [indata])


@tvm.testing.uses_gpu
def test_softmax():
    verify_softmax((1, 10), None)
    verify_softmax((1, 10), 1)


def verify_min(input_dim):
    dtype = "float32"

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    min_node = helper.make_node("Min", ["a_np1", "a_np2", "a_np3"], ["out"])

    graph = helper.make_graph(
        [min_node],
        "Min_test",
        inputs=[
            helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim)),
            helper.make_tensor_value_info("a_np2", TensorProto.FLOAT, list(input_dim)),
            helper.make_tensor_value_info("a_np3", TensorProto.FLOAT, list(input_dim)),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_dim))],
    )

    model = helper.make_model(graph, producer_name="Min_test")
    verify_with_ort_with_inputs(model, [a_np1, a_np2, a_np3])


@tvm.testing.uses_gpu
def test_forward_min():
    verify_min((1, 3, 20, 20))
    verify_min((20, 20))


def verify_max(input_dim):
    dtype = "float32"

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    max_node = helper.make_node("Max", ["a_np1", "a_np2", "a_np3"], ["out"])

    graph = helper.make_graph(
        [max_node],
        "Max_test",
        inputs=[
            helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim)),
            helper.make_tensor_value_info("a_np2", TensorProto.FLOAT, list(input_dim)),
            helper.make_tensor_value_info("a_np3", TensorProto.FLOAT, list(input_dim)),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_dim))],
    )

    model = helper.make_model(graph, producer_name="Max_test")
    verify_with_ort_with_inputs(model, [a_np1, a_np2, a_np3])


@tvm.testing.uses_gpu
def test_forward_max():
    verify_max((1, 3, 20, 20))
    verify_max((20, 20))


def verify_mean(input_dim):
    dtype = "float32"

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    mean_node = helper.make_node("Mean", ["a_np1", "a_np2", "a_np3"], ["out"])

    graph = helper.make_graph(
        [mean_node],
        "Mean_test",
        inputs=[
            helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim)),
            helper.make_tensor_value_info("a_np2", TensorProto.FLOAT, list(input_dim)),
            helper.make_tensor_value_info("a_np3", TensorProto.FLOAT, list(input_dim)),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_dim))],
    )

    model = helper.make_model(graph, producer_name="Mean_test")
    verify_with_ort_with_inputs(model, [a_np1, a_np2, a_np3])


@tvm.testing.uses_gpu
def test_forward_mean():
    verify_mean((1, 3, 20, 20))
    verify_mean((20, 20))


def verify_hardsigmoid(input_dim, alpha, beta):
    dtype = "float32"

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)

    hardsigmoid_node = helper.make_node("HardSigmoid", ["a_np1"], ["out"], alpha=alpha, beta=beta)

    graph = helper.make_graph(
        [hardsigmoid_node],
        "HardSigmoid_test",
        inputs=[helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_dim))],
    )

    model = helper.make_model(graph, producer_name="HardSigmoid_test")
    verify_with_ort_with_inputs(model, [a_np1])


@tvm.testing.uses_gpu
def test_forward_hardsigmoid():
    verify_hardsigmoid((1, 3, 20, 20), 0.5, 0.6)
    verify_hardsigmoid((20, 20), 0.3, 0.4)


def verify_argreduce(input_dim, op_name, axis=None, keepdims=None):
    a_np1 = np.random.uniform(-10, 10, input_dim).astype(np.int32)
    out_shape = list(a_np1.shape)
    def_axis = axis if axis is not None else 0
    if keepdims == 1 or keepdims == None:
        out_shape[def_axis] = 1
    else:
        out_shape.pop(def_axis)

    node = onnx.helper.make_node(op_name, inputs=["a_np1"], outputs=["out"])

    if keepdims is not None:
        keepdims_attr = helper.make_attribute("keepdims", keepdims)
        node.attribute.append(keepdims_attr)
    if axis is not None:
        axis_attr = helper.make_attribute("axis", axis)
        node.attribute.append(axis_attr)

    graph = helper.make_graph(
        [node],
        "argreduce_test",
        inputs=[helper.make_tensor_value_info("a_np1", TensorProto.INT32, list(a_np1.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.INT64, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="argreduce_test")
    verify_with_ort_with_inputs(model, [a_np1])


# TODO (mbrookhart, electriclilies) Fix argmin on GPU and enable this test
# @tvm.testing.uses_gpu
def test_forward_arg_min_max():
    """Verify argmin and argmax"""
    verify_argreduce([3, 4, 4], "ArgMin")
    verify_argreduce([3, 4, 4], "ArgMax")
    verify_argreduce([3, 4, 4], "ArgMin", axis=1)
    verify_argreduce([3, 4, 4], "ArgMax", axis=0)
    verify_argreduce([3, 4, 4], "ArgMin", keepdims=0)
    verify_argreduce([3, 4, 4], "ArgMax", keepdims=1)
    for axis in [None, 0, 1, 2]:
        for keepdims in [None, True, False]:
            verify_argreduce([3, 4, 4], "ArgMin", axis, keepdims)
            verify_argreduce([3, 4, 4], "ArgMax", axis, keepdims)


def verify_constantofshape(input_dim, value, dtype):
    fill_node = helper.make_node(
        "ConstantOfShape",
        ["input"],
        ["output"],
        value=helper.make_tensor(
            "value", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], (1,), (value,)
        ),
    )

    inputs = [helper.make_tensor_value_info("input", TensorProto.INT64, [len(input_dim)])]

    graph = helper.make_graph(
        [fill_node],
        "fill_test",
        inputs,
        outputs=[
            helper.make_tensor_value_info(
                "output", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], input_dim
            )
        ],
    )

    model = helper.make_model(graph, producer_name="fill_test")
    input_np = np.array(input_dim).astype("int64")
    verify_with_ort_with_inputs(model, [input_np], use_vm=True)


@tvm.testing.uses_gpu
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
    verify_with_ort_with_inputs(model, [indata], [outdata.shape], dtype="float32", opset=2)


def verify_pad_v11(indata, pads, mode="constant", value=0.0):
    indata = np.array(indata).astype(np.float32)
    #  numpy expect result
    len_dim = len(pads) // 2
    np_pads = [(pads[i], pads[i + len_dim]) for i in range(len_dim)]
    pads = np.array(pads)
    #  onnx graph
    if mode in ["edge", "reflect"]:
        inputs = [indata]
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
        inputs = [indata]
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
    verify_with_ort_with_inputs(model, inputs, opset=11, use_vm=True)


@tvm.testing.uses_gpu
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

    verify_with_ort_with_inputs(model, [data], [outshape], opset=11)


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


def verify_split(indata, outdatas, split, axis=0, pass_split=True, opset=11):
    indata = np.array(indata).astype(np.float32)
    outdatas = [np.array(o).astype(np.float32) for o in outdatas]
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape))]
    input_names = ["input"]
    initializer = []

    if split:
        split_index = range(len(split))
    else:
        split_index = range(len(outdatas))

    if pass_split:
        if opset >= 13:
            input_names.append("split")
            np_split = np.array(split).astype(np.int64)
            inputs.append(
                helper.make_tensor_value_info("split", TensorProto.INT64, list(np_split.shape))
            )
            indata = [indata, np_split]
            initializer.append(
                helper.make_tensor("split", TensorProto.INT64, list(np_split.shape), np_split)
            )
    node = helper.make_node(
        "Split",
        inputs=input_names,
        outputs=["output_{}".format(i) for i in range(len(split_index))],
        axis=axis,
    )

    if pass_split and opset < 13:
        split_attr = helper.make_attribute("split", split)
        node.attribute.append(split_attr)

    graph = helper.make_graph(
        [node],
        "split_test",
        inputs=inputs,
        initializer=initializer,
        outputs=[
            helper.make_tensor_value_info(
                "output_{}".format(i), TensorProto.FLOAT, list(outdatas[i].shape)
            )
            for i in range(len(split_index))
        ],
    )
    model = helper.make_model(graph, producer_name="split_test")
    verify_with_ort_with_inputs(model, indata, out_shape=list(range(len(split_index))), opset=opset)


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
    # Split a single value to a single value
    verify_split([1], [[1]], [1], pass_split=True)


@tvm.testing.uses_gpu
def test_binary_ops():
    in_shape = (1, 2, 3, 3)
    dtype = "float32"
    out_shape = in_shape

    def verify_binary_ops(op, x, y, out_type="float32"):
        z = helper.make_node(op, ["in1", "in2"], ["out"])
        graph = helper.make_graph(
            [z],
            "_test",
            inputs=[
                helper.make_tensor_value_info("in1", TensorProto.FLOAT, x.shape),
                helper.make_tensor_value_info("in2", TensorProto.FLOAT, y.shape),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "out", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(out_type)], list(out_shape)
                )
            ],
        )
        model = helper.make_model(graph, producer_name="_test")
        verify_with_ort_with_inputs(model, [x, y])

    x = np.random.uniform(size=in_shape).astype(dtype)
    y = np.random.uniform(size=in_shape).astype(dtype)
    z = np.random.uniform(size=(3,)).astype(dtype)
    verify_binary_ops("Add", x, y)
    verify_binary_ops("Add", x, z)
    verify_binary_ops("Sub", x, y)
    verify_binary_ops("Sub", x, z)
    verify_binary_ops("Mul", x, y)
    verify_binary_ops("Mul", x, z)
    verify_binary_ops("Div", x, y)
    verify_binary_ops("Div", x, z)
    verify_binary_ops("Sum", x, y)
    verify_binary_ops("Sum", x, z)
    verify_binary_ops("Greater", x, y, "bool")
    verify_binary_ops("Greater", x, z, "bool")
    verify_binary_ops("Less", x, y, "bool")
    verify_binary_ops("Less", x, z, "bool")
    verify_binary_ops("Equal", x, y, "bool")
    verify_binary_ops("Equal", x, z, "bool")


@tvm.testing.uses_gpu
def test_unary_ops():
    in_shape = (1, 2, 3, 3)
    dtype = "float32"
    out_shape = in_shape

    def verify_unary_ops(op, x, rtol=1e-5, atol=1e-5, dtype="float32"):
        x = x.astype(dtype)
        ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
        z = helper.make_node(op, ["in1"], ["out"])
        graph = helper.make_graph(
            [z],
            "_test",
            inputs=[
                helper.make_tensor_value_info("in1", ONNX_DTYPE, list(in_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, list(out_shape))],
        )
        model = helper.make_model(graph, producer_name="_test")
        verify_with_ort_with_inputs(model, [x], rtol=rtol, atol=atol)

    x = np.random.uniform(size=in_shape)
    verify_unary_ops("Neg", x)
    verify_unary_ops("Abs", x)
    verify_unary_ops("Reciprocal", x)
    verify_unary_ops("Reciprocal", x, dtype="float16")
    verify_unary_ops("Sqrt", x)
    verify_unary_ops("Relu", x)
    verify_unary_ops("Exp", x)
    verify_unary_ops("Log", x)
    verify_unary_ops("Log", x)
    verify_unary_ops("Acos", x)
    verify_unary_ops("Acosh", x)
    verify_unary_ops("Asin", x)
    verify_unary_ops("Asinh", x)
    verify_unary_ops("Atan", x)
    verify_unary_ops("Atanh", x)
    verify_unary_ops("Cos", x)
    verify_unary_ops("Cosh", x)
    verify_unary_ops("Sin", x)
    verify_unary_ops("Sinh", x)
    verify_unary_ops("Tan", x)
    verify_unary_ops("Tanh", x)
    verify_unary_ops("Sigmoid", x)
    verify_unary_ops("Softsign", x)


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

        verify_with_ort(
            model,
            [x_shape, a_shape],
            out_shape=[list(x_shape)],
            use_vm=True,
            convert_to_static=True,
        )

    verify_prelu([3, 4, 5, 6], [1, 4, 1, 1])
    verify_prelu([1, 8, 5, 6], [1, 8, 1, 1])
    verify_prelu([2, 12, 16, 16], [1, 12, 1, 1])
    verify_prelu([2, 12, 16, 16], [1])  # Test alpha broadcasting.
    verify_prelu([3, 1], [3, 1])  # Test non NCHW workload.


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
    input_data = np.random.uniform(size=input_size).astype("float32")
    verify_with_ort_with_inputs(onnx_model, [input_data], apply_softmax=True)


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

    node = helper.make_node(
        "Not",
        inputs=["in"],
        outputs=["out"],
    )

    graph = helper.make_graph(
        [node],
        "not_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.BOOL, list(x.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.BOOL, list(x.shape))],
    )

    model = helper.make_model(graph, producer_name="not_test")
    verify_with_ort_with_inputs(model, [x])


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
    verify_with_ort_with_inputs(model, [x, y], [outdata.shape])


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
    verify_with_ort_with_inputs(model, [indata, repeats], use_vm=True, opset=6)


@tvm.testing.uses_gpu
def test_tile():
    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    repeats = np.random.randint(low=1, high=10, size=(np.ndim(x),)).astype(np.int64)
    z = np.tile(x, repeats)
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
    verify_with_ort_with_inputs(model, [indata], [outdata.shape])


@tvm.testing.uses_gpu
def test_erf():
    x = np.random.rand(2, 3, 4, 6).astype(np.float32)
    z = scipy.special.erf(x)
    verify_erf(x, z)


def verify_where(condition, x, y, dtype, outdata, dynamic=False):
    node_list = []
    where_inputs = ["condition", "x", "y"]
    if dynamic:
        shape_node = helper.make_node("Shape", ["x"], ["shape"])
        reshape_node = helper.make_node("Reshape", ["x", "shape"], ["X"])
        where_inputs[1] = "X"
        node_list += [shape_node, reshape_node]
    node = helper.make_node("Where", inputs=where_inputs, outputs=["out"])
    node_list.append(node)
    graph = helper.make_graph(
        node_list,
        "where_test",
        inputs=[
            helper.make_tensor_value_info("condition", TensorProto.BOOL, list(condition.shape)),
            helper.make_tensor_value_info("x", dtype, list(x.shape)),
            helper.make_tensor_value_info("y", dtype, list(y.shape)),
        ],
        outputs=[helper.make_tensor_value_info("out", dtype, list(outdata.shape))],
    )
    model = helper.make_model(graph, producer_name="where_test")
    verify_with_ort_with_inputs(model, [condition, x, y], [outdata.shape], use_vm=True)


@tvm.testing.uses_gpu
def test_where():
    condition = np.array([[1, 0], [1, 1]], dtype=bool)
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

    condition = np.array(1, dtype=bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[5, 6], [7, 8]], dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)

    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[1], [7]], dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata, dynamic=True)


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
    verify_with_ort_with_inputs(model, [x, y], [outdata.shape])


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
        verify_with_ort(model, inshapes, out_shape=[in_shape])

    verify_batch_norm([1, 3, 224, 224])
    verify_batch_norm([1, 3, 24, 24])
    verify_batch_norm([16, 3, 24, 24])
    verify_batch_norm([16, 16, 24, 24])
    verify_batch_norm([16, 16, 10, 10])


@tvm.testing.uses_gpu
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
        verify_with_ort(model, inshapes, out_shape=[in_shape], use_vm=True)

    verify_batch_norm_dynamic_subgraph([16, 16, 10, 10], [160, 160])


def verify_conv(
    x_shape,
    w_shape,
    y_shape,
    padding,
    kernel_shape,
    strides,
    dilations,
    group=1,
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
            group=group,
        )
    elif padding is None:
        ## autopadding with unset default attributes
        kwargs = {}
        if not all([s == 1 for s in strides]):
            kwargs["strides"] = strides
        if not all([d == 1 for d in dilations]):
            kwargs["dilations"] = dilations

        node = helper.make_node(
            "Conv",
            inputs=["x", "W"],
            outputs=["y"],
            # Default values for other attributes:
            auto_pad=auto_pad,
            group=group,
            **kwargs,
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
            group=group,
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

    verify_with_ort(model, [x_shape, w_shape], [y_shape], use_vm=True, convert_to_static=True)


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

    # TODO(jwfromm): Merge with other tests once group_conv3d is supported.
    for D in [1, 2]:
        # Group Convolution
        verify_conv(
            (1, 8) + repeat(5, D),
            (8, 1) + repeat(3, D),
            (1, 8) + repeat(5, D),
            2 * repeat(1, D),
            repeat(3, D),
            repeat(1, D),
            repeat(1, D),
            group=8,
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
    group=1,
):
    node = helper.make_node(
        "ConvTranspose",
        inputs=["x", "W"],
        outputs=["y"],
        kernel_shape=kernel_shape,
        # Default values for other attributes:
        strides=strides,
        dilations=dilations,
    )
    if not unset_pad:
        if padding is None:
            pad_attr = helper.make_attribute("auto_pad", auto_pad)
        else:
            pad_attr = helper.make_attribute("pads", padding)
        node.attribute.append(pad_attr)

    if group is not None:
        group_attr = helper.make_attribute("group", group)
        node.attribute.append(group_attr)

    graph = helper.make_graph(
        [node],
        "convtranspose_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_shape)),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))],
    )

    model = helper.make_model(graph, producer_name="convtranspose_pad_test")

    verify_with_ort(model, [x_shape, w_shape], [y_shape], use_vm=True, convert_to_static=True)


def verify_convtranspose(x_shape, w_shape, y_shape, p, group=1):
    node = onnx.helper.make_node(
        "ConvTranspose",
        inputs=["x", "W"],
        outputs=["y"],
        strides=[3, 2],
        kernel_shape=[3, 3],
        pads=p,
    )

    if group is not None:
        group_attr = helper.make_attribute("group", group)
        node.attribute.append(group_attr)

    graph = helper.make_graph(
        [node],
        "verify_convtranspose_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_shape)),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))],
    )

    model = helper.make_model(graph, producer_name="convtranspose_test")
    verify_with_ort(model, [x_shape, w_shape], y_shape)


@tvm.testing.uses_gpu
def test_convtranspose():
    # Convolution Transpose with padding
    # (1, 1, 3, 3) input tensor
    # (1, 2, 3, 3) tensor for convolution weights
    # (1, 2, 7, 3) output tensor
    # [1, 2, 1, 2] list for pads
    verify_convtranspose((1, 1, 3, 3), (1, 2, 3, 3), (1, 2, 7, 3), [1, 2, 1, 2])
    # Test undefined groups.
    verify_convtranspose((1, 1, 3, 3), (1, 2, 3, 3), (1, 2, 7, 3), [1, 2, 1, 2], group=None)

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
    from torch.nn import Linear, Module, Sequential

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
    verify_with_ort(model, [x_shape], [out_shape], use_vm=False, convert_to_static=True)


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


def verify_global_pooling(x_shape, mode):
    out_shape = x_shape[:2] + [1] * (len(x_shape) - 2)

    if mode == "max":
        node_type = "GlobalMaxPool"
    elif mode == "average":
        node_type = "GlobalAveragePool"
    else:
        raise ValueError("Pool method {} is not supported.".format(mode))

    pool_node = helper.make_node(node_type, inputs=["x"], outputs=["y"])

    graph = helper.make_graph(
        [pool_node],
        "global_pooling_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="global_pooling_test")
    verify_with_ort(model, [x_shape], [out_shape], use_vm=False, convert_to_static=True)


@tvm.testing.uses_gpu
def test_global_pooling():
    # Test each pooling mode across all N-D inputs.
    for mode in ["average", "max"]:
        # 1D Pooling (NCW)
        verify_global_pooling([1, 8, 8], mode)
        verify_global_pooling([4, 1, 4], mode)
        # 2D Pooling (NCHW)
        verify_global_pooling([1, 8, 8, 8], mode)
        verify_global_pooling([4, 1, 6, 4], mode)
        # 3D Pooling (NCDHW)
        verify_global_pooling([1, 8, 6, 8, 8], mode)
        verify_global_pooling([4, 1, 2, 6, 4], mode)


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
    verify_with_ort_with_inputs(model, [x_np, y_np], [out_shape])


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
    verify_with_ort_with_inputs(model, [x_np, y_np], [out_shape])


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
    verify_with_ort(model, [x_shape, rois_shape], [out_shape])


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
    verify_with_ort(model, [x_shape], [out_shape], use_vm=True, convert_to_static=True)


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
    directions=1,
):
    if rnn_type == "LSTM":
        multiplier = 4
    elif rnn_type == "GRU":
        multiplier = 3
    else:
        raise NotImplementedError(f"{rnn_type} RNNs not yet supported.")

    if directions not in [1, 2]:
        raise ValueError(f"Direction should be either 1 or 2 (for bidirectional LSTMs)")

    def get_inputs():
        input_names = []
        input_values = []
        input_tensors = []

        def register(np_arr, name, shape=None):
            input_values.append(np_arr)
            input_names.append(name)

            # Map of numpy dtypes to the protobuf equivalent
            dtype_map = {
                "float32": TensorProto.FLOAT,
                "int32": TensorProto.INT32,
                "int8": TensorProto.INT8,
            }

            if np_arr.dtype.name not in dtype_map:
                raise ValueError(f"Unknown dtype we don't know how to handle {np.dtype.name}")
            if shape is None:
                shape = list(np_arr.shape)
            proto_type = dtype_map[np_arr.dtype.name]
            input_tensors.append(helper.make_tensor_value_info(name, proto_type, shape))

        x_np = np.random.uniform(size=(seq_length, batch_size, input_size)).astype("float32")
        w_np = np.random.uniform(size=(directions, multiplier * hidden_size, input_size)).astype(
            "float32"
        )
        r_np = np.random.uniform(size=(directions, multiplier * hidden_size, hidden_size)).astype(
            "float32"
        )
        register(x_np, "X")
        register(w_np, "W")
        register(r_np, "R")

        if use_bias:
            b_np = np.random.uniform(size=(directions, multiplier * 2 * hidden_size)).astype(
                "float32"
            )
            register(b_np, "B")

        if use_initial_state:
            assert use_bias == True, "Initial states must have bias specified."
            sequence_np = np.repeat(seq_length, batch_size).astype("int32")
            register(sequence_np, "sequence_lens")

            initial_h_np = np.random.uniform(size=(directions, batch_size, hidden_size)).astype(
                "float32"
            )
            register(initial_h_np, "initial_h")

            if rnn_type == "LSTM":
                initial_c_np = np.random.uniform(size=(directions, batch_size, hidden_size)).astype(
                    "float32"
                )
                register(initial_c_np, "initial_c")

        if use_peep and rnn_type == "LSTM":
            assert use_initial_state == True, "Peepholes require initial state to be specified."
            p_np = np.random.uniform(size=(directions, 3 * hidden_size)).astype("float32")
            register(p_np, "P")

        return input_names, input_tensors, input_values

    input_names, input_tensors, input_values = get_inputs()

    def get_outputs():
        output_names = []
        graph_outputs = []
        output_shapes = []

        def register(name, shape, proto_type):
            output_names.append(name)
            graph_outputs.append(helper.make_tensor_value_info(name, proto_type, list(shape)))
            output_shapes.append(list(shape))

        register("Y", [seq_length, directions, batch_size, hidden_size], TensorProto.FLOAT)
        register("Y_h", [directions, batch_size, hidden_size], TensorProto.FLOAT)

        if rnn_type == "LSTM":
            register("Y_c", [directions, batch_size, hidden_size], TensorProto.FLOAT)

        return output_names, graph_outputs, output_shapes

    output_names, graph_outputs, output_shapes = get_outputs()

    rnn_node = helper.make_node(
        rnn_type, inputs=input_names, outputs=output_names, hidden_size=hidden_size
    )
    if activations is not None:
        activations_attr = helper.make_attribute("activations", activations)
        rnn_node.attribute.append(activations_attr)
    if directions == 2:
        direction_attr = helper.make_attribute("direction", "bidirectional")
        rnn_node.attribute.append(direction_attr)
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

    verify_with_ort_with_inputs(model, input_values, output_shapes, atol=1e-2, rtol=1e-2)


@tvm.testing.uses_gpu
def test_lstm():
    for directions in [1, 2]:
        # No bias.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=False,
            rnn_type="LSTM",
            directions=directions,
        )
        # large batch.
        verify_rnn(
            seq_length=4,
            batch_size=8,
            input_size=16,
            hidden_size=32,
            use_bias=True,
            rnn_type="LSTM",
            directions=directions,
        )
        # Non power of two.
        verify_rnn(
            seq_length=3,
            batch_size=3,
            input_size=16,
            hidden_size=40,
            use_bias=True,
            rnn_type="LSTM",
            directions=directions,
        )
        # Long sequence.
        verify_rnn(
            seq_length=8,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=True,
            rnn_type="LSTM",
            directions=directions,
        )
        # Large hidden.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=128,
            use_bias=True,
            rnn_type="LSTM",
            directions=directions,
        )
        # Large input.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=64,
            hidden_size=32,
            use_bias=True,
            rnn_type="LSTM",
            directions=directions,
        )

        # Different activation testing.
        # Default value hardsigmoid.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=False,
            activations=["HardSigmoid", "Tanh", "Tanh"] * directions,
            rnn_type="LSTM",
            directions=directions,
        )
        # Multiple parameterized activations.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=False,
            activations=["HardSigmoid", "LeakyRelu", "Tanh"] * directions,
            alphas=[2.0, 0.5, 0.0] * directions,
            betas=[0.3, 0.0, 0.0] * directions,
            rnn_type="LSTM",
            directions=directions,
        )
        # All parameterized with new Affine activation.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=False,
            activations=["HardSigmoid", "LeakyRelu", "Affine"] * directions,
            alphas=[2.0, 0.5, 0.8] * directions,
            betas=[0.3, 0.1, 0.0] * directions,
            rnn_type="LSTM",
            directions=directions,
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
            directions=directions,
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
            directions=directions,
        )


@tvm.testing.uses_gpu
def test_gru():
    for directions in [1, 2]:
        # No bias.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=False,
            rnn_type="GRU",
            directions=directions,
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
            directions=directions,
        )
        # Non power of two.
        verify_rnn(
            seq_length=3,
            batch_size=3,
            input_size=16,
            hidden_size=40,
            use_bias=True,
            rnn_type="GRU",
            directions=directions,
        )
        # Long sequence.
        verify_rnn(
            seq_length=8,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=True,
            rnn_type="GRU",
            directions=directions,
        )
        # Large hidden.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=128,
            use_bias=True,
            rnn_type="GRU",
            directions=directions,
        )
        # Large input.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=64,
            hidden_size=32,
            use_bias=True,
            rnn_type="GRU",
            directions=directions,
        )

        # Different activation testing.
        # Default value hardsigmoid.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=False,
            activations=["HardSigmoid", "Softsign"] * directions,
            rnn_type="GRU",
            directions=directions,
        )
        # Multiple parameterized activations.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=False,
            activations=["HardSigmoid", "LeakyRelu"] * directions,
            alphas=[2.0, 0.5] * directions,
            betas=[0.3, 0.0] * directions,
            rnn_type="GRU",
            directions=directions,
        )
        # All parameterized with new Affine activation.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=False,
            activations=["HardSigmoid", "Affine"] * directions,
            alphas=[2.0, 0.8] * directions,
            betas=[0.3, 0.1] * directions,
            rnn_type="GRU",
            directions=directions,
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
            directions=directions,
        )


@tvm.testing.uses_gpu
def test_resize():
    def verify(ishape, oshape, scales, mode, coord_trans="asymmetric", alpha=0.5, exclude=False):
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
                cubic_coeff_a=alpha,
                exclude_outside=exclude,
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

        verify_with_ort(model, [ishape], [oshape], use_vm=True, opset=11, freeze_params=True)

    for ndim in [1, 2, 3]:
        method = "nearest"
        for coord_trans in ["asymmetric", "align_corners", "half_pixel"]:
            # upsampling
            verify([1, 16] + [32] * ndim, [1, 16] + [64] * ndim, [], method, coord_trans)
            # downsampling
            verify([1, 16] + [32] * ndim, [1, 16] + [16] * ndim, [], method, coord_trans)
            # scales are specified instead of sizes
            verify([1, 16] + [32] * ndim, [], [1, 1] + [0.5] * ndim, method, coord_trans)
            verify([1, 16] + [32] * ndim, [], [1, 1] + [2] * ndim, method, coord_trans)

        if ndim == 2:
            ## TODO(mbrookhart): ONNX Runtime in CI only supports 2D linear resize
            ## Remove this condition when updating CI
            method = "linear"
            # upsampling
            verify([1, 16] + [32] * ndim, [1, 16] + [64] * ndim, [], method)
            # downsampling
            verify([1, 16] + [32] * ndim, [1, 16] + [16] * ndim, [], method)
            # scales are specified instead of sizes
            verify([1, 16] + [32] * ndim, [], [1, 1] + [0.5] * ndim, method)
            verify([1, 16] + [32] * ndim, [], [1, 1] + [2] * ndim, method)

        if ndim == 2:
            # ONNX Runtime only supports cubic interpolation for 2D images
            method = "cubic"
            for alpha in [0.5, 0.75]:
                for exclude in [True, False]:
                    # upsampling
                    verify(
                        [1, 16] + [32] * ndim,
                        [1, 16] + [64] * ndim,
                        [],
                        method,
                        alpha=alpha,
                        exclude=exclude,
                    )
                    # downsampling
                    verify(
                        [1, 16] + [32] * ndim,
                        [1, 16] + [16] * ndim,
                        [],
                        method,
                        alpha=alpha,
                        exclude=exclude,
                    )
                    # scales are specified instead of sizes
                    verify(
                        [1, 16] + [32] * ndim,
                        [],
                        [1, 1] + [0.5] * ndim,
                        method,
                        alpha=alpha,
                        exclude=exclude,
                    )
                    verify(
                        [1, 16] + [32] * ndim,
                        [],
                        [1, 1] + [2] * ndim,
                        method,
                        alpha=alpha,
                        exclude=exclude,
                    )

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
        verify_with_ort(model, [ishape], [oshape], use_vm=True, freeze_params=True, opset=10)

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

        verify_with_ort_with_inputs(model, [indata], dtype="int64", use_vm=True, opset=9)

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
        verify_with_ort_with_inputs(model, [indata, np.array([K])], use_vm=True)

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
        input_dims,
        num_roi,
        output_height,
        output_width,
        sampling_ratio=0,
        spatial_scale=1.0,
        mode="avg",
    ):
        output_dims = [num_roi, input_dims[1], output_height, output_width]

        node = helper.make_node(
            "RoiAlign",
            inputs=["X", "rois", "batch_indicies"],
            outputs=["Y"],
            mode=mode,
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

        verify_with_ort_with_inputs(
            model, [np_data, np_rois, np_batch_indicies], out_shape=[output_dims]
        )

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

    # ONNX implementation of roi_align with max mode is incorrect, so we don't compare outputs here.


@tvm.testing.uses_gpu
def test_non_max_suppression():
    def verify_nms(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, output_dims
    ):
        input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold"]
        input_nodes = [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes.shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores.shape),
            helper.make_tensor_value_info(
                "max_output_boxes_per_class", TensorProto.INT64, max_output_boxes_per_class.shape
            ),
            helper.make_tensor_value_info("iou_threshold", TensorProto.FLOAT, iou_threshold.shape),
        ]
        inputs = [boxes, scores, max_output_boxes_per_class, iou_threshold]
        if score_threshold is not None:
            input_names.append("score_threshold")
            input_nodes.append(
                helper.make_tensor_value_info(
                    "score_threshold", TensorProto.FLOAT, score_threshold.shape
                )
            )
            inputs.append(score_threshold)
        node = helper.make_node(
            "NonMaxSuppression",
            inputs=input_names,
            outputs=["Y"],
            center_point_box=0,
        )

        graph = helper.make_graph(
            [node],
            "nms_test",
            inputs=input_nodes,
            outputs=[helper.make_tensor_value_info("Y", TensorProto.INT64, output_dims)],
        )

        model = helper.make_model(graph, producer_name="nms_test")

        verify_with_ort_with_inputs(model, inputs, use_vm=True)

    boxes = np.array(
        [
            [
                [0.0, 0.0, 0.3, 0.3],
                [0.0, 0.0, 0.4, 0.4],
                [0.0, 0.0, 0.5, 0.5],
                [0.5, 0.5, 0.9, 0.9],
                [0.5, 0.5, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.3, 0.3],
                [0.0, 0.0, 0.4, 0.4],
                [0.5, 0.5, 0.95, 0.95],
                [0.5, 0.5, 0.96, 0.96],
                [0.5, 0.5, 1.0, 1.0],
            ],
        ]
    ).astype("float32")

    scores = np.array(
        [
            [[0.1, 0.2, 0.6, 0.3, 0.9], [0.1, 0.2, 0.6, 0.3, 0.9]],
            [[0.1, 0.2, 0.6, 0.3, 0.9], [0.1, 0.2, 0.6, 0.3, 0.9]],
        ]
    ).astype("float32")
    max_output_boxes_per_class = np.array(2).astype("int64")
    iou_threshold = np.array(0.8).astype("float32")
    output_dims = [8, 3]
    verify_nms(boxes, scores, max_output_boxes_per_class, iou_threshold, None, output_dims)

    boxes = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ]
        ]
    ).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.4]).astype(np.float32)
    output_dims = [2, 3]
    verify_nms(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, output_dims
    )


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
    cond = np.array(1).astype(bool)
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
    cond = np.array(1).astype(bool)
    input_vals = [trip_count, cond, y]
    verify_with_ort_with_inputs(loop_model, input_vals, use_vm=True, freeze_params=True)


def verify_count_loop():
    y_in = helper.make_tensor_value_info("y_in", TensorProto.FLOAT, [])
    y_out = helper.make_tensor_value_info("y_out", TensorProto.FLOAT, [])
    scan_out = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [])
    cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
    cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
    iter_count = helper.make_tensor_value_info("iter_count", TensorProto.INT64, [])

    y = np.array(-2).astype(np.float32)

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
    cond = np.array(1).astype(bool)
    loop_graph = onnx.helper.make_graph(
        [loop_node],
        "loop_outer",
        inputs=[
            onnx.helper.make_tensor_value_info("trip_count", onnx.TensorProto.INT64, []),
            onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
            onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, []),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("res_y", onnx.TensorProto.FLOAT, []),
            onnx.helper.make_tensor_value_info("res_scan", onnx.TensorProto.FLOAT, [5]),
        ],
    )
    loop_model = onnx.helper.make_model(loop_graph)

    trip_count = np.array(5).astype(np.int64)
    cond = np.array(1).astype(bool)
    input_vals = [trip_count, cond, y]
    verify_with_ort_with_inputs(loop_model, input_vals, use_vm=True, freeze_params=True)


def verify_tensor_loop():
    y_in = helper.make_tensor_value_info("y_in", TensorProto.FLOAT, [3, 3, 3, 3])
    y_out = helper.make_tensor_value_info("y_out", TensorProto.FLOAT, [3, 3, 3, 3])
    scan_out = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [3, 3, 3, 3])
    cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
    cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
    iter_count = helper.make_tensor_value_info("iter_count", TensorProto.INT64, [])

    y = np.random.normal(size=[3, 3, 3, 3]).astype(np.float32)

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
    cond = np.array(1).astype(bool)
    loop_graph = onnx.helper.make_graph(
        [loop_node],
        "loop_outer",
        inputs=[
            onnx.helper.make_tensor_value_info("trip_count", onnx.TensorProto.INT64, []),
            onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
            onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [3, 3, 3, 3]),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("res_y", onnx.TensorProto.FLOAT, [3, 3, 3, 3]),
            onnx.helper.make_tensor_value_info("res_scan", onnx.TensorProto.FLOAT, [5, 3, 3, 3, 3]),
        ],
    )
    loop_model = onnx.helper.make_model(loop_graph)

    trip_count = np.array(5).astype(np.int64)
    cond = np.array(1).astype(bool)
    input_vals = [trip_count, cond, y]
    verify_with_ort_with_inputs(
        loop_model, input_vals, use_vm=True, freeze_params=True, convert_to_static=True
    )


def test_loop():
    # Test a loop that exits once a condition is met.
    verify_cond_loop()
    # Test a loop that exits after a fixed number of iterations with scalar outputs.
    verify_count_loop()
    # Test a loop that uses an array output.
    verify_tensor_loop()


def verify_if(cond_array):
    # Given a bool scalar input cond.
    # return constant tensor x if cond is True, otherwise return constant tensor y.
    then_out = onnx.helper.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, [5])
    else_out = onnx.helper.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, [5])

    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

    then_const_node = onnx.helper.make_node(
        "Constant", inputs=[], outputs=["then_out"], value=numpy_helper.from_array(x)
    )

    else_const_node = onnx.helper.make_node(
        "Constant", inputs=[], outputs=["else_out"], value=numpy_helper.from_array(y)
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
    if cond_array:
        cond = np.array([1]).astype("bool")
    else:
        cond = np.array(1).astype("bool")
    correct_out = x if cond else y

    # TODO(jwfromm): Onnxruntime 1.0.0 is buggy with If statements. Replace this with
    # verify_with_ort once we update versions.
    for target, dev in tvm.testing.enabled_targets():
        tvm_out = get_tvm_output_with_vm(if_model, [cond], target, dev, freeze_params=True)
        for i in range(len(tvm_out)):
            tvm.testing.assert_allclose(correct_out[i], tvm_out[i], rtol=1e-05, atol=1e-05)


@tvm.testing.uses_gpu
def test_if():
    # Confirm that if works with cond as an array or scalar.
    verify_if(cond_array=False)
    verify_if(cond_array=True)


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


@tvm.testing.uses_gpu
def test_cumsum():
    def verify_cumsum(indata, axis, exclusive=0, reverse=0, type="float32"):
        cumsum_node = onnx.helper.make_node(
            "CumSum",
            inputs=["X", "axis"],
            outputs=["Y"],
        )
        if exclusive != 0:
            exclusive_attr = helper.make_attribute("exclusive", exclusive)
            cumsum_node.attribute.append(exclusive_attr)
        if reverse != 0:
            reverse_attr = helper.make_attribute("reverse", reverse)
            cumsum_node.attribute.append(reverse_attr)
        nodes = [
            make_constant_node("axis", onnx.TensorProto.INT32, [1], [axis]),
            cumsum_node,
        ]
        if type == "float32":
            tensor_type = TensorProto.FLOAT
        else:
            tensor_type = TensorProto.INT32
            type = "int32"

        graph = helper.make_graph(
            nodes,
            "cumsum_test",
            inputs=[
                helper.make_tensor_value_info("X", tensor_type, list(indata.shape)),
            ],
            outputs=[helper.make_tensor_value_info("Y", tensor_type, list(indata.shape))],
        )

        model = helper.make_model(graph, producer_name="cumsum_test")

        verify_with_ort_with_inputs(model, [indata], dtype=type, use_vm=True, opset=11)

    data = (
        np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
            ]
        )
        .astype(np.float32)
        .reshape((3, 4))
    )

    verify_cumsum(data, 0)
    verify_cumsum(data, 1)
    verify_cumsum(data, 0, 1, 0)
    verify_cumsum(data, 1, 1, 0)
    verify_cumsum(data, 0, 0, 1)
    verify_cumsum(data, 1, 0, 1)
    verify_cumsum(data, 1, 1, 1)
    data = np.random.randn(1, 32, 32, 3).astype("float32")
    verify_cumsum(data, 1)
    data = np.random.randn(1, 32, 32, 3).astype("int32")
    verify_cumsum(data, 0, type="int32")
    verify_cumsum(data, 1, type="int32")
    verify_cumsum(data, 0, 1, 0, type="int32")
    verify_cumsum(data, 1, 1, 0, type="int32")
    verify_cumsum(data, 0, 0, 1, type="int32")
    verify_cumsum(data, 1, 0, 1, type="int32")
    verify_cumsum(data, 1, 1, 1, type="int32")


@tvm.testing.uses_gpu
def test_eyelike():
    def verify_eyelike(indata):
        node = helper.make_node(
            "EyeLike",
            inputs=["X"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [node],
            "eyelike_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, list(indata.shape))],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(indata.shape))],
        )

        model = helper.make_model(graph, producer_name="eyelike_test")

        verify_with_ort_with_inputs(model, [indata], dtype="float32", opset=9)

    input_data = np.zeros((5, 5), dtype=np.float32)
    verify_eyelike(input_data)


"""
  The following parameterized tests loads the tests that ONNX ships as
  serialized ONNX files, inputs, and outputs. The goal of this test
  is to ensure the ONNX importer is in line with the ONNX specification.
  To allow these tests to run in CI before all pass, a number of tests that
  are not yet supported are skipped.
"""

from onnx import numpy_helper

f = onnx.__file__
import glob

onnx_test_folders = sorted(glob.glob("/".join(f.split("/")[0:-1]) + "/backend/test/data/node/*/"))

unsupported_onnx_tests = [
    "test_basic_convinteger/",
    "test_cast_DOUBLE_to_FLOAT16/",
    "test_cast_FLOAT_to_STRING/",
    "test_cast_STRING_to_FLOAT/",
    "test_compress_0/",
    "test_compress_1/",
    "test_compress_default_axis/",
    "test_compress_negative_axis/",
    "test_convinteger_with_padding/",
    "test_convtranspose_dilations/",
    "test_convtranspose_output_shape/",
    "test_cumsum_1d/",
    "test_cumsum_1d_exclusive/",
    "test_cumsum_1d_reverse/",
    "test_cumsum_1d_reverse_exclusive/",
    "test_cumsum_2d_axis_0/",
    "test_cumsum_2d_axis_1/",
    "test_cumsum_2d_negative_axis/",
    "test_det_2d/",
    "test_det_nd/",
    "test_matmulinteger/",
    "test_maxpool_2d_same_lower/",
    "test_maxpool_2d_same_upper/",
    "test_maxpool_with_argmax_2d_precomputed_pads/",
    "test_maxpool_with_argmax_2d_precomputed_strides/",
    "test_maxunpool_export_with_output_shape/",
    "test_mvn/",
    "test_qlinearmatmul_2D/",
    "test_qlinearmatmul_3D/",
    "test_resize_tf_crop_and_resize/",
    ## For these three tests, ONNX 1.6.0 has incorrect graphs, they pass with ONNX 1.7.0
    "test_resize_upsample_sizes_nearest_ceil_half_pixel/",
    "test_resize_upsample_sizes_nearest_floor_align_corners/",
    "test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric/",
    "test_rnn_seq_length/",
    "test_round/",
    "test_scan9_sum/",
    "test_scan_sum/",
    "test_simple_rnn_defaults/",
    "test_simple_rnn_with_initial_bias/",
    "test_strnormalizer_export_monday_casesensintive_lower/",
    "test_strnormalizer_export_monday_casesensintive_nochangecase/",
    "test_strnormalizer_export_monday_casesensintive_upper/",
    "test_strnormalizer_export_monday_empty_output/",
    "test_strnormalizer_export_monday_insensintive_upper_twodim/",
    "test_strnormalizer_nostopwords_nochangecase/",
    "test_tfidfvectorizer_tf_batch_onlybigrams_skip0/",
    "test_tfidfvectorizer_tf_batch_onlybigrams_skip5/",
    "test_tfidfvectorizer_tf_batch_uniandbigrams_skip5/",
    "test_tfidfvectorizer_tf_only_bigrams_skip0/",
    "test_tfidfvectorizer_tf_onlybigrams_levelempty/",
    "test_tfidfvectorizer_tf_onlybigrams_skip5/",
    "test_tfidfvectorizer_tf_uniandbigrams_skip5/",
    "test_unique_sorted_with_axis/",
    "test_unique_sorted_with_axis_3d/",
    "test_unique_sorted_with_negative_axis/",
    "test_upsample_nearest/",
]


targets = [tgt for (tgt, _) in tvm.testing.enabled_targets()]

target_skips = {
    "cuda": [
        "test_mod_mixed_sign_float16/",
        "test_qlinearconv/",
        "test_resize_upsample_sizes_nearest/",
    ]
}


@pytest.mark.parametrize("target", targets)
@pytest.mark.parametrize("test", onnx_test_folders)
def test_onnx_nodes(test, target):
    if target in target_skips:
        for failure in target_skips[target]:
            if failure in test:
                pytest.skip()
                break
    for failure in unsupported_onnx_tests:
        if failure in test:
            pytest.skip()
            break
    atol = 1e-5
    rtol = 1e-5
    if "roialign" in test:
        # for some reason the ONNX test crops the
        # roialign results to 4 decimal places
        atol = 1e-4
    onnx_model = onnx.load(test + "/model.onnx")
    inputs = []
    outputs = []
    for dataset in glob.glob(test + "/*/"):
        tensors = sorted(glob.glob(dataset + "/*.pb"))
        for tensor in tensors:
            new_tensor = onnx.TensorProto()
            with open(tensor, "rb") as f:
                new_tensor.ParseFromString(f.read())
            if "input" in tensor.split("/")[-1]:
                inputs.append(numpy_helper.to_array(new_tensor))
            elif "output" in tensor.split("/")[-1]:
                outputs.append(numpy_helper.to_array(new_tensor))
            else:
                raise ImportError(str(tensor) + " not labeled as an import or an output")

    dev = tvm.device(target, 0)
    tvm_val = get_tvm_output_with_vm(onnx_model, inputs, target, dev)
    if len(outputs) == 1:
        tvm.testing.assert_allclose(outputs[0], tvm_val, rtol=rtol, atol=atol)
    else:
        for output, val in zip(outputs, tvm_val):
            tvm.testing.assert_allclose(output, val, rtol=rtol, atol=atol)


def test_wrong_input():
    node = helper.make_node(
        "Softplus",
        inputs=["X"],
        outputs=["Y"],
    )

    graph = helper.make_graph(
        [node],
        "softplus_test",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, list([5]))],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list([5]))],
    )
    model = helper.make_model(graph, producer_name="softplus_test")

    # Check that the graph can import correctly with proper shape definitions.
    correct_shape_dict = {"X": [5]}
    relay.frontend.from_onnx(model, shape=correct_shape_dict)

    # Check that an assertion is triggered when an input not in the graph is provided.
    wrong_shape_dict = {"Z": [5]}
    with pytest.raises(AssertionError):
        relay.frontend.from_onnx(model, shape=wrong_shape_dict)


def test_aten():
    torch.set_grad_enabled(False)

    def _convert_to_onnx(model, inputs):
        file_name = "{}.onnx".format("aten_model")
        torch.onnx.export(
            model,
            inputs,
            file_name,
            export_params=True,
            verbose=False,
            opset_version=10,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN,
        )
        onnx_model = onnx.load(file_name)
        assert 's: "embedding_bag"' in str(onnx_model)
        return onnx_model

    def verify_embedding_bag(num_embedding, embedding_dim, data_shape, num_bags=None):
        dummy_data = torch.randint(0, num_embedding - 1, data_shape)
        tvm_inputs = [dummy_data.numpy()]
        model = torch.nn.EmbeddingBag(num_embedding, embedding_dim)
        onnx_model = _convert_to_onnx(model, dummy_data)
        torch_out = model(dummy_data)
        for target, ctx in tvm.testing.enabled_targets():
            tvm_out = get_tvm_output_with_vm(
                onnx_model, tvm_inputs, target, ctx, freeze_params=True, convert_to_static=True
            )
            tvm.testing.assert_allclose(torch_out.numpy(), tvm_out)

    verify_embedding_bag(10, 3, [2, 10])
    verify_embedding_bag(32, 2, [3, 3])


def verify_reverse_sequence(x, sequence_lens, batch_axis, time_axis):
    node = onnx.helper.make_node(
        "ReverseSequence",
        inputs=["x", "sequence_lens"],
        outputs=["y"],
        time_axis=time_axis,
        batch_axis=batch_axis,
    )

    graph = helper.make_graph(
        [node],
        "reverse_sequence_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
            helper.make_tensor_value_info(
                "sequence_lens", TensorProto.INT64, list(sequence_lens.shape)
            ),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x.shape))],
    )

    model = helper.make_model(graph, producer_name="reverse_sequence_test")
    verify_with_ort_with_inputs(model, [x, sequence_lens], [x.shape])


@tvm.testing.uses_gpu
def test_reverse_sequence():
    x = np.array(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        dtype=np.float32,
    )
    sequence_lens = np.array([1, 2, 3, 4], dtype=np.int64)
    verify_reverse_sequence(x, sequence_lens, 0, 1)

    sequence_lens = np.array([4, 3, 2, 1], dtype=np.int64)
    verify_reverse_sequence(x, sequence_lens, 1, 0)


def verify_qlinearconv(
    x_shape,
    w_shape,
    y_shape,
    padding,
    kernel_shape,
    strides,
    dilations,
    auto_pad="NOTSET",
    bias=False,
):

    x_array = np.random.randint(low=0, high=255, size=x_shape).astype("uint8")
    w_array = np.random.uniform(low=0, high=255, size=w_shape).astype("uint8")

    initializer = [
        helper.make_tensor("x_scale", TensorProto.FLOAT, (), [np.random.rand()]),
        helper.make_tensor("x_zero_point", TensorProto.UINT8, (), [np.random.randint(0, 255)]),
        helper.make_tensor("w_scale", TensorProto.FLOAT, (), [np.random.rand()]),
        helper.make_tensor("w_zero_point", TensorProto.UINT8, (), [np.random.randint(0, 255)]),
        helper.make_tensor("y_scale", TensorProto.FLOAT, (), [np.random.rand()]),
        helper.make_tensor("y_zero_point", TensorProto.UINT8, (), [np.random.randint(0, 255)]),
    ]

    input_nodes = [
        helper.make_tensor_value_info("x", TensorProto.UINT8, list(x_shape)),
        helper.make_tensor_value_info("w", TensorProto.UINT8, list(w_shape)),
    ]
    input_names = [
        "x",
        "x_scale",
        "x_zero_point",
        "w",
        "w_scale",
        "w_zero_point",
        "y_scale",
        "y_zero_point",
    ]
    input_values = [x_array, w_array]

    if bias is True:
        b_shape = w_shape[0:1]
        b_array = np.random.randint(low=0, high=65536, size=b_shape).astype("int32")
        input_nodes.append(helper.make_tensor_value_info("B", TensorProto.INT32, list(b_shape)))
        input_names.append("B")
        input_values.append(b_array)

    if padding is None:
        ## autopadding with unset default attributes
        kwargs = {}
        if not all([s == 1 for s in strides]):
            kwargs["strides"] = strides
        if not all([d == 1 for d in dilations]):
            kwargs["dilations"] = dilations

        node = helper.make_node(
            "QLinearConv",
            inputs=input_names,
            outputs=["y"],
            # Default values for other attributes:
            auto_pad=auto_pad,
            **kwargs,
        )
    else:
        node = helper.make_node(
            "QLinearConv",
            inputs=input_names,
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
        inputs=input_nodes,
        outputs=[helper.make_tensor_value_info("y", TensorProto.UINT8, list(y_shape))],
        initializer=initializer,
    )
    model = helper.make_model(graph, producer_name="qlinearconv_test")
    # opt_level=1 will cause error
    verify_with_ort_with_inputs(model, input_values, opt_level=2)


def test_qlinearconv():
    def repeat(N, D):
        return tuple([N for _ in range(D)])

    # only support QLinearConv2d because only support qnn.conv2d
    D = 2

    # Convolution with padding
    verify_qlinearconv(
        (1, 1) + repeat(5, D),
        (1, 1) + repeat(3, D),
        (1, 1) + repeat(5, D),
        2 * repeat(1, D),
        repeat(3, D),
        repeat(1, D),
        repeat(1, D),
    )

    # Convolution with bias
    verify_qlinearconv(
        (1, 1) + repeat(5, D),
        (1, 1) + repeat(3, D),
        (1, 1) + repeat(5, D),
        2 * repeat(1, D),
        repeat(3, D),
        repeat(1, D),
        repeat(1, D),
        bias=True,
    )

    # Convolution with assymetric padding
    verify_qlinearconv(
        (1, 1) + repeat(5, D),
        (1, 1) + repeat(3, D),
        (1, 1) + repeat(4, D),
        repeat(0, D) + repeat(1, D),
        repeat(3, D),
        repeat(1, D),
        repeat(1, D),
    )
    # Convolution without padding
    verify_qlinearconv(
        (1, 1) + repeat(5, D),
        (1, 1) + repeat(3, D),
        (1, 1) + repeat(3, D),
        2 * repeat(0, D),
        repeat(3, D),
        repeat(1, D),
        repeat(1, D),
    )
    # Convolution with autopadding
    verify_qlinearconv(
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
    verify_qlinearconv(
        (1, 1) + repeat(5, D),
        (1, 1) + repeat(3, D),
        (1, 1) + repeat(3, D),
        None,
        repeat(3, D),
        repeat(1, D),
        repeat(1, D),
        auto_pad="VALID",
    )
    # Convolution with non uniform stride
    verify_qlinearconv(
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
    verify_qlinearconv(
        (1, 1) + repeat(5, D),
        (1, 1) + repeat(3, D),
        (1, 1) + repeat(5, D),
        2 * repeat(2, D),
        repeat(3, D),
        repeat(1, D),
        repeat(2, D),
    )


def verify_qlinearadd(a_shape, b_shape, c_shape):

    a_array = np.random.random(a_shape).astype("float32")
    b_array = np.random.random(b_shape).astype("float32")

    input_nodes = [
        helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
        helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
    ]
    input_names = [
        "a",
        "b",
    ]
    input_values = [a_array, b_array]

    node = helper.make_node("QLinearAdd", inputs=input_names, outputs=["C"])

    node = helper.make_node("Add", ["a", "b"], ["C"])
    graph = helper.make_graph(
        [node],
        "qlinearadd_test",
        inputs=input_nodes,
        outputs=[helper.make_tensor_value_info("C", TensorProto.FLOAT, list(c_shape))],
    )
    model = helper.make_model(graph, producer_name="qlinearconv_test")
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

    class RandomDataReader(CalibrationDataReader):
        def __init__(self, n=10):
            self.data = iter(
                [
                    {
                        "a": np.random.random(a_shape).astype("float32"),
                        "b": np.random.random(b_shape).astype("float32"),
                    }
                    for _ in range(n)
                ]
            )

        def get_next(self):
            return next(self.data, None)

    d = tvm.contrib.utils.tempdir()
    model_fp32 = os.path.join(d.temp_dir, "model.onnx")
    onnx.save_model(model, model_fp32)
    model_quant = os.path.join(d.temp_dir, "model.quant.onnx")
    quantized_model = quantize_static(model_fp32, model_quant, RandomDataReader())
    # opt_level=1 will cause error with qnn lowering
    model = onnx.load(model_quant)
    verify_with_ort_with_inputs(model, input_values, opt_level=2)


def test_qlinearadd():
    verify_qlinearadd([4, 2], [4, 2], [4, 2])
    verify_qlinearadd([4, 2], [2], [4, 2])
    verify_qlinearadd([5, 1, 7], [2, 7], [5, 2, 7])


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
    test_gemm()
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
    test_unary_ops()
    test_leaky_relu()
    test_elu()
    test_selu()
    test_prelu()
    test_ThresholdedRelu()
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
    test_cumsum()
    test_wrong_input()
    test_aten()
    test_reverse_sequence()
    test_eyelike()
    test_qlinearconv()
