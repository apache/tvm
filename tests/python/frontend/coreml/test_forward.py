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
"""
CoreML testcases
====================
This article is a test script to test CoreML operator with Relay.
"""
from os import path
from enum import Enum
import tempfile
import numpy as np
import model_zoo
import coremltools as cm
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import datatypes
from tensorflow import keras

import tvm
import tvm.topi.testing
import tvm.testing
from tvm.contrib import graph_executor
from tvm.topi.testing import conv2d_nchw_python
from tvm import relay


def get_tvm_output(
    func, x, params, target, device, out_shape=(1, 1000), input_name="image", dtype="float32"
):
    """Generic function to execute and get tvm output"""
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target, params=params)
    m = graph_executor.GraphModule(lib["default"](device))
    # set inputs
    m.set_input(input_name, tvm.nd.array(x.astype(dtype)))
    m.run()
    # get outputs
    out = m.get_output(0, tvm.nd.empty(out_shape, dtype))
    return out.numpy()


def run_model_checkonly(model_file, model_name="", input_name="image"):
    model = cm.models.MLModel(model_file)
    x = model_zoo.get_cat_image()
    shape_dict = {input_name: x.shape}
    # Some Relay passes change operators on the fly. Ensuring that we generate
    # new graph for each target.
    for target, dev in tvm.testing.enabled_targets():
        mod, params = relay.frontend.from_coreml(model, shape_dict)
        tvm_output = get_tvm_output(mod["main"], x, params, target, dev)
        print(target, dev, model_name, "prediction id: ", np.argmax(tvm_output.flat))


@tvm.testing.uses_gpu
def test_mobilenet_checkonly():
    model_file = model_zoo.get_mobilenet()
    run_model_checkonly(model_file, "mobilenet")


@tvm.testing.uses_gpu
def test_resnet50_checkonly():
    model_file = model_zoo.get_resnet50()
    run_model_checkonly(model_file, "resnet50")


def run_tvm_graph(
    coreml_model, target, device, input_data, input_name, output_shape, output_dtype="float32"
):
    """Generic function to compile on relay and execute on tvm"""
    if isinstance(input_data, list):
        shape_dict = {}
        dtype_dict = {}
        for i, inp in enumerate(input_name):
            shape_dict[inp] = input_data[i].shape
            dtype_dict[inp] = input_data[i].dtype
    else:
        shape_dict = {input_name: input_data.shape}
        dtype_dict = {input_name: input_data.dtype}

    mod, params = relay.frontend.from_coreml(coreml_model, shape_dict)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)

    m = graph_executor.GraphModule(lib["default"](device))
    # set inputs
    if isinstance(input_data, list):
        for i, inp in enumerate(input_name):
            m.set_input(inp, tvm.nd.array(input_data[i].astype(input_data[i].dtype)))
    else:
        m.set_input(input_name, tvm.nd.array(input_data.astype(input_data.dtype)))

    # execute
    m.run()
    # get outputs
    if isinstance(output_shape, list) and isinstance(output_dtype, list):
        tvm_output_list = []
        for i, s in enumerate(output_shape):
            tvm_output = m.get_output(i, tvm.nd.empty((s), output_dtype[i]))
            tvm_output_list.append(tvm_output.numpy())
        return tvm_output_list
    else:
        if not output_shape:
            tvm_output = m.get_output(0)
        else:
            tvm_output = m.get_output(0, tvm.nd.empty((output_shape), output_dtype))
        return tvm_output.numpy()


def verify_add_layer_params(input_dim, alpha=2):
    """Verify add layer params"""
    dtype = "float32"

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.add(a_np1, a_np2) + alpha
    inputs = [("input1", datatypes.Array(*input_dim)), ("input2", datatypes.Array(*input_dim))]
    output = [("output", datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(
        name="Add", alpha=alpha, input_names=["input1", "input2"], output_name="output", mode="ADD"
    )
    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(
            model, target, dev, [a_np1, a_np2], ["input1", "input2"], b_np.shape, dtype
        )
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)


@tvm.testing.uses_gpu
def test_forward_add_layer_params():
    verify_add_layer_params((1, 2, 2), 0)
    verify_add_layer_params((1, 2, 2), 1)
    verify_add_layer_params((1, 3, 3), 2)


def verify_multiply_layer_params(input_dim, alpha):
    """Verify multiply layer params"""
    dtype = "float32"

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.multiply(a_np1, a_np2) * alpha
    inputs = [("input1", datatypes.Array(*input_dim)), ("input2", datatypes.Array(*input_dim))]
    output = [("output", datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(
        name="Mul",
        alpha=alpha,
        input_names=["input1", "input2"],
        output_name="output",
        mode="MULTIPLY",
    )
    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(
            model, target, dev, [a_np1, a_np2], ["input1", "input2"], b_np.shape, dtype
        )
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)


@tvm.testing.uses_gpu
def test_forward_multiply_layer_params():
    verify_multiply_layer_params((1, 2, 2), 0)
    verify_multiply_layer_params((1, 2, 2), 1)
    verify_multiply_layer_params((1, 3, 3), 2)


def verify_concat_layer_params(input1_dim, input2_dim):
    """Verify concat layer params"""
    dtype = "float32"

    a_np1 = np.random.uniform(size=input1_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input2_dim).astype(dtype)

    b_np = np.concatenate((a_np1, a_np2), axis=1)
    inputs = [("input1", datatypes.Array(*input1_dim)), ("input2", datatypes.Array(*input2_dim))]
    output = [("output", datatypes.Array(*b_np.shape))]  # pylint:disable=not-an-iterable
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(
        name="Concate", input_names=["input1", "input2"], output_name="output", mode="CONCAT"
    )
    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(
            model, target, dev, [a_np1, a_np2], ["input1", "input2"], b_np.shape, dtype
        )
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)


@tvm.testing.uses_gpu
def test_forward_concat_layer_params():
    verify_concat_layer_params((1, 1, 2, 2), (1, 2, 2, 2))
    verify_concat_layer_params((1, 2, 4, 4), (1, 3, 4, 4))


def _verify_upsample_layer_params(input_dim, scale, mode):
    dtype = "float32"

    a_np = np.full(input_dim, 1, dtype=dtype)

    if mode == "NN":
        method = "nearest_neighbor"
        coord_trans = "asymmetric"
    else:
        method = "linear"
        coord_trans = "align_corners"

    b_np = tvm.topi.testing.resize2d_python(a_np, (scale, scale), "NCHW", method, coord_trans)

    input_data = [("input", datatypes.Array(*input_dim))]
    output = [("output", datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(input_data, output)
    builder.add_upsample(
        name="Upsample",
        scaling_factor_h=scale,
        scaling_factor_w=scale,
        mode=mode,
        input_name="input",
        output_name="output",
    )

    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(model, target, dev, a_np, "input", b_np.shape, dtype)
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)


@tvm.testing.uses_gpu
def test_forward_upsample_layer_params():
    """Upsample Layer Params"""
    _verify_upsample_layer_params((1, 16, 32, 32), 2, "NN")
    _verify_upsample_layer_params((1, 4, 6, 6), 3, "BILINEAR")


def _verify_l2_normalize(input_dim, eps):
    dtype = "float32"

    a_np = np.random.uniform(size=input_dim).astype(dtype)
    b_np = tvm.topi.testing.l2_normalize_python(a_np, eps, 1)

    input_data = [("input", datatypes.Array(*input_dim))]
    output = [("output", datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(input_data, output)
    builder.add_l2_normalize(name="L2", epsilon=eps, input_name="input", output_name="output")

    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(model, target, dev, a_np, "input", b_np.shape, dtype)
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)


@tvm.testing.uses_gpu
def test_forward_l2_normalize():
    _verify_l2_normalize((1, 3, 20, 20), 0.001)


def _verify_lrn(input_dim, size, bias, alpha, beta):
    dtype = "float32"
    axis = 1
    a_np = np.random.uniform(size=input_dim).astype(dtype)
    b_np = tvm.topi.testing.lrn_python(a_np, size, axis, bias, alpha, beta)

    input_data = [("input", datatypes.Array(*input_dim))]
    output = [("output", datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(input_data, output)
    builder.add_lrn(
        name="LRN",
        input_name="input",
        output_name="output",
        alpha=alpha,
        beta=beta,
        k=bias,
        local_size=size,
    )

    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(model, target, dev, a_np, "input", b_np.shape, dtype)
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)


@tvm.testing.uses_gpu
def test_forward_lrn():
    _verify_lrn((1, 3, 10, 20), 3, 1.0, 1.0, 0.5)


def _verify_average(input_dim1, input_dim2, axis=0):
    dtype = "float32"

    a_np1 = np.random.uniform(size=input_dim1).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim2).astype(dtype)

    b_np = np.mean((a_np1, a_np2), axis=axis)

    inputs = [("input1", datatypes.Array(*input_dim1)), ("input2", datatypes.Array(*input_dim2))]
    output = [("output", datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(
        name="MEAN", input_names=["input1", "input2"], output_name="output", mode="AVE"
    )
    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(
            model, target, dev, [a_np1, a_np2], ["input1", "input2"], b_np.shape, dtype
        )
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)


@tvm.testing.uses_gpu
def test_forward_average():
    _verify_average((1, 3, 20, 20), (1, 3, 20, 20))
    _verify_average((3, 20, 20), (1, 3, 20, 20))
    _verify_average((20, 20), (1, 3, 20, 20))


def _verify_max(input_dim):
    dtype = "float32"

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.max((a_np1, a_np2, a_np3), axis=0)

    inputs = [
        ("input1", datatypes.Array(*input_dim)),
        ("input2", datatypes.Array(*input_dim)),
        ("input3", datatypes.Array(*input_dim)),
    ]
    output = [("output", datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(
        name="Max", input_names=["input1", "input2", "input3"], output_name="output", mode="MAX"
    )
    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(
            model,
            target,
            dev,
            [a_np1, a_np2, a_np3],
            ["input1", "input2", "input3"],
            b_np.shape,
            dtype,
        )
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)


@tvm.testing.uses_gpu
def test_forward_max():
    _verify_max((1, 3, 20, 20))
    _verify_max((20, 20))


def _verify_min(input_dim):
    dtype = "float32"

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.min((a_np1, a_np2, a_np3), axis=0)

    inputs = [
        ("input1", datatypes.Array(*input_dim)),
        ("input2", datatypes.Array(*input_dim)),
        ("input3", datatypes.Array(*input_dim)),
    ]
    output = [("output", datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(
        name="Min", input_names=["input1", "input2", "input3"], output_name="output", mode="MIN"
    )
    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(
            model,
            target,
            dev,
            [a_np1, a_np2, a_np3],
            ["input1", "input2", "input3"],
            b_np.shape,
            dtype,
        )
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)


@tvm.testing.uses_gpu
def test_forward_min():
    _verify_min((1, 3, 20, 20))
    _verify_min((20, 20))


def verify_unary_sqrt(input_dim):
    """Verify unary sqrt"""
    dtype = "float32"

    a_np = np.random.uniform(size=input_dim).astype(dtype)
    ref_val = np.sqrt(a_np)

    inputs = [("input", datatypes.Array(*input_dim))]
    output = [("output", datatypes.Array(*ref_val.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_unary(name="sqrt", input_name="input", output_name="output", mode="sqrt")

    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(model, target, dev, [a_np], ["input"], ref_val.shape, dtype)
        tvm.testing.assert_allclose(out, ref_val, rtol=1e-5)


def verify_unary_rsqrt(input_dim, epsilon=0):
    """Verify unary rsqrt"""
    dtype = "float32"

    a_np = np.random.uniform(size=input_dim).astype(dtype)
    ref_val = 1 / np.sqrt(a_np + epsilon)

    inputs = [("input", datatypes.Array(*input_dim))]
    output = [("output", datatypes.Array(*ref_val.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_unary(
        name="rsqrt", input_name="input", output_name="output", mode="rsqrt", epsilon=epsilon
    )

    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(model, target, dev, [a_np], ["input"], ref_val.shape, dtype)
        tvm.testing.assert_allclose(out, ref_val, rtol=1e-5)


def verify_unary_inverse(input_dim, epsilon=0):
    """Verify unary inverse"""
    dtype = "float32"

    a_np = np.random.uniform(size=input_dim).astype(dtype)
    ref_val = 1 / (a_np + epsilon)

    inputs = [("input", datatypes.Array(*input_dim))]
    output = [("output", datatypes.Array(*ref_val.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_unary(
        name="inverse", input_name="input", output_name="output", mode="inverse", epsilon=epsilon
    )

    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(model, target, dev, [a_np], ["input"], ref_val.shape, dtype)
        tvm.testing.assert_allclose(out, ref_val, rtol=1e-5)


def verify_unary_power(input_dim, alpha):
    """Verify unary power"""
    dtype = "float32"

    a_np = np.random.uniform(size=input_dim).astype(dtype)
    ref_val = np.power(a_np, alpha)

    inputs = [("input", datatypes.Array(*input_dim))]
    output = [("output", datatypes.Array(*ref_val.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_unary(
        name="power", input_name="input", output_name="output", mode="power", alpha=alpha
    )

    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(model, target, dev, [a_np], ["input"], ref_val.shape, dtype)
        tvm.testing.assert_allclose(out, ref_val, rtol=1e-5)


def verify_unary_exp(input_dim):
    """Verify unary exp"""
    dtype = "float32"

    a_np = np.random.uniform(size=input_dim).astype(dtype)
    ref_val = np.exp(a_np)

    inputs = [("input", datatypes.Array(*input_dim))]
    output = [("output", datatypes.Array(*ref_val.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_unary(name="exp", input_name="input", output_name="output", mode="exp")

    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(model, target, dev, [a_np], ["input"], ref_val.shape, dtype)
        tvm.testing.assert_allclose(out, ref_val, rtol=1e-5)


def verify_unary_log(input_dim):
    """Verify unary log"""
    dtype = "float32"

    a_np = np.random.uniform(size=input_dim).astype(dtype)
    ref_val = np.log(a_np)

    inputs = [("input", datatypes.Array(*input_dim))]
    output = [("output", datatypes.Array(*ref_val.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_unary(name="log", input_name="input", output_name="output", mode="log")

    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(model, target, dev, [a_np], ["input"], ref_val.shape, dtype)
        tvm.testing.assert_allclose(out, ref_val, rtol=1e-5)


def verify_unary_abs(input_dim):
    """Verify unary abs"""
    dtype = "float32"

    a_np = np.random.uniform(-100.0, 100.0, size=input_dim).astype(dtype)
    ref_val = np.abs(a_np)

    inputs = [("input", datatypes.Array(*input_dim))]
    output = [("output", datatypes.Array(*ref_val.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_unary(name="abs", input_name="input", output_name="output", mode="abs")

    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(model, target, dev, [a_np], ["input"], ref_val.shape, dtype)
        tvm.testing.assert_allclose(out, ref_val, rtol=1e-5)


def verify_unary_threshold(input_dim, alpha):
    """Verify unary threshold"""
    dtype = "float32"

    a_np = np.random.uniform(-100.0, 100.0, size=input_dim).astype(dtype)
    ref_val = np.maximum(a_np, alpha)

    inputs = [("input", datatypes.Array(*input_dim))]
    output = [("output", datatypes.Array(*ref_val.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_unary(
        name="threshold", input_name="input", output_name="output", mode="threshold", alpha=alpha
    )

    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(model, target, dev, [a_np], ["input"], ref_val.shape, dtype)
        tvm.testing.assert_allclose(out, ref_val, rtol=1e-5)


@tvm.testing.uses_gpu
def test_forward_unary():
    """All unary"""
    verify_unary_sqrt((1, 3, 20, 20))
    verify_unary_rsqrt((1, 3, 20, 20))
    verify_unary_rsqrt((1, 3, 20, 20), epsilon=1e-6)
    verify_unary_inverse((1, 3, 20, 20))
    verify_unary_inverse((1, 3, 20, 20), epsilon=1e-6)
    verify_unary_power((1, 3, 20, 20), alpha=0.5)
    verify_unary_power((1, 3, 20, 20), alpha=4)
    verify_unary_exp((1, 3, 20, 20))
    verify_unary_log((1, 3, 20, 20))
    verify_unary_abs((1, 3, 20, 20))
    verify_unary_threshold((1, 3, 20, 20), alpha=-6.0)
    verify_unary_threshold((1, 3, 20, 20), alpha=5.0)


@tvm.testing.uses_gpu
def test_forward_reduce():
    """Reduce"""

    class ReduceAxis(Enum):
        # pylint: disable=invalid-name
        CHW = 0
        HW = 1
        C = 2
        H = 3
        W = 4

    def _verify_reduce(input_dim, mode, axis, ref_func, dtype="float32"):
        print(input_dim, mode, axis)
        a_np = np.random.uniform(size=input_dim).astype(dtype)

        # translate to axis from coreml format
        if axis == ReduceAxis.CHW:
            np_axis = (-3, -2, -1)
        elif axis == ReduceAxis.HW:
            np_axis = (-2, -1)
        elif axis == ReduceAxis.C:
            np_axis = -3
        elif axis == ReduceAxis.H:
            np_axis = -2
        elif axis == ReduceAxis.W:
            np_axis = -1

        if ref_func is np.argmax:
            ref_val = np.expand_dims(ref_func(a_np, np_axis), np_axis).astype(dtype)
        else:
            ref_val = ref_func(a_np, np_axis, keepdims=True)

        inputs = [("input", datatypes.Array(*input_dim))]
        output = [("output", datatypes.Array(*ref_val.shape))]
        builder = NeuralNetworkBuilder(inputs, output)
        builder.add_reduce(
            name=mode, input_name="input", output_name="output", axis=axis.name, mode=mode
        )

        model = cm.models.MLModel(builder.spec)
        for target, dev in tvm.testing.enabled_targets():
            out = run_tvm_graph(model, target, dev, [a_np], ["input"], ref_val.shape, dtype)
            tvm.testing.assert_allclose(out, ref_val, rtol=1e-5, atol=1e-5)

    dshapes = [[10, 10], [1, 10, 10], [1, 3, 10, 10]]
    for dshape in dshapes:
        for axis in ReduceAxis:
            if len(dshape) < 3 and axis in [ReduceAxis.CHW, ReduceAxis.C]:
                # input must have rank at least 3
                continue
            _verify_reduce(dshape, "sum", axis, np.sum)
            _verify_reduce(dshape, "avg", axis, np.mean)
            _verify_reduce(dshape, "prod", axis, np.prod)
            _verify_reduce(dshape, "min", axis, np.min)
            _verify_reduce(dshape, "max", axis, np.max)
            if axis in [ReduceAxis.C, ReduceAxis.H, ReduceAxis.W]:
                # For mode ArgMax, axis must be [-1] or [-2] or [-3]
                _verify_reduce(dshape, "argmax", axis, np.argmax, dtype="int32")


def verify_reshape(input_dim, target_shape, mode):
    """Reshape"""
    dtype = "float32"

    a_np = np.random.uniform(-100.0, 100.0, size=input_dim).astype(dtype)
    ref_val = np.reshape(a_np, target_shape)

    inputs = [("input", datatypes.Array(*input_dim))]
    output = [("output", datatypes.Array(*ref_val.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_reshape(
        name="reshape",
        input_name="input",
        output_name="output",
        target_shape=target_shape,
        mode=mode,
    )

    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(model, target, dev, [a_np], ["input"], ref_val.shape, dtype)
        tvm.testing.assert_allclose(out, ref_val, rtol=1e-5)


def test_forward_reshape():
    for mode in [0, 1]:
        verify_reshape((20,), (1, 2, 2, 5), mode)
        verify_reshape((1, 3, 20, 20), (1, 12, 10, 10), mode)


def _verify_split(input_dim, out_nums):
    dtype = "float32"

    a_np = np.random.uniform(-100.0, 100.0, size=input_dim).astype(dtype)
    ref_val = np.split(a_np, out_nums, axis=-3)

    inputs = [("input", datatypes.Array(*input_dim))]

    output_names = []
    outputs = []
    output_shapes = []
    for i, out in enumerate(ref_val):
        output_name = "output" + str(i)
        output_names = output_names + [output_name]
        outputs = outputs + [(output_name, datatypes.Array(*out.shape))]
        output_shapes = output_shapes + [out.shape]

    builder = NeuralNetworkBuilder(inputs, outputs)
    builder.add_split(name="split", input_name="input", output_names=output_names)

    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(
            model, target, dev, [a_np], ["input"], output_shapes, [dtype] * len(output_shapes)
        )
        tvm.testing.assert_allclose(out, ref_val, rtol=1e-5)


def test_forward_split():
    """Split"""
    _verify_split(
        (
            1,
            4,
            4,
            4,
        ),
        2,
    )
    _verify_split(
        (
            1,
            3,
            30,
            20,
        ),
        3,
    )


def verify_image_scaler(input_dim, blue_bias=0.0, green_bias=0.0, red_bias=0.0, image_scale=1.0):
    """Verify image scaler"""
    dtype = "float32"
    a_np = np.random.uniform(size=input_dim).astype(dtype)
    # make sure it is valid image format CHW.
    assert len(a_np.shape) == 3 and a_np.shape[0] == 3
    b_np = np.zeros(a_np.shape, dtype=dtype)
    b_np[0, :, :] = image_scale * a_np[0, :, :] + blue_bias
    b_np[1, :, :] = image_scale * a_np[1, :, :] + green_bias
    b_np[2, :, :] = image_scale * a_np[2, :, :] + red_bias
    b_np = np.add(a_np, b_np)
    inputs = [("input1", datatypes.Array(*input_dim)), ("input2", datatypes.Array(*input_dim))]
    output = [("output", datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.set_pre_processing_parameters(
        image_input_names=["input1"],
        is_bgr=True,
        blue_bias=blue_bias,
        green_bias=green_bias,
        red_bias=red_bias,
        image_scale=image_scale,
    )
    # add one add layer to make CoreML model format valid
    # add layer has been tested before.
    builder.add_elementwise(
        name="add", input_names=["input1", "input2"], output_name="output", alpha=0, mode="ADD"
    )
    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(
            model, target, dev, [a_np, a_np], ["input1", "input2"], b_np.shape, dtype
        )
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)


@tvm.testing.uses_gpu
def test_forward_image_scaler():
    verify_image_scaler((3, 224, 224), image_scale=0.17)
    verify_image_scaler(
        (3, 224, 224),
        blue_bias=-1.7669800519943237,
        green_bias=-1.985260009765625,
        red_bias=-2.102560043334961,
        image_scale=0.379,
    )


def verify_convolution(input_dim, filter_, padding):
    """Verify convolution"""
    dtype = "float32"
    _, c, h, width = input_dim
    out_c, _, kernel_h, kernel_w = filter_
    a_np = np.random.uniform(size=input_dim).astype(dtype)
    w_np = np.random.uniform(size=(out_c, c, kernel_h, kernel_w)).astype(dtype)
    w_np_cm = np.transpose(w_np, axes=(2, 3, 1, 0))
    b_np = conv2d_nchw_python(a_np, w_np, [1, 1], padding)
    inputs = [("input1", datatypes.Array(c, h, width))]
    output = [("output", datatypes.Array(*b_np.shape))]  # pylint:disable=not-an-iterable
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_convolution(
        name="conv",
        kernel_channels=3,
        output_channels=out_c,
        height=kernel_h,
        width=kernel_w,
        stride_height=1,
        stride_width=1,
        border_mode=padding.lower(),
        groups=1,
        W=w_np_cm,
        b=None,
        has_bias=False,
        is_deconv=False,
        input_name="input1",
        output_name="output",
    )
    model = cm.models.MLModel(builder.spec)
    for target, dev in tvm.testing.enabled_targets():
        out = run_tvm_graph(model, target, dev, [a_np], ["input1"], output_shape=None)
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)


@tvm.testing.uses_gpu
def test_forward_convolution():
    verify_convolution((1, 3, 224, 224), filter_=(32, 3, 3, 3), padding="VALID")
    verify_convolution((1, 3, 224, 224), filter_=(32, 3, 3, 3), padding="SAME")


def test_can_build_keras_to_coreml_to_relay():
    """Test multiple conversion paths and importing from a saved file."""
    model = keras.models.Sequential()
    model.add(
        keras.layers.Conv2D(
            filters=6,
            kernel_size=(1, 1),
            activation="relu",
            padding="same",
            input_shape=(3, 3, 1),
            data_format="channels_first",
        )
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        kmodel_fn = path.join(tmpdir, "c1mdl.h5")
        model.save(kmodel_fn)

        mdl = cm.convert(
            kmodel_fn, convert_to="neuralnetwork", minimum_deployment_target=cm.target.macOS11
        )
        model_file = path.join(tmpdir, "c1.mlmodel")
        mdl.save(model_file)

        mdl = cm.models.MLModel(model_file)
        desc = mdl.get_spec().description
        iname = desc.input[0].name
        ishape = desc.input[0].type.multiArrayType.shape
        shape_dict = {}
        for i in mdl.get_spec().description.input:
            iname = i.name
            ishape = i.type.multiArrayType.shape
            shape_dict[iname] = ishape
        mod, params = relay.frontend.from_coreml(mdl, shape_dict)

        with tvm.transform.PassContext(opt_level=3):
            relay.build(mod, "llvm", params=params)


if __name__ == "__main__":
    tvm.testing.main()
