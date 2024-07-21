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
import pytest

import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relax
from tvm.relax.backend.contrib.cudnn import partition_for_cudnn
from tvm.relax.testing import get_relax_matmul_module, get_relax_stacked_attention_module
from tvm.contrib.pickle_memoize import memoize
from tvm.script import relax as R

from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import relax as relax_builder


@pytest.fixture(autouse=True)
def reset_seed():
    np.random.seed(0)


pytestmark = tvm.testing.requires_cudnn.marks()


_activation_table = {
    "none": None,
    "relu": R.nn.relu,
    "gelu": R.nn.gelu,
    "silu": R.nn.silu,
}


def get_relax_conv2d_module(
    data_shape,
    weight_shape,
    dtype,
    with_bias=False,
    activation=None,
    residual_bin_op=None,
    residual_activation=None,
    data_layout="NHWC",
    kernel_layout="OHWI",
):
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            data = R.arg("data", R.Tensor(data_shape, dtype))
            weight = R.arg("weight", R.Tensor(weight_shape, dtype))
            if with_bias:
                if data_layout == "NHWC":
                    bias = R.arg("bias", R.Tensor((1, 1, 1, weight_shape[0]), dtype))
                elif data_layout == "NCHW":
                    bias = R.arg("bias", R.Tensor((1, weight_shape[0], 1, 1), dtype))
                else:
                    raise ValueError("Unsupported data_layout: {}".format(data_layout))

            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.conv2d(
                        data,
                        weight,
                        out_dtype=dtype,
                        padding=(1, 1),
                        data_layout=data_layout,
                        kernel_layout=kernel_layout,
                    )
                )
                if with_bias:
                    output = R.emit(output + bias)
                if activation is not None:
                    output = R.emit(activation(output))
                if residual_bin_op is not None:
                    output = R.emit(residual_bin_op(output, data))
                    if residual_activation is not None:
                        output = R.emit(residual_activation(output))
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


def get_result_with_relax_cudnn_offload(mod, np_inputs, cuda_graph=False):
    mod = partition_for_cudnn(mod)
    mod = relax.transform.RunCodegen()(mod)
    return build_and_run(mod, np_inputs, "cuda", cuda_graph=cuda_graph)


def build_and_run(mod, inputs_np, target, legalize=False, cuda_graph=False):
    dev = tvm.device(target, 0)
    with tvm.transform.PassContext(
        config={
            "relax.backend.use_cuda_graph": cuda_graph,
            "relax.transform.apply_legalize_ops": legalize,
        }
    ):
        ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    inputs = [tvm.nd.array(inp, dev) for inp in inputs_np]

    # For cuda graph, run the compiled function twice to make sure that we can launch the cached
    # graph on the second run.
    if cuda_graph:
        f(*inputs)

    return f(*inputs).numpy()


@pytest.mark.parametrize(
    "data_shape, weight_shape, dtype, with_bias, activation",
    [
        # Regular
        ((16, 32, 32, 16), (32, 3, 3, 16), "float16", False, "none"),
    ],
)
def test_cudnn_partition_conv2d_without_bias(
    data_shape, weight_shape, dtype, with_bias, activation
):
    low, high = -1, 1
    data = np.random.randint(low, high, size=data_shape).astype(dtype)
    weight = np.random.randint(low, high, size=weight_shape).astype(dtype)
    bias = np.random.randint(low, high, size=(1, 1, 1, weight_shape[0])).astype(dtype)
    activation = _activation_table[activation]
    if with_bias:
        args = (data, weight, bias)
    else:
        args = (data, weight)
    mod = get_relax_conv2d_module(
        data_shape,
        weight_shape,
        dtype,
        with_bias=with_bias,
        activation=activation,
    )
    mod = partition_for_cudnn(mod)
    assert (
        mod["main"].body.blocks[0].bindings[0].value.op.name_hint == "fused_relax_nn_conv2d_cudnn"
    )


@pytest.mark.parametrize(
    "data_shape, weight_shape, dtype, with_bias, activation",
    [
        # Regular
        ((16, 32, 32, 16), (32, 3, 3, 16), "float32", False, "none"),
        # Bias
        ((16, 32, 32, 16), (32, 3, 3, 16), "float32", True, "none"),
        # Bias+ReLU
        ((16, 32, 32, 16), (32, 3, 3, 16), "float32", True, "relu"),
        # Bias+ReLU+half
        ((16, 32, 32, 16), (32, 3, 3, 16), "float16", True, "relu"),
    ],
)
def test_conv2d_offload(data_shape, weight_shape, dtype, with_bias, activation):
    input = np.random.randn(*data_shape).astype(dtype)
    weight = np.random.randn(*weight_shape).astype(dtype)

    if with_bias:
        oc = weight_shape[0]
        bias = np.random.randn(1, 1, 1, oc).astype(dtype)
        args = (input, weight, bias)
    else:
        bias = None
        args = (input, weight)

    activation = _activation_table[activation]

    mod = get_relax_conv2d_module(
        data_shape,
        weight_shape,
        dtype,
        with_bias=with_bias,
        activation=activation,
    )

    out = get_result_with_relax_cudnn_offload(mod, args)
    ref = build_and_run(mod, args, "llvm", legalize=True)
    if dtype == "float16":
        tvm.testing.assert_allclose(out, ref, rtol=1e-1, atol=1e-1)
    else:
        tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skip(reason="flaky test")
@pytest.mark.parametrize(
    "data_shape, weight_shape, dtype, with_bias, activation",
    [
        # Regular
        ((16, 16, 32, 32), (32, 16, 3, 3), "float32", False, "none"),
        # Bias
        ((16, 16, 32, 32), (32, 16, 3, 3), "float32", True, "none"),
        # Bias+ReLU
        ((16, 16, 32, 32), (32, 16, 3, 3), "float32", True, "relu"),
        # Bias+ReLU+half
        ((16, 16, 32, 32), (32, 16, 3, 3), "float16", True, "relu"),
    ],
)
def test_conv2d_nchw_oihw_offload(data_shape, weight_shape, dtype, with_bias, activation):
    input = np.random.randn(*data_shape).astype(dtype)
    weight = np.random.randn(*weight_shape).astype(dtype)

    if with_bias:
        oc = weight_shape[0]
        bias = np.random.randn(1, oc, 1, 1).astype(dtype)
        args = (input, weight, bias)
    else:
        bias = None
        args = (input, weight)

    activation = _activation_table[activation]

    mod = get_relax_conv2d_module(
        data_shape,
        weight_shape,
        dtype,
        with_bias=with_bias,
        activation=activation,
        data_layout="NCHW",
        kernel_layout="OIHW",
    )

    out = get_result_with_relax_cudnn_offload(mod, args)
    ref = build_and_run(mod, args, "llvm", legalize=True)
    if dtype == "float16":
        tvm.testing.assert_allclose(out, ref, rtol=1e-1, atol=1e-1)
    else:
        tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


@memoize("topi.tests.test_codegen_cudnn.test_stacked_attention_offload")
def get_numpy_stacked_attention_ref(b, s, n, h, h_v, bias_shape, qk_scale, dtype, layout):
    if layout == "BS3NH":
        qkv = np.random.randn(b, s, n * h * 2 + n * h_v).astype(dtype)
        split_qkv = np.split(qkv, [n * h, n * h * 2], axis=2)
        q = split_qkv[0].reshape(b, s, n, h)
        k = split_qkv[1].reshape(b, s, n, h)
        v = split_qkv[2].reshape(b, s, n, h_v)
        layout = "BSNH"
    elif layout == "SBN3H":
        qkv = np.random.randn(s, b, n, h * 2 + h_v).astype(dtype)
        q, k, v = np.split(qkv, [h, h * 2], axis=3)
        layout = "SBNH"
    else:
        raise ValueError("Unsupported layout: {}".format(layout))
    if not bias_shape == "none":
        bias = np.random.randn(*bias_shape).astype(dtype)
        score = score + bias  # b, n, s, s
    else:
        bias = None
    ref = tvm.topi.testing.attention_python(q, k, v, bias, qk_scale, "none", None, layout)
    return qkv, bias, ref


@pytest.fixture(
    params=[
        # B, S, N, H, bias_shape scale, single_shape, layout
        (4, 8, 32, (64, 32), "none", 1.0, False, "BS3NH"),
        (4, 8, 32, (64, 64), "none", "none", True, "BS3NH"),
        (4, 8, 32, (64, 32), "none", 1.0, False, "SBN3H"),
        (4, 8, 32, (64, 64), "none", "none", True, "SBN3H"),
    ]
)
def stacked_attention_size(request):
    return request.param


@pytest.mark.skip(reason="require cudnn frontend")
def test_stacked_attention_split_offload(stacked_attention_size):
    b, s, n, (h, h_v), bias_shape, scale, single_shape, layout = stacked_attention_size
    qkv, bias, ref = get_numpy_stacked_attention_ref(
        b, s, n, h, h_v, bias_shape, scale, "float16", layout
    )
    if scale == "none":
        mod = get_relax_stacked_attention_module(
            qkv, b, s, n, h, h_v, "split", bias, single_shape=single_shape, layout=layout
        )
        scale = 1.0 / np.sqrt(h)
    else:
        mod = get_relax_stacked_attention_module(
            qkv, b, s, n, h, h_v, "split", bias, scale, single_shape=single_shape, layout=layout
        )

    if bias is None:
        out = get_result_with_relax_cudnn_offload(mod, [qkv])
    else:
        out = get_result_with_relax_cudnn_offload(mod, [qkv, bias])
    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=2e-2)


if __name__ == "__main__":
    tvm.testing.main()
