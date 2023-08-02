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
from tvm.relax.testing import get_relax_matmul_module
from tvm.script import relax as R

from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import relax as relax_builder


# @pytest.fixture(autouse=True)
# def reset_seed():
#     np.random.seed(0)

# has_cudnn = tvm.get_global_func("relax.ext.cudnn", False)

# cudnn_enabled = pytest.mark.skipif(
#     not has_cudnn,
#     reason="cuDNN not enabled.",
# )

# pytestmark = [cudnn_enabled]


_activation_table = {
    "none": None,
    "bias": None,
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
):
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            data = R.arg("data", R.Tensor(data_shape, dtype))
            weight = R.arg("weight", R.Tensor(weight_shape, dtype))
            if with_bias:
                bias = R.arg("bias", R.Tensor((1, 1, 1, weight_shape[0]), dtype))

            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.conv2d(
                        data,
                        weight,
                        out_dtype=dtype,
                        padding=(1, 1),
                        data_layout="NHWC",
                        kernel_layout="OHWI",
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


def build_and_run(mod, inputs_np, target, legalize=False, cuda_graph=False):
    if legalize:
        mod = relax.transform.LegalizeOps()(mod)

    dev = tvm.device(target, 0)
    with tvm.transform.PassContext(config={"relax.backend.use_cuda_graph": cuda_graph}):
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
def test_cudnn_partition_conv2d_without_bias(data_shape, weight_shape, dtype, with_bias, activation):
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
    assert mod["main"].body.blocks[0].bindings[0].value.op.name_hint == 'fused_relax_nn_conv2d_cudnn'


if __name__ == "__main__":
    tvm.testing.main()
