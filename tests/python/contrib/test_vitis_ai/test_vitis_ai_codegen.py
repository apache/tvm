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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, W0611, C0413

"""Vitis-AI codegen tests"""

import sys

import numpy as np
import pytest

pytest.importorskip("pyxir")
import pyxir.contrib.target.DPUCADF8H
import pyxir.contrib.target.DPUCAHX8H
import pyxir.contrib.target.DPUCAHX8L
import pyxir.contrib.target.DPUCVDX8G
import pyxir.contrib.target.DPUCVDX8H
import pyxir.contrib.target.DPUCZDX8G
import tvm
from tvm import relay
from tvm.testing import requires_vitis_ai
from tvm.contrib.target import vitis_ai
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.contrib.vitis_ai import annotation

from .infrastructure import verify_codegen


def set_func_attr(func, compile_name, symbol_name):
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", compile_name)
    func = func.with_attr("global_symbol", symbol_name)
    return func


@requires_vitis_ai
@pytest.mark.parametrize(
    "dpu_target",
    ["DPUCADF8H", "DPUCAHX8H-u50", "DPUCAHX8L", "DPUCVDX8H", "DPUCVDX8G", "DPUCZDX8G-zcu104"],
)
def test_conv2d(dpu_target):
    """Test conv2d operator for Vitis AI DPU targets"""

    x = relay.var("x", shape=(1, 3, 224, 224))
    w = relay.const(np.zeros((16, 3, 3, 3), dtype="float32"))
    y = relay.nn.conv2d(x, w, strides=[2, 2], padding=[1, 1, 1, 1], kernel_size=[3, 3])
    func = relay.Function([x], y)
    params = {}
    params["w"] = np.random.rand(16, 3, 3, 3).astype("float32")
    mod = tvm.IRModule()
    mod["main"] = func
    verify_codegen(mod, params=params, dpu_target=dpu_target, tvm_ops=2)


@requires_vitis_ai
@pytest.mark.parametrize("dpu_target", ["DPUCAHX8L", "DPUCZDX8G-zcu104"])
def test_depthwise_conv(dpu_target):
    """Test depthwise_conv operator for Vitis-AI DPUCZDX8G-zcu104 target"""

    dtype = "float32"
    ishape = (1, 32, 14, 14)
    wshape = (32, 1, 3, 3)
    data = relay.var("data", shape=(ishape), dtype=dtype)
    weights = relay.var("weights", shape=(wshape), dtype=dtype)
    depthwise_conv2d = relay.nn.conv2d(data, weights, kernel_size=(3, 3), padding=(1, 1), groups=32)
    func = relay.Function([data, weights], depthwise_conv2d)
    params = {}
    params["weights"] = np.random.randn(32, 1, 3, 3).astype(dtype)
    mod = tvm.IRModule()
    mod["main"] = func
    verify_codegen(mod, params=params, dpu_target=dpu_target, tvm_ops=2)


@requires_vitis_ai
@pytest.mark.parametrize(
    "dpu_target",
    ["DPUCADF8H", "DPUCAHX8H-u50", "DPUCAHX8L", "DPUCVDX8H", "DPUCVDX8G", "DPUCZDX8G-zcu104"],
)
def test_bias_add(dpu_target):
    """Test bias_add operator for Vitis AI DPU targets"""

    dtype = "float32"
    ishape = (1, 32, 14, 14)
    data = relay.var("data", shape=(ishape), dtype=dtype)
    bias = relay.var("bias", relay.TensorType((32,), dtype))
    out = relay.nn.bias_add(data, bias)
    func = relay.Function([data, bias], out)
    params = {}
    params["bias"] = np.random.randn(32).astype(dtype)
    mod = tvm.IRModule()
    mod["main"] = func
    verify_codegen(mod, params=params, dpu_target=dpu_target)


@requires_vitis_ai
@pytest.mark.parametrize(
    "dpu_target",
    ["DPUCADF8H", "DPUCAHX8H-u50", "DPUCAHX8L", "DPUCVDX8H", "DPUCVDX8G", "DPUCZDX8G-zcu104"],
)
def test_relu(dpu_target):
    """Test relu operator for Vitis AI DPU targets"""

    shape = (10, 10)
    x = relay.var("x", shape=shape)
    y = relay.nn.relu(x)
    func = relay.Function([x], y)
    mod = tvm.IRModule()
    mod["main"] = func
    verify_codegen(mod, dpu_target=dpu_target, num_vitis_ai_modules=0, tvm_ops=1)


@requires_vitis_ai
@pytest.mark.parametrize(
    "dpu_target",
    ["DPUCADF8H", "DPUCAHX8H-u50", "DPUCAHX8L", "DPUCVDX8H", "DPUCVDX8G", "DPUCZDX8G-zcu104"],
)
def test_batchnorm(dpu_target):
    """Test batchnorm operator for Vitis AI DPU targets"""

    data = relay.var("data", shape=(1, 16, 112, 112))
    bn_gamma = relay.var("bn_gamma", relay.TensorType((16,), "float32"))
    bn_beta = relay.var("bn_beta", relay.TensorType((16,), "float32"))
    bn_mmean = relay.var("bn_mean", relay.TensorType((16,), "float32"))
    bn_mvar = relay.var("bn_var", relay.TensorType((16,), "float32"))
    bn_output = relay.nn.batch_norm(data, bn_gamma, bn_beta, bn_mmean, bn_mvar)
    func = relay.Function([data, bn_gamma, bn_beta, bn_mmean, bn_mvar], bn_output[0])
    params = {}
    params["bn_gamma"] = np.random.rand(16).astype("float32")
    params["bn_beta"] = np.random.rand(16).astype("float32")
    params["bn_mean"] = np.random.rand(16).astype("float32")
    params["bn_var"] = np.random.rand(16).astype("float32")
    mod = tvm.IRModule()
    mod["main"] = func
    verify_codegen(mod, params=params, dpu_target=dpu_target)


@requires_vitis_ai
@pytest.mark.parametrize(
    "dpu_target",
    ["DPUCADF8H", "DPUCAHX8H-u50", "DPUCAHX8L", "DPUCVDX8H", "DPUCVDX8G", "DPUCZDX8G-zcu104"],
)
def test_add(dpu_target):
    """Test add operator for Vitis AI DPU targets"""

    shape = (10, 10)
    x = relay.var("x", shape=shape)
    y = x + x
    func = relay.Function([x], y)
    mod = tvm.IRModule()
    mod["main"] = func
    verify_codegen(mod, dpu_target=dpu_target)


@requires_vitis_ai
@pytest.mark.parametrize(
    "dpu_target",
    ["DPUCADF8H", "DPUCAHX8H-u50", "DPUCAHX8L", "DPUCVDX8H", "DPUCVDX8G", "DPUCZDX8G-zcu104"],
)
def test_global_avg_pool2d(dpu_target):
    """Test global_avg_pool2d operator for Vitis AI DPU targets"""

    shape = (10, 10, 7, 7)
    x = relay.var("x", shape=shape)
    y = relay.nn.global_avg_pool2d(x)
    func = relay.Function([x], y)
    mod = tvm.IRModule()
    mod["main"] = func
    verify_codegen(mod, dpu_target=dpu_target)


@requires_vitis_ai
@pytest.mark.parametrize(
    "dpu_target",
    ["DPUCADF8H", "DPUCAHX8H-u50", "DPUCAHX8L", "DPUCVDX8H", "DPUCVDX8G", "DPUCZDX8G-zcu104"],
)
def test_avg_pool2d(dpu_target):
    """Test avg_pool2d for operator Vitis AI DPU targets"""

    shape = (10, 10, 10, 10)
    x = relay.var("x", shape=shape)
    y = relay.nn.avg_pool2d(x, pool_size=(3, 3))
    func = relay.Function([x], y)
    mod = tvm.IRModule()
    mod["main"] = func
    verify_codegen(mod, dpu_target=dpu_target)


@requires_vitis_ai
@pytest.mark.parametrize(
    "dpu_target",
    ["DPUCADF8H", "DPUCAHX8H-u50", "DPUCAHX8L", "DPUCVDX8H", "DPUCVDX8G", "DPUCZDX8G-zcu104"],
)
def test_max_pool2d(dpu_target):
    """Test max_pool2d for operator Vitis AI DPU targets"""

    shape = (64, 512, 10, 10)
    x = relay.var("x", shape=shape)
    y = relay.nn.max_pool2d(x, pool_size=(3, 3))
    func = relay.Function([x], y)
    mod = tvm.IRModule()
    mod["main"] = func
    verify_codegen(mod, dpu_target=dpu_target)


@requires_vitis_ai
@pytest.mark.parametrize(
    "dpu_target",
    ["DPUCADF8H", "DPUCAHX8H-u50", "DPUCAHX8L", "DPUCVDX8H", "DPUCVDX8G", "DPUCZDX8G-zcu104"],
)
def test_global_max_pool2d(dpu_target):
    """Test global_maxpool2d operator for Vitis AI DPU targets"""

    shape = (1, 512, 7, 7)
    x = relay.var("x", shape=shape)
    y = relay.nn.global_max_pool2d(x)
    func = relay.Function([x], y)
    mod = tvm.IRModule()
    mod["main"] = func
    verify_codegen(mod, dpu_target=dpu_target)


@requires_vitis_ai
@pytest.mark.parametrize(
    "dpu_target",
    ["DPUCADF8H", "DPUCAHX8H-u50", "DPUCAHX8L", "DPUCVDX8H", "DPUCVDX8G", "DPUCZDX8G-zcu104"],
)
def test_upsampling(dpu_target):
    """Test upsampling operator for Vitis AI DPU targets"""

    shape = (64, 512, 10, 10)
    x = relay.var("x", shape=shape)
    y = relay.nn.upsampling(x, scale_h=2, scale_w=2)
    func = relay.Function([x], y)
    mod = tvm.IRModule()
    mod["main"] = func
    verify_codegen(mod, dpu_target=dpu_target)


@pytest.mark.skip(
    reason="I and O used to be mixed up in kernel layouts in TVM."
    "This is fixed, but vitis needs to adopt the new convention."
    "To change, simply remove this line:"
    "https://github.com/Xilinx/pyxir/blob/bef661d6d77adcdbd2cf4163f2cf3a1d31d40406/"
    "python/pyxir/frontend/tvm/relay_tools/relay_l2_convolution.py#L380"
)
@pytest.mark.parametrize(
    "dpu_target",
    ["DPUCADF8H", "DPUCAHX8H-u50", "DPUCAHX8L", "DPUCVDX8H", "DPUCVDX8G", "DPUCZDX8G-zcu104"],
)
def test_conv2d_transpose(dpu_target):
    """Test conv2d_transpose operator for Vitis AI DPU targets"""

    dshape = (1, 3, 18, 18)
    kshape = (3, 10, 3, 3)
    x = relay.var("x", shape=dshape)
    w = relay.const(np.zeros(kshape, dtype="float32"))
    y = relay.nn.conv2d_transpose(
        x,
        w,
        channels=10,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=(1, 1),
        data_layout="NCHW",
        kernel_layout="IOHW",
    )
    func = relay.Function([x], y)
    params = {}
    dtype = "float32"
    params["w"] = np.random.uniform(size=kshape).astype(dtype)
    mod = tvm.IRModule()
    mod["main"] = func
    verify_codegen(mod, params=params, dpu_target=dpu_target)


@requires_vitis_ai
@pytest.mark.parametrize(
    "dpu_target",
    ["DPUCADF8H", "DPUCAHX8H-u50", "DPUCAHX8L", "DPUCVDX8H", "DPUCVDX8G", "DPUCZDX8G-zcu104"],
)
def test_annotate(dpu_target):
    """Test annotation operator for Vitis AI DPU targets"""

    def partition(dpu_target):
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("weight", relay.TensorType((16, 3, 3, 3), "float32"))
        bn_gamma = relay.var("bn_gamma", relay.TensorType((16,), "float32"))
        bn_beta = relay.var("bn_beta", relay.TensorType((16,), "float32"))
        bn_mmean = relay.var("bn_mean", relay.TensorType((16,), "float32"))
        bn_mvar = relay.var("bn_var", relay.TensorType((16,), "float32"))

        conv = relay.nn.conv2d(
            data=data, weight=weight, kernel_size=(3, 3), channels=16, padding=(1, 1)
        )
        bn_output = relay.nn.batch_norm(conv, bn_gamma, bn_beta, bn_mmean, bn_mvar)

        func = relay.Function(
            [data, weight, bn_gamma, bn_beta, bn_mmean, bn_mvar], bn_output.astuple()
        )
        mod = tvm.IRModule()
        mod["main"] = func
        params = {}
        params["weight"] = np.random.rand(16, 3, 3, 3).astype("float32")
        params["bn_gamma"] = np.random.rand(16).astype("float32")
        params["bn_beta"] = np.random.rand(16).astype("float32")
        params["bn_mean"] = np.random.rand(16).astype("float32")
        params["bn_var"] = np.random.rand(16).astype("float32")
        mod = annotation(mod, params, dpu_target)

        opt_pass = tvm.transform.Sequential(
            [
                transform.MergeCompilerRegions(),
                transform.PartitionGraph(),
            ]
        )

        with tvm.transform.PassContext(opt_level=3):
            mod = opt_pass(mod)
        return mod

    def expected():
        # function variables for conv2d
        data0 = relay.var("data0", relay.TensorType((1, 3, 224, 224), "float32"))
        weight0 = relay.var("weight0", relay.TensorType((16, 3, 3, 3), "float32"))
        conv = relay.nn.conv2d(
            data=data0, weight=weight0, kernel_size=(3, 3), channels=16, padding=(1, 1)
        )

        # function variables for batch_norm
        bn_gamma0 = relay.var("bn_gamma0", relay.TensorType((16,), "float32"))
        bn_beta0 = relay.var("bn_beta0", relay.TensorType((16,), "float32"))
        bn_mmean0 = relay.var("bn_mean0", relay.TensorType((16,), "float32"))
        bn_mvar0 = relay.var("bn_var0", relay.TensorType((16,), "float32"))
        bn = relay.nn.batch_norm(conv, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0)
        func0 = relay.Function(
            [data0, weight0, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0], bn.astuple()
        )
        func0 = set_func_attr(func0, "vitis_ai", "tvmgen_default_vitis_ai_main_0")
        gv0 = relay.GlobalVar("tvmgen_default_vitis_ai_main_0")
        mod = tvm.IRModule()
        mod[gv0] = func0
        mod = relay.transform.InferType()(mod)

        # main function
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("weight", relay.TensorType((16, 3, 3, 3), "float32"))
        bn_gamma = relay.var("bn_gamma", relay.TensorType((16,), "float32"))
        bn_beta = relay.var("bn_beta", relay.TensorType((16,), "float32"))
        bn_mmean = relay.var("bn_mean", relay.TensorType((16,), "float32"))
        bn_mvar = relay.var("bn_var", relay.TensorType((16,), "float32"))
        call0 = gv0(data, weight, bn_gamma, bn_beta, bn_mmean, bn_mvar)
        mod["main"] = relay.Function([data, weight, bn_gamma, bn_beta, bn_mmean, bn_mvar], call0)
        mod = relay.transform.InferType()(mod)
        return mod

    partitioned_mod = partition(dpu_target)

    ref_mod = expected()

    tvm.ir.assert_structural_equal(partitioned_mod, ref_mod, map_free_vars=True)


if __name__ == "__main__":
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        sys.exit(0)
    tvm.testing.main()
