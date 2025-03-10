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
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm import dlight


@tvm.testing.requires_cuda_compute_version(8, 9)
@pytest.mark.parametrize("original_dtype", ["float16", "float32"])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2", "float16"])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_fp8_matmul_compile(dtype, original_dtype, batch_size):
    bb = relax.BlockBuilder()
    batch = T.int64(batch_size)
    x = relax.Var("x", R.Tensor((batch, 784), original_dtype))
    weight = relax.const(np.random.randn(784, 128), original_dtype)

    with bb.function("forward", [x]):
        with bb.dataflow():
            lv1 = bb.emit(relax.op.astype(x, dtype))
            lv2 = bb.emit(relax.op.astype(weight, dtype))
            lv3 = bb.emit(relax.op.matmul(lv1, lv2, dtype))
            lv4 = bb.emit(relax.op.astype(lv3, original_dtype))
            gv = bb.emit_output(lv4)
        bb.emit_func_output(gv)

    mod = bb.get()
    mod.show()

    dev = tvm.device("cuda", 0)
    target = tvm.target.Target.from_device(dev)

    with target:
        mod = relax.get_pipeline("zero")(mod)
        mod = dlight.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dlight.gpu.Matmul(),
            dlight.gpu.GEMV(),
            dlight.gpu.Reduction(),
            dlight.gpu.GeneralReduction(),
            dlight.gpu.Fallback(),
        )(mod)

    _exe = relax.build(mod, target)


@tvm.testing.requires_cuda_compute_version(8, 9)
@pytest.mark.parametrize("original_dtype", ["float16", "float32"])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_fp8_conv2d_compile(dtype, original_dtype, batch_size):
    bb = relax.BlockBuilder()
    batch = batch_size
    x = relax.Var("x", R.Tensor((batch, 1, 28, 28), original_dtype))
    weight = relax.const(np.random.randn(32, 1, 3, 3), original_dtype)

    with bb.function("forward", [x]):
        with bb.dataflow():
            lv1 = bb.emit(relax.op.astype(x, dtype))
            lv2 = bb.emit(relax.op.astype(weight, dtype))
            lv3 = bb.emit(
                relax.op.nn.conv2d(
                    lv1,
                    lv2,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
            )
            lv4 = bb.emit(relax.op.astype(lv3, original_dtype))
            gv = bb.emit_output(lv4)
        bb.emit_func_output(gv)

    mod = bb.get()
    mod.show()

    dev = tvm.device("cuda", 0)
    target = tvm.target.Target.from_device(dev)

    with target:
        mod = relax.get_pipeline("zero")(mod)
        mod = dlight.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dlight.gpu.Matmul(),
            dlight.gpu.GEMV(),
            dlight.gpu.Reduction(),
            dlight.gpu.GeneralReduction(),
            dlight.gpu.Fallback(),
        )(mod)

    _exe = relax.build(mod, target)


@tvm.testing.requires_cuda_compute_version(8, 9)
@pytest.mark.parametrize("original_dtype", ["float16", "float32"])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_fp8_maxpool2d_compile(dtype, original_dtype, batch_size):
    bb = relax.BlockBuilder()
    batch = batch_size
    x = relax.Var("x", R.Tensor((batch, 1, 28, 28), original_dtype))

    with bb.function("forward", [x]):
        with bb.dataflow():
            lv1 = bb.emit(relax.op.astype(x, dtype))
            lv3 = bb.emit(
                relax.op.nn.max_pool2d(
                    lv1,
                    pool_size=[3, 3],
                    strides=[2, 2],
                    dilation=[1, 1],
                    padding=[1, 1, 1, 1],
                    ceil_mode=False,
                    count_include_pad=False,
                    layout="NCHW",
                    out_layout="NCHW",
                )
            )
            lv4 = bb.emit(relax.op.astype(lv3, original_dtype))
            gv = bb.emit_output(lv4)
        bb.emit_func_output(gv)

    mod = bb.get()
    mod.show()

    dev = tvm.device("cuda", 0)
    target = tvm.target.Target.from_device(dev)

    with target:
        mod = relax.get_pipeline("zero")(mod)
        mod = dlight.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dlight.gpu.Matmul(),
            dlight.gpu.GEMV(),
            dlight.gpu.Reduction(),
            dlight.gpu.GeneralReduction(),
            dlight.gpu.Fallback(),
        )(mod)

    _exe = relax.build(mod, target)


@tvm.testing.requires_cuda_compute_version(8, 9)
@pytest.mark.parametrize("original_dtype", ["float16", "float32"])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_fp8_add_compile(dtype, original_dtype, batch_size):
    bb = relax.BlockBuilder()
    batch = batch_size
    x = relax.Var("x", R.Tensor((batch, 784), original_dtype))
    bias = relax.const(np.random.randn(784), original_dtype)

    with bb.function("forward", [x]):
        with bb.dataflow():
            lv1 = bb.emit(relax.op.astype(x, dtype))
            lv2 = bb.emit(relax.op.astype(bias, dtype))
            lv3 = bb.emit(relax.op.add(lv1, lv2))
            lv4 = bb.emit(relax.op.astype(lv3, original_dtype))
            gv = bb.emit_output(lv4)
        bb.emit_func_output(gv)

    mod = bb.get()
    mod.show()

    dev = tvm.device("cuda", 0)
    target = tvm.target.Target.from_device(dev)

    with target:
        mod = relax.get_pipeline("zero")(mod)
        mod = dlight.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dlight.gpu.Matmul(),
            dlight.gpu.GEMV(),
            dlight.gpu.Reduction(),
            dlight.gpu.GeneralReduction(),
            dlight.gpu.Fallback(),
        )(mod)

    _exe = relax.build(mod, target)


@tvm.testing.requires_cuda_compute_version(8, 9)
@pytest.mark.parametrize("original_dtype", ["float16", "float32"])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_fp8_relu_compile(dtype, original_dtype, batch_size):
    bb = relax.BlockBuilder()
    batch = batch_size
    x = relax.Var("x", R.Tensor((batch, 784), original_dtype))

    with bb.function("forward", [x]):
        with bb.dataflow():
            lv1 = bb.emit(relax.op.astype(x, dtype))
            lv2 = bb.emit(relax.op.nn.relu(lv1))
            lv3 = bb.emit(relax.op.astype(lv2, original_dtype))
            gv = bb.emit_output(lv3)
        bb.emit_func_output(gv)

    mod = bb.get()
    mod.show()

    dev = tvm.device("cuda", 0)
    target = tvm.target.Target.from_device(dev)

    with target:
        mod = relax.get_pipeline("zero")(mod)
        mod = dlight.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dlight.gpu.Matmul(),
            dlight.gpu.GEMV(),
            dlight.gpu.Reduction(),
            dlight.gpu.GeneralReduction(),
            dlight.gpu.Fallback(),
        )(mod)

    _exe = relax.build(mod, target)


@tvm.testing.requires_cuda_compute_version(8, 9)
@pytest.mark.parametrize("original_dtype", ["float16", "float32"])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_fp8_gelu_compile(dtype, original_dtype, batch_size):
    bb = relax.BlockBuilder()
    batch = batch_size
    x = relax.Var("x", R.Tensor((batch, 784), original_dtype))

    with bb.function("forward", [x]):
        with bb.dataflow():
            lv1 = bb.emit(relax.op.astype(x, dtype))
            lv2 = bb.emit(relax.op.nn.gelu(lv1))
            lv3 = bb.emit(relax.op.astype(lv2, original_dtype))
            gv = bb.emit_output(lv3)
        bb.emit_func_output(gv)

    mod = bb.get()
    mod.show()

    dev = tvm.device("cuda", 0)
    target = tvm.target.Target.from_device(dev)

    with target:
        mod = relax.get_pipeline("zero")(mod)
        mod = dlight.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dlight.gpu.Matmul(),
            dlight.gpu.GEMV(),
            dlight.gpu.Reduction(),
            dlight.gpu.GeneralReduction(),
            dlight.gpu.Fallback(),
        )(mod)

    _exe = relax.build(mod, target)


@tvm.testing.requires_cuda_compute_version(8, 9)
@pytest.mark.parametrize("original_dtype", ["float16", "float32"])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_fp8_gelu_tanh_compile(dtype, original_dtype, batch_size):
    bb = relax.BlockBuilder()
    batch = batch_size
    x = relax.Var("x", R.Tensor((batch, 784), original_dtype))

    with bb.function("forward", [x]):
        with bb.dataflow():
            lv1 = bb.emit(relax.op.astype(x, dtype))
            lv2 = bb.emit(relax.op.nn.gelu(lv1))
            lv3 = bb.emit(relax.op.astype(lv2, original_dtype))
            gv = bb.emit_output(lv3)
        bb.emit_func_output(gv)

    mod = bb.get()
    mod.show()

    dev = tvm.device("cuda", 0)
    target = tvm.target.Target.from_device(dev)

    with target:
        mod = relax.get_pipeline("zero")(mod)
        mod = dlight.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dlight.gpu.Matmul(),
            dlight.gpu.GEMV(),
            dlight.gpu.Reduction(),
            dlight.gpu.GeneralReduction(),
            dlight.gpu.Fallback(),
        )(mod)

    _exe = relax.build(mod, target)


@tvm.testing.requires_cuda_compute_version(8, 9)
@pytest.mark.parametrize("original_dtype", ["float16", "float32"])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_fp8_sigmoid_compile(dtype, original_dtype, batch_size):
    bb = relax.BlockBuilder()
    batch = batch_size
    x = relax.Var("x", R.Tensor((batch, 784), original_dtype))

    with bb.function("forward", [x]):
        with bb.dataflow():
            lv1 = bb.emit(relax.op.astype(x, dtype))
            lv2 = bb.emit(relax.op.nn.silu(lv1))
            lv3 = bb.emit(relax.op.astype(lv2, original_dtype))
            gv = bb.emit_output(lv3)
        bb.emit_func_output(gv)

    mod = bb.get()
    mod.show()

    dev = tvm.device("cuda", 0)
    target = tvm.target.Target.from_device(dev)

    with target:
        mod = relax.get_pipeline("zero")(mod)
        mod = dlight.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dlight.gpu.Matmul(),
            dlight.gpu.GEMV(),
            dlight.gpu.Reduction(),
            dlight.gpu.GeneralReduction(),
            dlight.gpu.Fallback(),
        )(mod)

    _exe = relax.build(mod, target)


@tvm.testing.requires_cuda_compute_version(8, 9)
@pytest.mark.parametrize("original_dtype", ["float16", "float32"])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_fp8_silu_compile(dtype, original_dtype, batch_size):
    bb = relax.BlockBuilder()
    batch = batch_size
    x = relax.Var("x", R.Tensor((batch, 784), original_dtype))

    with bb.function("forward", [x]):
        with bb.dataflow():
            lv1 = bb.emit(relax.op.astype(x, dtype))
            lv2 = bb.emit(relax.op.nn.silu(lv1))
            lv3 = bb.emit(relax.op.astype(lv2, original_dtype))
            gv = bb.emit_output(lv3)
        bb.emit_func_output(gv)

    mod = bb.get()
    mod.show()

    dev = tvm.device("cuda", 0)
    target = tvm.target.Target.from_device(dev)

    with target:
        mod = relax.get_pipeline("zero")(mod)
        mod = dlight.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dlight.gpu.Matmul(),
            dlight.gpu.GEMV(),
            dlight.gpu.Reduction(),
            dlight.gpu.GeneralReduction(),
            dlight.gpu.Fallback(),
        )(mod)

    _exe = relax.build(mod, target)


@tvm.testing.requires_cuda_compute_version(8, 9)
@pytest.mark.parametrize("original_dtype", ["float16", "float32"])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_fp8_softmax_compile(dtype, original_dtype, batch_size):
    bb = relax.BlockBuilder()
    batch = batch_size
    x = relax.Var("x", R.Tensor((batch, 784), original_dtype))

    with bb.function("forward", [x]):
        with bb.dataflow():
            lv1 = bb.emit(relax.op.astype(x, dtype))
            lv2 = bb.emit(relax.op.nn.softmax(lv1))
            lv3 = bb.emit(relax.op.astype(lv2, original_dtype))
            gv = bb.emit_output(lv3)
        bb.emit_func_output(gv)

    mod = bb.get()
    mod.show()

    dev = tvm.device("cuda", 0)
    target = tvm.target.Target.from_device(dev)

    with target:
        mod = relax.get_pipeline("zero")(mod)
        mod = dlight.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dlight.gpu.Matmul(),
            dlight.gpu.GEMV(),
            dlight.gpu.Reduction(),
            dlight.gpu.GeneralReduction(),
            dlight.gpu.Fallback(),
        )(mod)

    _exe = relax.build(mod, target)


if __name__ == "__main__":
    tvm.testing.main()
