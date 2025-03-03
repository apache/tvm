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

from __future__ import annotations
import tempfile

from tvm import relax, runtime
import tvm
import tvm.testing
from tvm import relax
import scipy
from scipy.special import erf
import numpy as np
from tvm.target import Target
from tvm.relax.vm_build import build as relax_build
from tvm.script.ir_builder import relax as R
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import tir as T
from tvm.script.ir_builder import IRBuilder

from tvm.relax.backend_tir import get_tir_pattern
from tvm.relax.backend_tir.contrib.cutlass import cutlass_fcodegen, compile_options

A_TYPE = "float16"
B_TYPE = "float16"
C_TYPE = "float16"

target = Target("cuda")


def f_run(rt_mod: runtime.Module, device: runtime.ndarray.Device, *input):
    vm = relax.vm.VirtualMachine(rt_mod=rt_mod, device=device)
    return vm["main"](*input)


def build(mod):
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.FuseTIR()(mod)
    mod = relax.transform.SplitCallTIRByPattern(get_tir_pattern(), cutlass_fcodegen())(mod)
    mod = relax.transform.DeadCodeElimination()(mod)
    print(mod.script())
    f = tempfile.NamedTemporaryFile(suffix=".so", delete=True)
    executable = relax_build(mod, target)

    executable.mod.export_library(f.name, **compile_options(target))
    rt_mod = runtime.load_module(f.name)
    f.close()
    return rt_mod


def build_and_run_reference(mod, inputs_np):
    dev = tvm.device("llvm", 0)
    ex = relax.build(mod, "llvm")
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    inputs = [tvm.nd.array(inp, dev) for inp in inputs_np]
    return f(*inputs).numpy()


def constructGEMM(M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    R.output(C)
                (C,) = df.output_vars
                R.func_ret_value(C)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_dense():
    m, n, k = 128, 64, 256
    executable = build(constructGEMM(m, n, k))
    dev = tvm.cuda()
    A = np.random.randn(m, k).astype("float16")
    B = np.random.randn(k, n).astype("float16")
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=5e-2, atol=5e-2)


def constructGEMM_bias(M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, N), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    relax_mod = ib.get()
    return relax_mod


def constructGEMM_bias2(M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((N,), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_dense_bias():
    m, n, k = 128, 64, 256
    executable = build(constructGEMM_bias(m, n, k))
    dev = tvm.cuda()
    A = np.random.randn(m, k).astype("float16")
    B = np.random.randn(k, n).astype("float16")
    bias = np.random.randn(1, n).astype("float16")
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=5e-2, atol=5e-2)


@tvm.testing.requires_cutlass
def test_cutlass_dense_bias2():
    m, n, k = 128, 64, 256
    executable = build(constructGEMM_bias2(m, n, k))
    dev = tvm.cuda()
    A = np.random.randn(m, k).astype("float16")
    B = np.random.randn(k, n).astype("float16")
    bias = np.random.randn(n).astype("float16")
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=5e-2, atol=5e-2)


def constructGEMM_bias_relu(M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, N), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    E = R.emit(R.nn.relu(D))
                    R.output(E)
                (E,) = df.output_vars
                R.func_ret_value(E)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_dense_bias_relu():
    m, n, k = 128, 64, 256
    executable = build(constructGEMM_bias_relu(m, n, k))
    dev = tvm.cuda()
    A = np.random.randn(m, k).astype("float16")
    B = np.random.randn(k, n).astype("float16")
    bias = np.random.randn(1, n).astype("float16")
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), np.maximum(A @ B + bias, 0), rtol=5e-2, atol=5e-2)


def constructBatchGEMM(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    R.output(C)
                (C,) = df.output_vars
                R.func_ret_value(C)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense():
    b, m, n, k = 2, 128, 256, 64
    executable = build(constructBatchGEMM(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.randn(b, m, k).astype("float16")
    B = np.random.randn(k, n).astype("float16")
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=5e-2, atol=5e-2)


def constructBatchGEMM2(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((batch, K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    R.output(C)
                (C,) = df.output_vars
                R.func_ret_value(C)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense2():
    b, m, n, k = 2, 128, 256, 64
    executable = build(constructBatchGEMM2(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.randn(b, m, k).astype("float16")
    B = np.random.randn(b, k, n).astype("float16")
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=5e-2, atol=5e-2)


def constructBatchGEMM_bias(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, N), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense_bias():
    b, m, n, k = 2, 128, 256, 64
    executable = build(constructBatchGEMM_bias(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.randn(b, m, k).astype("float16")
    B = np.random.randn(k, n).astype("float16")
    bias = np.random.randn(1, n).astype("float16")
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=5e-2, atol=5e-2)


def constructBatchGEMM_bias2(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((N,), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense_bias2():
    b, m, n, k = 2, 128, 256, 64
    executable = build(constructBatchGEMM_bias2(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.randn(b, m, k).astype("float16")
    B = np.random.randn(k, n).astype("float16")
    bias = np.random.randn(n).astype("float16")
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=5e-2, atol=5e-2)


def constructBatchGEMM_bias2_gelu(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((N,), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    E = R.emit(R.nn.gelu(D))
                    R.output(E)
                (E,) = df.output_vars
                R.func_ret_value(E)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense_bias2_gelu():
    b, m, n, k = 2, 128, 64, 256
    executable = build(constructBatchGEMM_bias2_gelu(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.randn(b, m, k).astype("float16")
    B = np.random.randn(k, n).astype("float16")
    bias = np.random.randn(n).astype("float16")
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    C = A @ B + bias
    O = 0.5 * C * (1 + erf(C / np.sqrt(2)))
    np.testing.assert_allclose(result.numpy(), O, rtol=5e-2, atol=5e-2)


def constructBatchGEMM_bias2_mul(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((N,), A_TYPE)
                )  # pylint: disable=invalid-name
                residual = R.arg("residual", relax.TensorStructInfo((batch, M, N), A_TYPE))
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    E = R.emit(R.multiply(D, residual))
                    R.output(E)
                (E,) = df.output_vars
                R.func_ret_value(E)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense_bias2_mul():
    b, m, n, k = 2, 128, 256, 64
    executable = build(constructBatchGEMM_bias2_mul(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.randn(b, m, k).astype("float16")
    B = np.random.randn(k, n).astype("float16")
    bias = np.random.randn(n).astype("float16")
    residual = np.random.randn(b, m, n).astype("float16")
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    residual_tvm = tvm.nd.array(residual, dev)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm, residual_tvm)
    np.testing.assert_allclose(result.numpy(), ((A @ B) + bias) * residual, rtol=5e-2, atol=5e-2)


def constructBatchGEMM2_bias(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((batch, K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, N), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense2_bias():
    b, m, n, k = 2, 128, 256, 64
    executable = build(constructBatchGEMM2_bias(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.randn(b, m, k).astype("float16")
    B = np.random.randn(b, k, n).astype("float16")
    bias = np.random.randn(1, n).astype("float16")
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=5e-2, atol=5e-2)


def constructConv2D(N, C, H, W, KH, KW, O, strides, padding, dilation):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                x = R.arg(
                    "x", relax.TensorStructInfo((N, H, W, C), A_TYPE)
                )  # pylint: disable=invalid-name
                w = R.arg(
                    "w", relax.TensorStructInfo((O, KH, KW, C), B_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(
                        R.nn.conv2d(
                            x,
                            w,
                            strides=strides,
                            padding=padding,
                            dilation=dilation,
                            groups=1,
                            data_layout="NHWC",
                            kernel_layout="OHWI",
                            out_layout="NHWC",
                            out_dtype=C_TYPE,
                        )
                    )
                    R.output(C)
                (C,) = df.output_vars
                R.func_ret_value(C)
    mod = ib.get()
    return mod


@tvm.testing.requires_cutlass
def test_cutlass_conv2d():
    n, c, h, w = 1, 3, 224, 224
    kh, kw, o = 3, 3, 64
    for strides in [(1, 1), (2, 2)]:
        for padding in [(0, 0), (3, 3)]:
            for dilation in [(1, 1), (4, 4)]:
                mod = constructConv2D(n, c, h, w, kh, kw, o, strides, padding, dilation)
                executable = build(mod)
                dev = tvm.cuda()
                np.random.seed(0)
                A = np.random.randn(n, h, w, c).astype("float16")
                B = np.random.randn(o, kh, kw, c).astype("float16")
                A_tvm = tvm.nd.array(A, dev)
                B_tvm = tvm.nd.array(B, dev)
                result = f_run(executable, dev, A_tvm, B_tvm)
                result_ref = build_and_run_reference(mod, [A, B])
                np.testing.assert_allclose(
                    result.numpy(),
                    result_ref,
                    rtol=5e-2,
                    atol=5e-2,
                )


def constructConv2D_bias(N, C, H, W, KH, KW, O, strides, padding, dilation):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                x = R.arg(
                    "x", relax.TensorStructInfo((N, H, W, C), A_TYPE)
                )  # pylint: disable=invalid-name
                w = R.arg(
                    "w", relax.TensorStructInfo((O, KH, KW, C), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, 1, 1, O), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(
                        R.nn.conv2d(
                            x,
                            w,
                            strides=strides,
                            padding=padding,
                            dilation=dilation,
                            groups=1,
                            data_layout="NHWC",
                            kernel_layout="OHWI",
                            out_layout="NHWC",
                            out_dtype=C_TYPE,
                        )
                    )
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    mod = ib.get()
    return mod


@tvm.testing.requires_cutlass
def test_cutlass_conv2d_bias():
    c, h, w = 3, 224, 224
    kh, kw, o = 3, 3, 64
    for n in [1, 2]:
        for strides in [(1, 1), (2, 2)]:
            for padding in [(0, 0), (3, 3)]:
                for dilation in [(1, 1), (4, 4)]:
                    mod = constructConv2D_bias(n, c, h, w, kh, kw, o, strides, padding, dilation)
                    executable = build(mod)
                    dev = tvm.cuda()
                    np.random.seed(0)
                    A = np.random.randn(n, h, w, c).astype("float16")
                    B = np.random.randn(o, kh, kw, c).astype("float16")
                    bias = np.random.randn(1, 1, 1, o).astype("float16")
                    A_tvm = tvm.nd.array(A, dev)
                    B_tvm = tvm.nd.array(B, dev)
                    bias_tvm = tvm.nd.array(bias, dev)
                    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
                    result_ref = build_and_run_reference(mod, [A, B, bias])
                    np.testing.assert_allclose(
                        result.numpy(),
                        result_ref,
                        rtol=5e-2,
                        atol=5e-2,
                    )


def constructConv2D_bias_add(N, C, H, W, KH, KW, O, OH, OW, strides, padding, dilation):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                x = R.arg(
                    "x", relax.TensorStructInfo((N, H, W, C), A_TYPE)
                )  # pylint: disable=invalid-name
                w = R.arg(
                    "w", relax.TensorStructInfo((O, KH, KW, C), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, 1, 1, O), A_TYPE)
                )  # pylint: disable=invalid-name
                res = R.arg(
                    "res", relax.TensorStructInfo((N, OH, OW, O), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(
                        R.nn.conv2d(
                            x,
                            w,
                            strides=strides,
                            padding=padding,
                            dilation=dilation,
                            groups=1,
                            data_layout="NHWC",
                            kernel_layout="OHWI",
                            out_layout="NHWC",
                            out_dtype=C_TYPE,
                        )
                    )
                    D = R.emit(R.add(C, bias))
                    E = R.emit(R.add(D, res))
                    R.output(E)
                (E,) = df.output_vars
                R.func_ret_value(E)
    mod = ib.get()
    return mod


@tvm.testing.requires_cutlass
def test_cutlass_conv2d_bias_add():
    n, c, h, w = 2, 3, 224, 224
    kh, kw, o = 3, 3, 64
    for strides in [(1, 1), (2, 2)]:
        for padding in [(0, 0), (3, 3)]:
            for dilation in [(1, 1), (4, 4)]:
                oh = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // strides[0] + 1
                ow = (w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // strides[1] + 1
                mod = constructConv2D_bias_add(
                    n, c, h, w, kh, kw, o, oh, ow, strides, padding, dilation
                )
                executable = build(mod)
                dev = tvm.cuda()
                np.random.seed(0)
                A = np.random.randn(n, h, w, c).astype("float16")
                B = np.random.randn(o, kh, kw, c).astype("float16")
                bias = np.random.randn(1, 1, 1, o).astype("float16")
                res = np.random.randn(n, oh, ow, o).astype("float16")
                A_tvm = tvm.nd.array(A, dev)
                B_tvm = tvm.nd.array(B, dev)
                bias_tvm = tvm.nd.array(bias, dev)
                res_tvm = tvm.nd.array(res, dev)
                result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm, res_tvm)
                result_ref = build_and_run_reference(mod, [A, B, bias, res])
                np.testing.assert_allclose(
                    result.numpy(),
                    result_ref,
                    rtol=5e-2,
                    atol=5e-2,
                )


if __name__ == "__main__":
    tvm.testing.main()
