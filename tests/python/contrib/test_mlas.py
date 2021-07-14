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
import tvm
import tvm.testing
from tvm import te, topi, relay
from tvm._ffi.registry import get_global_func
from tvm.relay.testing.temp_op_attr import TempOpAttr


def _verify_topi_mlas_matmul(batch_A, batch_B, m, k, n, dtype="float32"):
    # batch_matmul
    A_shape = [batch_A, m, k]
    B_shape = [batch_B, n, k]
    C_shape = [batch_A, m, n]
    A = te.placeholder(A_shape, name="A", dtype=dtype)
    B = te.placeholder(B_shape, name="B", dtype=dtype)
    C = topi.mlas_matmul(A, B)
    s = te.create_schedule(C.op)
    for target, dev in tvm.testing.enabled_targets():
        f = tvm.build(s, [A, B, C], target)
        a = tvm.nd.array(np.random.uniform(size=A_shape).astype(dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=B_shape).astype(dtype), dev)
        c = tvm.nd.array(np.zeros(C_shape, dtype=dtype), dev)
        c_ref = np.matmul(a.asnumpy(), b.asnumpy().transpose(0, 2, 1))
        f(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), c_ref, rtol=1e-5)
    # dense
    A_shape = [m, k]
    B_shape = [n, k]
    C_shape = [m, n]
    A = te.placeholder(A_shape, name="A", dtype=dtype)
    B = te.placeholder(B_shape, name="B", dtype=dtype)
    C = topi.mlas_matmul(A, B)
    s = te.create_schedule(C.op)
    for target, dev in tvm.testing.enabled_targets():
        f = tvm.build(s, [A, B, C], target)
        a = tvm.nd.array(np.random.uniform(size=A_shape).astype(dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=B_shape).astype(dtype), dev)
        c = tvm.nd.array(np.zeros(C_shape, dtype=dtype), dev)
        c_ref = np.matmul(a.asnumpy(), b.asnumpy().transpose(1, 0))
        f(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), c_ref, rtol=1e-5)


def _verify_topi_mlas_matmul_packed(batch_A, batch_B, m, k, n, dtype="float32"):
    assert batch_B == 1
    get_packb_size = get_global_func("tvm.contrib.mlas.gemm_packb_size")
    packb_size = get_packb_size(n, k)
    arr_size = int(packb_size / 4)
    # batch_matmul
    A_shape = [batch_A, m, k]
    B_shape = [batch_B, n, k]
    C_shape = [batch_A, m, n]
    A = te.placeholder(A_shape, name="A", dtype=dtype)
    B = te.placeholder(B_shape, name="B", dtype=dtype)
    B_packed = topi.mlas_packb(B, k, n, arr_size)
    C = topi.mlas_matmul(A, B_packed, True, k, n)
    s = te.create_schedule(C.op)
    for target, dev in tvm.testing.enabled_targets():
        f = tvm.build(s, [A, B, C], target)
        a = tvm.nd.array(np.random.uniform(size=A_shape).astype(dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=B_shape).astype(dtype), dev)
        c = tvm.nd.array(np.zeros(C_shape, dtype=dtype), dev)
        c_ref = np.matmul(a.asnumpy(), b.asnumpy().transpose(0, 2, 1))
        f(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), c_ref, rtol=1e-5)
    # dense
    A_shape = [m, k]
    B_shape = [n, k]
    C_shape = [m, n]
    A = te.placeholder(A_shape, name="A", dtype=dtype)
    B = te.placeholder(B_shape, name="B", dtype=dtype)
    B_packed = topi.mlas_packb(B, k, n, arr_size)
    C = topi.mlas_matmul(A, B_packed, True, k, n)
    s = te.create_schedule(C.op)
    for target, dev in tvm.testing.enabled_targets():
        f = tvm.build(s, [A, B, C], target)
        a = tvm.nd.array(np.random.uniform(size=A_shape).astype(dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=B_shape).astype(dtype), dev)
        c = tvm.nd.array(np.zeros(C_shape, dtype=dtype), dev)
        c_ref = np.matmul(a.asnumpy(), b.asnumpy().transpose(1, 0))
        f(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), c_ref, rtol=1e-5)


def _verify_relay_mlas_matmul(batch_A, batch_B, m, k, n, dtype="float32"):
    # batch_matmul
    A_shape = [batch_A, m, k]
    B_shape = [batch_B, n, k]
    A = relay.var("A", shape=A_shape, dtype=dtype)
    B = relay.var("B", shape=B_shape, dtype=dtype)
    C = relay.op.mlas_matmul(A, B)
    func = relay.Function([A, B], C)

    for target, dev in tvm.testing.enabled_targets():
        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, device=dev, target=target)
            a = tvm.nd.array(np.random.uniform(size=A_shape).astype(dtype), dev)
            b = tvm.nd.array(np.random.uniform(size=B_shape).astype(dtype), dev)
            c_ref = np.matmul(a.asnumpy(), b.asnumpy().transpose(0, 2, 1))
            c = intrp.evaluate(func)(a, b)
            tvm.testing.assert_allclose(c.asnumpy(), c_ref, rtol=1e-5)

    # dense
    A_shape = [m, k]
    B_shape = [n, k]
    A = relay.var("A", shape=A_shape, dtype=dtype)
    B = relay.var("B", shape=B_shape, dtype=dtype)
    C = relay.op.mlas_matmul(A, B)
    func = relay.Function([A, B], C)

    for target, dev in tvm.testing.enabled_targets():
        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, device=dev, target=target)
            a = tvm.nd.array(np.random.uniform(size=A_shape).astype(dtype), dev)
            b = tvm.nd.array(np.random.uniform(size=B_shape).astype(dtype), dev)
            c_ref = np.matmul(a.asnumpy(), b.asnumpy().transpose(1, 0))
            c = intrp.evaluate(func)(a, b)
            tvm.testing.assert_allclose(c.asnumpy(), c_ref, rtol=1e-5)


def _verify_relay_mlas_matmul_packed(batch_A, batch_B, m, k, n, dtype="float32"):
    # batch_matmul
    A_shape = [batch_A, m, k]
    B_shape = [batch_B, n, k]
    A = relay.var("A", shape=A_shape, dtype=dtype)
    B = relay.var("B", shape=B_shape, dtype=dtype)
    B_packed = relay.op.mlas_packb(B, k, n)
    C = relay.op.mlas_matmul(A, B_packed, True, k, n)
    func = relay.Function([A, B], C)

    for target, dev in tvm.testing.enabled_targets():
        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, device=dev, target=target)
            a = tvm.nd.array(np.random.uniform(size=A_shape).astype(dtype), dev)
            b = tvm.nd.array(np.random.uniform(size=B_shape).astype(dtype), dev)
            c_ref = np.matmul(a.asnumpy(), b.asnumpy().transpose(0, 2, 1))
            c = intrp.evaluate(func)(a, b)
            tvm.testing.assert_allclose(c.asnumpy(), c_ref, rtol=1e-5)

    # dense
    A_shape = [m, k]
    B_shape = [n, k]
    A = relay.var("A", shape=A_shape, dtype=dtype)
    B = relay.var("B", shape=B_shape, dtype=dtype)
    B_packed = relay.op.mlas_packb(B, k, n)
    C = relay.op.mlas_matmul(A, B_packed, True, k, n)
    func = relay.Function([A, B], C)

    for target, dev in tvm.testing.enabled_targets():
        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, device=dev, target=target)
            a = tvm.nd.array(np.random.uniform(size=A_shape).astype(dtype), dev)
            b = tvm.nd.array(np.random.uniform(size=B_shape).astype(dtype), dev)
            c_ref = np.matmul(a.asnumpy(), b.asnumpy().transpose(1, 0))
            c = intrp.evaluate(func)(a, b)
            tvm.testing.assert_allclose(c.asnumpy(), c_ref, rtol=1e-5)


def test_topi_mlas_matmul():
    if not get_global_func("tvm.contrib.mlas.batch_sgemm", allow_missing=True):
        print("skip because mlas is not enabled...")
        return
    for _ in range(10):
        m, k, n = np.random.randint(1, 100), np.random.randint(1, 100), np.random.randint(1, 100)
        _verify_topi_mlas_matmul(1, 1, m, k, n)
        _verify_topi_mlas_matmul(10, 1, m, k, n)
        _verify_topi_mlas_matmul(10, 10, m, k, n)


def test_topi_mlas_matmul_packed():
    if not get_global_func("tvm.contrib.mlas.batch_sgemm", allow_missing=True):
        print("skip because mlas is not enabled...")
        return
    for _ in range(10):
        m, k, n = np.random.randint(1, 100), np.random.randint(1, 100), np.random.randint(1, 100)
        _verify_topi_mlas_matmul_packed(1, 1, m, k, n)
        _verify_topi_mlas_matmul_packed(10, 1, m, k, n)


def test_relay_mlas_matmul():
    if not get_global_func("tvm.contrib.mlas.batch_sgemm", allow_missing=True):
        print("skip because mlas is not enabled...")
        return
    m, k, n = np.random.randint(1, 100), np.random.randint(1, 100), np.random.randint(1, 100)
    _verify_relay_mlas_matmul(1, 1, m, k, n)
    _verify_relay_mlas_matmul(10, 1, m, k, n)
    _verify_relay_mlas_matmul(10, 10, m, k, n)


def test_relay_mlas_matmul_packed():
    if not get_global_func("tvm.contrib.mlas.batch_sgemm", allow_missing=True):
        print("skip because mlas is not enabled...")
        return
    m, k, n = np.random.randint(1, 100), np.random.randint(1, 100), np.random.randint(1, 100)
    _verify_relay_mlas_matmul_packed(1, 1, m, k, n)
    _verify_relay_mlas_matmul_packed(10, 1, m, k, n)


def _run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_alter_op_layout_dnese():
    if not get_global_func("tvm.contrib.mlas.batch_sgemm", allow_missing=True):
        print("skip because mlas is not enabled...")
        return
    target = "llvm -libs=mlas"
    m, k, n = 32, 48, 64
    B_const = np.random.uniform(size=[n, k]).astype("float32")

    def pack_before():
        A = relay.var("A", shape=(m, k), dtype="float32")
        B = relay.const(B_const, "float32")
        C = relay.nn.dense(A, B)
        f = relay.Function(relay.analysis.free_vars(C), C)
        return f

    def pack_expected():
        A = relay.var("A", shape=(m, k), dtype="float32")
        B = relay.const(B_const, "float32")
        B_packed = relay.op.mlas_packb(B, k, n)
        C = relay.op.mlas_matmul(A, B_packed, True, k, n)
        f = relay.Function(relay.analysis.free_vars(C), C)
        return f

    with tvm.target.Target(target):
        with TempOpAttr(
            "nn.dense", "FTVMAlterOpLayout", topi.x86.dense_alter_op._alter_dense_layout
        ):
            a = pack_before()
            a = _run_opt_pass(a, relay.transform.AlterOpLayout())
            b = _run_opt_pass(pack_expected(), relay.transform.InferType())
            assert tvm.ir.structural_equal(a, b)

    def nopack_before():
        A = relay.var("A", shape=(m, k), dtype="float32")
        B = relay.var("B", shape=(n, k), dtype="float32")
        C = relay.nn.dense(A, B)
        f = relay.Function(relay.analysis.free_vars(C), C)
        return f

    def nopack_expected():
        A = relay.var("A", shape=(m, k), dtype="float32")
        B = relay.var("B", shape=(n, k), dtype="float32")
        C = relay.op.mlas_matmul(A, B, False)
        f = relay.Function(relay.analysis.free_vars(C), C)
        return f

    with tvm.target.Target(target):
        with TempOpAttr(
            "nn.dense", "FTVMAlterOpLayout", topi.x86.dense_alter_op._alter_dense_layout
        ):
            a = nopack_before()
            a = _run_opt_pass(a, relay.transform.AlterOpLayout())
            b = _run_opt_pass(nopack_expected(), relay.transform.InferType())
            assert tvm.ir.structural_equal(a, b)

    def dynamic_before():
        A = relay.var("A", shape=(relay.Any(), k), dtype="float32")
        B = relay.var("B", shape=(n, k), dtype="float32")
        C = relay.nn.dense(A, B)
        f = relay.Function(relay.analysis.free_vars(C), C)
        return f

    def dynamic_expected():
        A = relay.var("A", shape=(relay.Any(), k), dtype="float32")
        B = relay.var("B", shape=(n, k), dtype="float32")
        target_layout = "NK16n"
        weight_transform = relay.layout_transform(B, "NK", target_layout)
        y = relay.nn.contrib_dense_pack(A, weight_transform, units=None, out_dtype="float32")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    with tvm.target.Target(target):
        with TempOpAttr(
            "nn.dense", "FTVMAlterOpLayout", topi.x86.dense_alter_op._alter_dense_layout
        ):
            a = dynamic_before()
            a = _run_opt_pass(a, relay.transform.AlterOpLayout())
            b = _run_opt_pass(dynamic_expected(), relay.transform.InferType())
            assert tvm.ir.structural_equal(a, b)


def test_alter_op_layout_batch_matmul():
    if not get_global_func("tvm.contrib.mlas.batch_sgemm", allow_missing=True):
        print("skip because mlas is not enabled...")
        return
    target = "llvm -libs=mlas"
    m, k, n = 32, 48, 64
    B_const = np.random.uniform(size=[1, n, k]).astype("float32")

    def pack_before():
        A = relay.var("A", shape=(1, m, k), dtype="float32")
        B = relay.const(B_const, "float32")
        C = relay.nn.batch_matmul(A, B)
        f = relay.Function(relay.analysis.free_vars(C), C)
        return f

    def pack_expected():
        A = relay.var("A", shape=(1, m, k), dtype="float32")
        B = relay.const(B_const, "float32")
        B_packed = relay.op.mlas_packb(B, k, n)
        C = relay.op.mlas_matmul(A, B_packed, True, k, n)
        f = relay.Function(relay.analysis.free_vars(C), C)
        return f

    with tvm.target.Target(target):
        with TempOpAttr(
            "nn.batch_matmul", "FTVMAlterOpLayout", relay.op._mlas._alter_batch_matmul_layout
        ):
            a = pack_before()
            a = _run_opt_pass(a, relay.transform.AlterOpLayout())
            b = _run_opt_pass(pack_expected(), relay.transform.InferType())
            assert tvm.ir.structural_equal(a, b)

    def nopack_before():
        A = relay.var("A", shape=(1, m, k), dtype="float32")
        B = relay.var("B", shape=(1, n, k), dtype="float32")
        C = relay.nn.batch_matmul(A, B)
        f = relay.Function(relay.analysis.free_vars(C), C)
        return f

    def nopack_expected():
        A = relay.var("A", shape=(1, m, k), dtype="float32")
        B = relay.var("B", shape=(1, n, k), dtype="float32")
        C = relay.op.mlas_matmul(A, B, False)
        f = relay.Function(relay.analysis.free_vars(C), C)
        return f

    with tvm.target.Target(target):
        with TempOpAttr(
            "nn.batch_matmul", "FTVMAlterOpLayout", relay.op._mlas._alter_batch_matmul_layout
        ):
            a = nopack_before()
            a = _run_opt_pass(a, relay.transform.AlterOpLayout())
            b = _run_opt_pass(nopack_expected(), relay.transform.InferType())
            assert tvm.ir.structural_equal(a, b)

    def dynamic_expected():
        A = relay.var("A", shape=(1, relay.Any(), k), dtype="float32")
        B = relay.var("B", shape=(1, n, k), dtype="float32")
        C = relay.nn.batch_matmul(A, B)
        f = relay.Function(relay.analysis.free_vars(C), C)
        return f

    with tvm.target.Target(target):
        with TempOpAttr(
            "nn.batch_matmul", "FTVMAlterOpLayout", relay.op._mlas._alter_batch_matmul_layout
        ):
            a = dynamic_expected()
            a = _run_opt_pass(a, relay.transform.AlterOpLayout())
            b = _run_opt_pass(dynamic_expected(), relay.transform.InferType())
            assert tvm.ir.structural_equal(a, b)


if __name__ == "__main__":
    test_topi_mlas_matmul()
    test_topi_mlas_matmul_packed()
    test_relay_mlas_matmul()
    test_relay_mlas_matmul_packed()
    test_alter_op_layout_dnese()
    test_alter_op_layout_batch_matmul()
