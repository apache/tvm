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
import scipy
from scipy import special
import tvm
from tvm import te
from tvm import topi
import tvm.testing
import tvm.topi.testing
from tvm.topi import utils


def test_util():
    x = tvm.tir.const(100, "int32")
    assert utils.get_const_int(x) == 100
    assert utils.get_const_tuple((x, x)) == (100, 100)


@tvm.testing.uses_gpu
def test_ewise():
    def test_apply(
        func,
        name,
        f_numpy,
        low,
        high,
        shape=(20, 3),
        dtype="float32",
        check_round=False,
        skip_name_check=False,
    ):
        m = te.var("m")
        l = te.var("l")
        A = te.placeholder((m, l), dtype=dtype, name="A")

        B = func(A)
        assert tuple(B.shape) == tuple(A.shape)
        if not skip_name_check:
            assert B.op.body[0].op.name == "tir." + name
        a_np = np.random.uniform(low=low, high=high, size=shape).astype(A.dtype) * 10
        # avoid round check too close to boundary
        if check_round:
            a_np += ((np.abs(np.fmod(a_np, 1)) - 0.5) < 1e-6) * 1e-4
        b_np = f_numpy(a_np)

        def check_target(target, dev):
            print("Running on target: %s" % target)
            with tvm.target.Target(target):
                s = tvm.topi.testing.get_injective_schedule(target)(B)
            foo = tvm.build(s, [A, B], target, name=name)
            a = tvm.nd.array(a_np, dev)
            b = tvm.nd.array(np.zeros_like(b_np), dev)
            foo(a, b)
            tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5, atol=1e-5)

        for target, dev in tvm.testing.enabled_targets():
            check_target(target, dev)

    def test_isnan(
        low, high, shape=(20, 3), dtype="float32", check_round=False, skip_name_check=False
    ):
        m = te.var("m")
        l = te.var("l")
        A = te.placeholder((m, l), dtype=dtype, name="A")

        B = topi.isnan(A)
        assert tuple(B.shape) == tuple(A.shape)
        if not skip_name_check:
            assert B.op.body[0].op.name == "tir.isnan"
        a_np = np.random.uniform(low=low, high=high, size=shape).astype(A.dtype) * 10
        a_np.ravel()[np.random.choice(a_np.size, int(a_np.size * 0.5), replace=False)] = np.nan
        # avoid round check too close to boundary
        if check_round:
            a_np += ((np.abs(np.fmod(a_np, 1)) - 0.5) < 1e-6) * 1e-5
        b_np = np.isnan(a_np)

        def check_target(target, dev):
            print("Running on target: %s" % target)
            with tvm.target.Target(target):
                s = tvm.topi.testing.get_injective_schedule(target)(B)
            foo = tvm.build(s, [A, B], target, name="isnan")
            a = tvm.nd.array(a_np, dev)
            b = tvm.nd.array(np.zeros_like(b_np), dev)
            foo(a, b)
            tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5, atol=1e-5)

        for target, dev in tvm.testing.enabled_targets():
            check_target(target, dev)

    def test_infiniteness_ops(topi_op, ref_op, name):
        for dtype in ["float32", "float64", "int32", "int16"]:
            m = te.var("m")
            l = te.var("l")
            A = te.placeholder((m, l), dtype=dtype, name="A")
            B = topi_op(A)
            assert tuple(B.shape) == tuple(A.shape)

            a_np = np.random.uniform(size=(8, 8)).astype(A.dtype) * 10
            if dtype.startswith("float"):
                a_np.ravel()[
                    np.random.choice(a_np.size, int(a_np.size * 0.5), replace=False)
                ] = np.infty
                a_np.ravel()[
                    np.random.choice(a_np.size, int(a_np.size * 0.5), replace=False)
                ] = np.nan
            b_np = ref_op(a_np)

            def check_target(target, dev):
                with tvm.target.Target(target):
                    s = tvm.topi.testing.get_injective_schedule(target)(B)
                foo = tvm.build(s, [A, B], target, name=name)
                a = tvm.nd.array(a_np, dev)
                b = tvm.nd.array(np.zeros_like(b_np), dev)
                foo(a, b)
                tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5, atol=1e-5)

            for target, dev in tvm.testing.enabled_targets():
                check_target(target, dev)

    test_apply(topi.floor, "floor", np.floor, -100, 100)
    test_apply(topi.ceil, "ceil", np.ceil, -100, 100)
    test_apply(topi.sign, "sign", np.sign, -100, 100, skip_name_check=True)
    test_apply(topi.trunc, "trunc", np.trunc, -100, 100)
    test_apply(topi.abs, "fabs", np.abs, -100, 100)
    test_apply(topi.round, "round", np.round, -100, 100, check_round=True)
    test_apply(topi.exp, "exp", np.exp, -1, 1)
    test_apply(topi.tanh, "tanh", np.tanh, -10, 10, shape=(128, 128))
    test_apply(topi.tanh, "tanh", np.tanh, -10, 10, shape=(128, 128), dtype="float64")
    test_apply(topi.sigmoid, "sigmoid", lambda x: 1 / (1 + np.exp(-x)), -1, 1)
    test_apply(topi.log, "log", np.log, 0, 100)
    test_apply(topi.sqrt, "sqrt", np.sqrt, 0, 100)
    test_apply(
        topi.rsqrt, "rsqrt", lambda x: np.ones_like(x) / np.sqrt(x), 0, 100, skip_name_check=True
    )
    test_apply(topi.cos, "cos", np.cos, -2.0 * np.pi, 2.0 * np.pi)
    test_apply(topi.tan, "tan", np.tan, -2.0 * np.pi, 2.0 * np.pi, dtype="float32")
    test_apply(topi.tan, "tan", np.tan, -2.0 * np.pi, 2.0 * np.pi, dtype="float64")
    test_apply(topi.sin, "sin", np.sin, -2.0 * np.pi, 2.0 * np.pi)
    test_apply(topi.erf, "erf", scipy.special.erf, -0.1, 0.1, dtype="float32")
    test_isnan(-100, 100)
    test_infiniteness_ops(topi.isfinite, np.isfinite, "isifinite")
    test_infiniteness_ops(topi.isinf, np.isinf, "isinf")


@tvm.testing.uses_gpu
def test_cast():
    def verify(from_dtype, to_dtype, low=-100, high=100):
        shape = (5, 4)
        A = te.placeholder(shape, dtype=from_dtype, name="A")
        B = topi.cast(A, to_dtype)

        if from_dtype == "bool":
            a_np = np.random.choice([True, False], size=shape)
        else:
            a_np = np.random.uniform(low, high, size=shape).astype(from_dtype)
        if to_dtype == "bool":
            a_np = a_np - a_np[2, 3]
        b_np = a_np.astype(to_dtype)

        for target, dev in tvm.testing.enabled_targets():
            print("Running on target: %s" % target)
            with tvm.target.Target(target):
                s = tvm.topi.testing.get_injective_schedule(target)(B)
            foo = tvm.build(s, [A, B], target)
            a = tvm.nd.array(a_np, dev)
            b = tvm.nd.empty(shape=shape, dtype=to_dtype, device=dev)
            foo(a, b)
            tvm.testing.assert_allclose(b.numpy(), b_np)

    verify("int32", "float32")
    verify("int32", "float64")
    verify("int32", "bool")
    verify("float32", "int32")
    verify("float32", "float64")
    verify("float32", "bool")
    verify("bool", "float32")
    verify("bool", "int32")


def test_fastmath():
    def test_apply(func, name, f_numpy, low, high, step, dtype="float32"):
        a_np = np.arange(low, high, step).astype(dtype).reshape((1, -1))
        b_np = f_numpy(a_np)
        A = te.placeholder(a_np.shape, dtype=dtype, name="A")
        B = func(A)
        assert tuple(B.shape) == tuple(A.shape)

        def check_target(target):
            dev = tvm.device(target, 0)
            if not tvm.testing.device_enabled(target):
                print("Skip because %s is not enabled" % target)
                return
            with tvm.target.Target(target):
                s = topi.generic.schedule_injective(B)
            func = tvm.build(s, [A, B], target, name=name)
            a = tvm.nd.array(a_np, dev)
            b = tvm.nd.array(np.zeros_like(b_np), dev)
            func(a, b)
            tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5, atol=1e-5)

        check_target("llvm")
        check_target("llvm -device=arm-cpu")

    test_apply(topi.fast_exp, "fast_exp", np.exp, low=-88, high=88, step=0.01)
    test_apply(topi.fast_erf, "fast_erf", scipy.special.erf, low=-10, high=10, step=0.01)
    test_apply(topi.fast_tanh, "fast_tanh", np.tanh, low=-10, high=10, step=0.01)
    test_apply(
        topi.nn.fast_softmax,
        "fast_softmax",
        tvm.topi.testing.softmax_python,
        low=-10,
        high=10,
        step=0.01,
    )


if __name__ == "__main__":
    test_util()
    test_ewise()
    test_cast()
    test_fastmath()
