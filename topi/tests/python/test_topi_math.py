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
import tvm
import topi
import topi.testing
from topi import util
from common import get_all_backend


def test_util():
    x = tvm.const(100, "int32")
    assert util.get_const_int(x) == 100
    assert util.get_const_tuple((x, x)) == (100, 100)


def test_ewise():
    def test_apply(
        func,
        name,
        f_numpy,
        low,
        high,
        shape=(20, 3),
        dtype=tvm.float32,
        check_round=False,
        skip_name_check=False,
    ):
        m = tvm.var("m")
        l = tvm.var("l")
        A = tvm.placeholder((m, l), dtype=dtype, name="A")

        B = func(A)
        assert tuple(B.shape) == tuple(A.shape)
        if not skip_name_check:
            assert B.op.body[0].name == name
        a_np = np.random.uniform(low=low, high=high, size=shape).astype(A.dtype) * 10
        # avoid round check too close to boundary
        if check_round:
            a_np += ((np.fmod(a_np, 1) - 0.5) < 1e-6) * 1e-5
        b_np = f_numpy(a_np)

        def check_device(device):
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                return
            print("Running on target: %s" % device)
            with tvm.target.create(device):
                s = topi.generic.schedule_injective(B)
            foo = tvm.build(s, [A, B], device, name=name)
            a = tvm.nd.array(a_np, ctx)
            b = tvm.nd.array(np.zeros_like(b_np), ctx)
            foo(a, b)
            tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5, atol=1e-5)

        check_device('llvm')
        check_device('cuda')
        check_device('opencl')
        check_device('metal')
        check_device('rocm')
        check_device('vulkan')
        check_device('nvptx')
        check_device('llvm -device=arm-cpu')
        check_device('opencl -device=mali')
        check_device('aocl_sw_emu')

    def test_isnan(
        low,
        high,
        shape=(20, 3),
        dtype=tvm.float32,
        check_round=False,
        skip_name_check=False,
    ):
        m = tvm.var("m")
        l = tvm.var("l")
        A = tvm.placeholder((m, l), dtype=dtype, name="A")

        B = topi.isnan(A)
        assert tuple(B.shape) == tuple(A.shape)
        if not skip_name_check:
            assert B.op.body[0].name == "isnan"
        a_np = np.random.uniform(low=low, high=high, size=shape).astype(A.dtype) * 10
        a_np.ravel()[np.random.choice(a_np.size, int(a_np.size * 0.5), replace=False)] = np.nan
        # avoid round check too close to boundary
        if check_round:
            a_np += ((np.fmod(a_np, 1) - 0.5) < 1e-6) * 1e-5
        b_np = np.isnan(a_np)

        def check_device(device):
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                return
            print("Running on target: %s" % device)
            with tvm.target.create(device):
                s = topi.generic.schedule_injective(B)
            foo = tvm.build(s, [A, B], device, name="isnan")
            a = tvm.nd.array(a_np, ctx)
            b = tvm.nd.array(np.zeros_like(b_np), ctx)
            foo(a, b)
            tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5, atol=1e-5)

        check_device('llvm')
        check_device('cuda')
        check_device('opencl')
        check_device('metal')
        check_device('rocm')
        check_device('vulkan')
        check_device('nvptx')
        check_device('llvm -device=arm-cpu')
        check_device('opencl -device=mali')
        check_device('aocl_sw_emu')

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
    test_apply(topi.rsqrt, "rsqrt", lambda x: np.ones_like(x) / np.sqrt(x), 0, 100, skip_name_check=True)
    test_apply(topi.cos, "cos", np.cos, -2.0*np.pi, 2.0*np.pi)
    test_apply(topi.sin, "sin", np.sin, -2.0*np.pi, 2.0*np.pi)
    test_apply(topi.erf, "erf", scipy.special.erf, -.1, .1, dtype="float32")
    test_isnan(-100, 100)


def test_cast():
    def verify(from_dtype, to_dtype, low=-100, high=100):
        shape = (5, 4)
        A = tvm.placeholder(shape, dtype=from_dtype, name="A")
        B = topi.cast(A, to_dtype)

        if from_dtype == "bool":
            a_np = np.random.choice([True, False], size=shape)
        else:
            a_np = np.random.uniform(low, high, size=shape).astype(from_dtype)
        if to_dtype == "bool":
            a_np = a_np - a_np[2, 3]
        b_np = a_np.astype(to_dtype)

        for device in get_all_backend():
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                continue
            print("Running on target: %s" % device)
            with tvm.target.create(device):
                s = topi.generic.schedule_injective(B)
            foo = tvm.build(s, [A, B], device)
            a = tvm.nd.array(a_np, ctx)
            b = tvm.nd.empty(shape=shape, dtype=to_dtype, ctx=ctx)
            foo(a, b)
            tvm.testing.assert_allclose(b.asnumpy(), b_np)

    verify("int32", "float32")
    verify("int32", "float64")
    verify("int32", "bool")
    verify("float32", "int32")
    verify("float32", "float64")
    verify("float32", "bool")
    verify("bool", "float32")
    verify("bool", "int32")


if __name__ == "__main__":
    test_util()
    test_ewise()
    test_cast()
