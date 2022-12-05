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
from tvm import te
import scipy
from tvm import relay
import pytest
from tvm.relay.testing import run_infer_type
import tvm.topi.testing
from tvm.contrib.nvcc import have_fp16
import tvm.testing

executor_kind = tvm.testing.parameter("graph", "vm")


def sigmoid(x):
    one = np.ones_like(x)
    return one / (one + np.exp(-x))


def relu(x):
    x_copy = np.copy(x)
    np.maximum(x_copy, 0, x_copy)
    return x_copy


def rsqrt(x):
    one = np.ones_like(x)
    return one / np.sqrt(x)


class TestUnaryOp:
    # Tuple of (operator, reference op, supports fp16)
    op_list = {
        "log": (tvm.relay.log, np.log, True),
        "exp": (tvm.relay.exp, np.exp, True),
        "erf": (tvm.relay.erf, scipy.special.erf, True),
        "sqrt": (tvm.relay.sqrt, np.sqrt, True),
        "rqsrt": (tvm.relay.rsqrt, rsqrt, True),
        "sigmoid": (tvm.relay.sigmoid, sigmoid, True),
        "tanh": (tvm.relay.tanh, np.tanh, False),
        "relu": (relay.nn.relu, relu, True),
        "cos": (tvm.relay.cos, np.cos, True),
        "sin": (tvm.relay.sin, np.sin, True),
        "tan": (tvm.relay.tan, np.tan, False),
        "atan": (tvm.relay.atan, np.arctan, False),
        "ceil": (tvm.relay.ceil, np.ceil, True),
        "floor": (tvm.relay.floor, np.floor, True),
        "trunc": (tvm.relay.trunc, np.trunc, True),
        "round": (tvm.relay.round, np.round, False),
    }

    dtype = tvm.testing.parameter("float16", "float32")

    relay_op, ref_func, supports_fp16 = tvm.testing.parameters(
        *op_list.values(), ids=op_list.keys()
    )

    def test_unary_op(self, target, dev, relay_op, ref_func, supports_fp16, dtype):
        target = tvm.target.Target(target)
        if dtype == "float16":
            if target.kind.name == "cuda":
                if not have_fp16(tvm.cuda(0).compute_version):
                    pytest.xfail(
                        "No float16 support on local cuda device (compute_version != 5.3 and < 6.0)"
                    )
            elif target.kind.name == "vulkan" and not target.attrs.get("supports_float16", False):
                pytest.xfail("No float16 support on vulkan target (supports_float16=False)")
            elif not supports_fp16:
                pytest.xfail(f"No float16 support on {target.kind.name} target")

        if target.kind.name == "vulkan" and relay_op in [
            tvm.relay.erf,
            tvm.relay.tan,
            tvm.relay.atan,
        ]:
            pytest.xfail(f"Vulkan runtime doesn't yet support {relay_op}")

        shape = (10, 4)
        dtype = dtype
        tp = relay.TensorType(shape, dtype=dtype)
        x = relay.var("x", type_annotation=tp)
        y = relay_op(x)
        # test printer
        assert ("{}(%x)".format(y.op.name)) in y.astext()
        # test type inference
        yy = run_infer_type(y)
        assert yy.checked_type == tp

        if ref_func is not None:
            data = np.random.rand(*shape).astype(dtype)
            ref_res = ref_func(data).astype(dtype)
            func = relay.Function([x], y)
            # use graph by execuor default for testing, as we need
            # create function explicitly to avoid constant-folding.
            op_res = relay.create_executor("graph", device=dev, target=target).evaluate(func)(data)
            tolerance = 1e-2 if dtype == "float16" else 1e-5
            np.testing.assert_allclose(op_res.numpy(), ref_res, rtol=tolerance)


@tvm.testing.uses_gpu
def test_binary_op():
    def inst(vars, sh):
        return [vars.get(s, s) for s in sh]

    def check_binary_op(opfunc, ref, dtype):
        # TODO(@jroesch): this piece of code improperly uses type variables.
        n = te.var("n")
        s1 = (5, n, 5)
        s2 = (n, 1)
        t1 = relay.TensorType(s1)
        t2 = relay.TensorType(s2)
        x = relay.var("x", t1, dtype=dtype)
        y = relay.var("y", t2, dtype=dtype)
        z = opfunc(x, y)
        # test printer
        assert ("{}(%x, %y)".format(z.op.name)) in z.astext()
        zz = run_infer_type(z)
        assert zz.checked_type == t1

        if ref is not None:
            t1 = relay.TensorType((5, 10, 5))
            t2 = relay.TensorType((5, 10, 5))
            x = relay.var("x", t1, dtype=dtype)
            y = relay.var("y", t2, dtype=dtype)
            z = opfunc(x, y)
            x_data = np.random.rand(5, 10, 5).astype(dtype)
            y_data = np.random.rand(5, 10, 5).astype(dtype)
            ref_res = ref(x_data, y_data)
            func = relay.Function([x, y], z)

            for target, dev in tvm.testing.enabled_targets():
                # use graph by execuor default for testing, as we need
                # create function explicitly to avoid constant-folding.
                if (
                    dtype == "float16"
                    and target == "cuda"
                    and not have_fp16(tvm.cuda(0).compute_version)
                ):
                    continue
                op_res = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                    x_data, y_data
                )
                np.testing.assert_allclose(op_res.numpy(), ref_res, rtol=0.01, atol=1e-3)

    for opfunc, ref in [
        (relay.add, np.add),
        (relay.subtract, np.subtract),
        (relay.multiply, np.multiply),
        (relay.divide, np.divide),
        (relay.floor_divide, np.floor_divide),
        (relay.floor_mod, np.fmod),
    ]:
        for dtype in ["float16", "float32"]:
            check_binary_op(opfunc, ref, dtype)


@tvm.testing.uses_gpu
def test_expand_dims():
    # based on topi test
    def verify_expand_dims(dshape, dtype, oshape, axis, num_newaxis):
        x = relay.Var("x", relay.TensorType(dshape, dtype))
        func = relay.Function([x], relay.expand_dims(x, axis, num_newaxis))
        for target, dev in tvm.testing.enabled_targets():
            if (
                dtype == "float16"
                and target == "cuda"
                and not have_fp16(tvm.cuda(0).compute_version)
            ):
                continue
            data = np.random.uniform(size=dshape).astype(dtype)
            ref_res = data.reshape(oshape)
            op_res = relay.create_executor("graph", device=dev, target=target).evaluate(func)(data)
            np.testing.assert_allclose(op_res.numpy(), ref_res, rtol=0.01)

    for dtype in ["float16", "float32"]:
        verify_expand_dims((3, 10), dtype, (3, 10, 1, 1), 2, 2)
        verify_expand_dims((3, 10), dtype, (1, 3, 10), -3, 1)


@tvm.testing.uses_gpu
def test_bias_add():
    for dtype in ["float16", "float32"]:
        xshape = (10, 2, 3, 4)
        bshape = (2,)
        rtol = 1e-2 if dtype == "float16" else 1e-5
        x = relay.var("x", shape=xshape, dtype=dtype)
        bias = relay.var("bias", dtype=dtype)
        z = relay.nn.bias_add(x, bias)
        zz = run_infer_type(z)
        assert "axis=" not in zz.astext()
        assert zz.args[1].checked_type == relay.TensorType(bshape, dtype)

        func = relay.Function([x, bias], z)
        x_data = np.random.uniform(size=xshape).astype(dtype)
        y_data = np.random.uniform(size=bshape).astype(dtype)
        ref_res = x_data + y_data.reshape((2, 1, 1))
        for target, dev in tvm.testing.enabled_targets():
            if (
                dtype == "float16"
                and target == "cuda"
                and not have_fp16(tvm.cuda(0).compute_version)
            ):
                continue
            op_res = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                x_data, y_data
            )
            np.testing.assert_allclose(op_res.numpy(), ref_res, rtol=rtol)


def test_bias_add_type_failure():
    def assert_failure(expr):
        try:
            run_infer_type(expr)
        except tvm._ffi.base.TVMError:
            return
        else:
            assert False

    for axis in (0, -1, -3, 1):
        assert_failure(relay.nn.bias_add(relay.const(1), relay.const(2), axis=axis))


def test_expand_dims_infer_type():
    for dtype in ["float16", "float32"]:
        n, t, d = te.size_var("n"), te.size_var("t"), 100
        x = relay.var("x", shape=(n, t, d), dtype=dtype)
        y = relay.expand_dims(x, axis=2)
        assert "axis=2" in y.astext()
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((n, t, 1, 100), dtype)


@tvm.testing.uses_gpu
def test_softmax():
    for shape in [(10, 4), (10, 5, 4)]:
        for dtype in ["float16", "float32"]:
            # Softmax accuracy for float16 is poor
            if dtype == "float16":
                continue
            x = relay.var("x", shape=shape, dtype=dtype)
            y = relay.nn.softmax(x, axis=1)
            assert "nn.softmax" in y.astext()
            yy = run_infer_type(y)
            assert yy.checked_type == relay.TensorType(shape, dtype)
            func = relay.Function([x], y)
            x_data = np.random.uniform(size=shape).astype(dtype)
            ref_res = tvm.topi.testing.softmax_python(x_data, axis=1)
            for target, dev in tvm.testing.enabled_targets():
                op_res = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                    x_data
                )
                np.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
def test_log_softmax():
    for shape in [(10, 4), (10, 5, 4)]:
        for dtype in ["float16", "float32"]:
            # Softmax accuracy for float16 is poor
            if dtype == "float16":
                continue
            x = relay.var("x", shape=shape, dtype=dtype)
            y = relay.nn.log_softmax(x, axis=1)
            assert "nn.log_softmax" in y.astext()
            yy = run_infer_type(y)
            assert yy.checked_type == relay.TensorType(shape, dtype)
            func = relay.Function([x], y)
            x_data = np.random.uniform(size=shape).astype(dtype)
            ref_res = tvm.topi.testing.log_softmax_python(x_data, axis=1)
            for target, dev in tvm.testing.enabled_targets():
                if target == "nvptx":
                    continue
                op_res = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                    x_data
                )
                np.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
def test_concatenate(executor_kind):
    for dtype in ["float16", "float32"]:
        n, t, d = te.size_var("n"), te.size_var("t"), 100
        x = relay.var("x", shape=(n, t, d))
        y = relay.var("y", shape=(n, t, d))
        z = relay.concatenate((x, y), axis=-1)
        assert "axis=" in z.astext()
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType((n, t, 200))

        x = relay.exp(x)
        z = relay.concatenate((x, y), axis=2)
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType((n, t, 200))

        z = relay.concatenate((x, y), axis=1)
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType((n, t + t, 100))

        # check shape mismatches (the following case is expected to raise tvm._ffi.base.TVMError.
        try:
            x = relay.var("p1", shape=(2, 5))
            y = relay.var("p2", shape=(2, 3))
            c = relay.concatenate([x, y], axis=0)
            func = relay.Function([x, y], c)
            zz = run_infer_type(func)
        except tvm._ffi.base.TVMError:
            pass
        else:
            assert False

        x = relay.var("x", shape=(10, 5), dtype=dtype)
        y = relay.var("y", shape=(10, 5), dtype=dtype)
        t = relay.var("z", shape=(), dtype=dtype)
        z = relay.concatenate((x, y), axis=1)
        z = relay.add(z, t)
        # Check result.
        func = relay.Function([x, y, t], z)
        x_data = np.random.rand(10, 5).astype(dtype)
        y_data = np.random.rand(10, 5).astype(dtype)
        t_data = np.random.uniform(size=()).astype(dtype)
        ref_res = np.concatenate((x_data, y_data), axis=1) + t_data

        for target, dev in tvm.testing.enabled_targets():
            if (
                dtype == "float16"
                and target == "cuda"
                and not have_fp16(tvm.cuda(0).compute_version)
            ):
                continue
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                x_data, y_data, t_data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=0.01)


def test_dropout(executor_kind):
    for dtype in ["float16", "float32"]:
        n, t, d = te.size_var("n"), te.size_var("t"), te.size_var("d")
        input_ty = relay.TensorType((n, t, d), dtype)
        x = relay.var("x", input_ty)
        y = relay.nn.dropout(x, rate=0.75)
        assert "rate=" in y.astext()
        yy = run_infer_type(y)
        assert yy.checked_type == input_ty

    in_np = np.random.random([4, 5, 6]).astype("float32")
    x = relay.const(in_np)
    y = relay.nn.dropout(x, rate=0.5)
    func = relay.Function([], y)
    for target, dev in tvm.testing.enabled_targets():
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)()
        tvm.testing.assert_allclose(op_res.numpy(), in_np, rtol=0.01)


def test_batch_norm():
    for dtype in ["float16", "float32"]:
        # beta and gamma ignored
        data = relay.var("data", relay.TensorType((3, 2, 1), dtype))
        beta = relay.var("beta", relay.TensorType((2,), dtype))
        gamma = relay.var("gamma", relay.TensorType((2,), dtype))
        moving_mean = relay.var("moving_mean", relay.TensorType((2,), dtype))
        moving_var = relay.var("moving_var", relay.TensorType((2,), dtype))
        y = relay.nn.batch_norm(
            data, gamma, beta, moving_mean, moving_var, center=False, scale=False
        )
        yy = run_infer_type(y.astuple())
        assert "center=" in yy.astext()
        assert yy.checked_type == relay.ty.TupleType(
            tvm.runtime.convert(
                [
                    relay.TensorType((3, 2, 1), dtype),
                    relay.TensorType((2,), dtype),
                    relay.TensorType((2,), dtype),
                ]
            )
        )

        # axis=1
        beta = relay.var("beta", relay.TensorType((3,), dtype))
        gamma = relay.var("gamma", relay.TensorType((3,), dtype))
        moving_mean = relay.var("moving_mean", relay.TensorType((3,), dtype))
        moving_var = relay.var("moving_var", relay.TensorType((3,), dtype))

        y = relay.nn.batch_norm(
            data, gamma, beta, moving_mean, moving_var, axis=0, center=False, scale=False
        )
        yy = run_infer_type(y.astuple())
        assert yy.checked_type == relay.ty.TupleType(
            tvm.runtime.convert(
                [
                    relay.ty.TensorType((3, 2, 1), dtype),
                    relay.ty.TensorType((3,), dtype),
                    relay.ty.TensorType((3,), dtype),
                ]
            )
        )

        # axis=-1
        data = relay.var("data", relay.TensorType((1, 2, 3), dtype))
        beta = relay.var("beta", relay.TensorType((3,), dtype))
        gamma = relay.var("gamma", relay.TensorType((3,), dtype))
        moving_mean = relay.var("moving_mean", relay.TensorType((3,), dtype))
        moving_var = relay.var("moving_var", relay.TensorType((3,), dtype))
        y = relay.nn.batch_norm(
            data, gamma, beta, moving_mean, moving_var, axis=-1, center=False, scale=False
        )
        yy = run_infer_type(y.astuple())
        assert yy.checked_type == relay.ty.TupleType(
            tvm.runtime.convert(
                [
                    relay.ty.TensorType((1, 2, 3), dtype),
                    relay.ty.TensorType((3,), dtype),
                    relay.ty.TensorType((3,), dtype),
                ]
            )
        )


def do_concat_test(shapes, t_shape, dtype, axis, dev, target):
    varsToConcat = []
    inputData = []
    pos = 0
    for s in shapes:
        varsToConcat.append(relay.var("x{}".format(pos), shape=s))
        inputData.append(np.random.rand(*s).astype(dtype))
        pos += 1
    t = relay.var("z", shape=t_shape, dtype=dtype)
    z = relay.concatenate(varsToConcat, axis=axis)
    z = relay.add(z, t)
    params = varsToConcat
    params.append(t)
    func = relay.Function(params, z)
    t_data = np.random.uniform(low=-10, high=10, size=t_shape).astype(dtype)
    ref_res = np.concatenate((tuple(inputData)), axis=axis) + t_data
    mod = tvm.IRModule.from_expr(func)

    executor = relay.create_executor("graph", mod=mod, device=dev, target=target)
    op_res1 = executor.evaluate()(*inputData, t_data)

    tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=0.000001)
    op_res2 = relay.create_executor("debug", device=dev, target=target).evaluate(func)(
        *inputData, t_data
    )
    tvm.testing.assert_allclose(op_res2.numpy(), ref_res, rtol=0.000001)


@tvm.testing.parametrize_targets("llvm")
def test_concatenate1(target, dev):
    np.random.seed(471)
    maxNumDimensions = 6
    shape = [4, 32, 16, 1, 31, 20, 21, 8, 28, 7]  # just randomly selected 10 numbers
    for dtype in ["float32"]:
        for dimsNum in range(1, maxNumDimensions):
            np.random.shuffle(shape)
            for axis in range(0, dimsNum):  # range should be (-dimsNum + 1, dimsNum)
                numToConcat = np.random.uniform(low=2, high=10, size=(1)).astype("int64")[0]
                shapes = []
                # the code below to normalize axes index. For some reasons tvm notifies about error if the axis is negative
                normalizedAxis = axis
                if axis < 0:
                    normalizedAxis += dimsNum
                finalSize = 0
                for i in range(0, numToConcat):
                    shp = tuple(shape[:dimsNum])
                    finalSize += shape[(i % len(shape))]
                    shapes.append(
                        shp[:normalizedAxis]
                        + tuple([shape[(i % len(shape))]])
                        + shp[normalizedAxis + 1 :]
                    )
                t_shape = shp[:normalizedAxis] + tuple([finalSize]) + shp[normalizedAxis + 1 :]
                do_concat_test(shapes, t_shape, dtype, axis, dev, target)


@tvm.testing.parametrize_targets("llvm")
def test_concatenate2(target, dev):
    # test to cover cases (1, .. , x, 1, .. , 1)
    np.random.seed(13)
    maxNumDimensions = 6
    shape = [8, 3, 25, 33, 12, 29, 5, 11, 29, 11]  # just randomly selected 10 numbers
    ind = 0
    for dtype in ["float32"]:
        for dimsNum in range(2, maxNumDimensions):
            np.random.shuffle(shape)
            for axis in range(-dimsNum + 1, dimsNum):  # range should be (-dimsNum + 1, dimsNum)
                numToConcat = np.random.uniform(low=2, high=10, size=(1)).astype("int64")[0]
                shapes = []
                # the code below to normalize axes index. For some reasons tvm notifies about error if the axis is negative
                normalizedAxis = axis
                if axis < 0:
                    normalizedAxis += dimsNum
                finalSize = 0
                for i in range(0, numToConcat):
                    axisVal = [1] * dimsNum
                    axisVal[axis] = shape[(ind % len(shape))]
                    ind += 1
                    finalSize += axisVal[axis]
                    shapes.append(tuple(axisVal))
                temp = [1] * dimsNum
                temp[axis] = finalSize
                t_shape = tuple(temp)
                do_concat_test(shapes, t_shape, dtype, axis, dev, target)


@tvm.testing.parametrize_targets("llvm")
def test_concatenate3(target, dev):
    np.random.seed(477)
    for dtype in ["float32"]:
        axis = -2
        ending = 1
        shapes = [[3, 2, 1, ending], [3, 2, 1, ending]]
        t_shape = [3, 2, 2, ending]
        do_concat_test(shapes, t_shape, dtype, axis, dev, target)


@tvm.testing.parametrize_targets("llvm")
def test_concatenate4(target, dev):
    np.random.seed(7)
    x_shape = (2, 1)
    x = relay.var("x", shape=x_shape, dtype="int64")
    concat = relay.concatenate([x], axis=1)
    f = relay.Function([x], concat)
    x_val = np.array([[33], [13]], dtype="int64")
    graph = relay.create_executor("graph", device=tvm.cpu(), target="llvm")
    op_res = graph.evaluate(f)(x_val)
    ref_res = np.concatenate([x_val], axis=1)
    tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=0.000001)


def test_batch_norm_fold_const():
    axis = 1
    dtype = "float32"
    shape = [4, 5, 6]

    data_np = np.random.random(shape).astype(dtype)
    beta_np = np.random.random(shape[axis]).astype(dtype)
    gamma_np = np.random.random(shape[axis]).astype(dtype)
    moving_mean_np = np.random.random(shape[axis]).astype(dtype)
    moving_var_np = np.random.random(shape[axis]).astype(dtype)

    data = relay.var("data", relay.TensorType(shape, dtype))
    beta = relay.var("beta", relay.TensorType((shape[1],), dtype))
    gamma = relay.var("gamma", relay.TensorType((shape[1],), dtype))
    moving_mean = relay.var("moving_mean", relay.TensorType((shape[1],), dtype))
    moving_var = relay.var("moving_var", relay.TensorType((shape[1],), dtype))
    out = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var, axis=axis).astuple()
    func = relay.Function([data, gamma, beta, moving_mean, moving_var], out)

    out_const = relay.nn.batch_norm(
        relay.const(data_np),
        relay.const(gamma_np),
        relay.const(beta_np),
        relay.const(moving_mean_np),
        relay.const(moving_var_np),
        axis=axis,
    ).astuple()
    func_const = relay.Function([], out_const)

    # Build the module with constants to have FoldConstant transform batch_norm.
    mod_const = tvm.IRModule.from_expr(func_const)
    mod_const = relay.transform.FoldConstant()(mod_const)

    const_data_out = mod_const["main"].body[0].data
    const_moving_mean_out = mod_const["main"].body[1].data
    const_moving_var_out = mod_const["main"].body[2].data

    # Run the Relay func without constants. This will use SimplyInference instead.
    vm_data_out, vm_moving_mean_out, vm_moving_var_out = relay.create_executor(
        "vm", device=tvm.device("llvm"), target="llvm"
    ).evaluate(func)(data_np, gamma_np, beta_np, moving_mean_np, moving_var_np)

    tvm.testing.assert_allclose(const_data_out.numpy(), vm_data_out.numpy())
    tvm.testing.assert_allclose(const_moving_mean_out.numpy(), vm_moving_mean_out.numpy())
    tvm.testing.assert_allclose(const_moving_var_out.numpy(), vm_moving_var_out.numpy())


@pytest.mark.xfail
def test_matmul_type_check():
    dtype = "float16"
    n, c, h, w = 2, 2, 2, 2
    x = relay.var("x", relay.TensorType((n, c, h, w), dtype))
    # it should fail since it does not match with m(2)
    mismatch_w = 3
    w = relay.var("w", relay.TensorType((mismatch_w, 2), dtype))
    y = relay.nn.matmul(x, w)
    yy = run_infer_type(y)

    i0 = relay.var("i0", shape=(1, 1), dtype="float32")
    i1 = relay.var("i1", shape=(1,), dtype="float32")
    with pytest.raises(tvm.TVMError):
        run_infer_type(relay.nn.matmul(i0, i1))


@tvm.testing.uses_gpu
def test_matmul(executor_kind):
    for dtype in ["float16", "float32"]:
        # Matmul accuracy for float16 is poor
        if dtype == "float16":
            continue
        n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
        x = relay.var("x", relay.TensorType((n, c, h, w), dtype))
        w = relay.var("w", relay.TensorType((2, w), dtype))
        y = relay.nn.matmul(x, w, units=2, transpose_b=True)
        assert "units=2" in y.astext()
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((n, c, h, 2), dtype)

        n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), 2
        x = relay.var("x", relay.TensorType((n, c, w, h), dtype))
        wh, ww = te.size_var("wh"), te.size_var("ww")
        w = relay.var("w", relay.TensorType((wh, ww), dtype))
        y = relay.nn.matmul(x, w, transpose_a=True)
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((n, c, h, ww), dtype)

        n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), 2
        x = relay.var("x", relay.TensorType((n, c, h, w), dtype))
        w = relay.var("w", relay.IncompleteType())
        y = relay.nn.matmul(x, w, units=2)
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((n, c, h, 2), dtype)

        x = relay.var("x", shape=(5, 10), dtype=dtype)
        w = relay.var("w", shape=(5, 2), dtype=dtype)
        z = relay.nn.matmul(x, w, transpose_a=True)

        # Check result.
        func = relay.Function([x, w], z)
        x_data = np.random.rand(5, 10).astype(dtype)
        w_data = np.random.rand(5, 2).astype(dtype)
        ref_res = np.dot(x_data.transpose(), w_data)

        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                x_data, w_data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


@pytest.mark.xfail
def test_dense_type_check():
    dtype = "float16"
    n, c, h, w = 2, 2, 2, 2
    x = relay.var("x", relay.TensorType((n, c, h, w), dtype))
    # it should fail since it does not match with m(2)
    mismatch_w = 3
    w = relay.var("w", relay.TensorType((2, mismatch_w), dtype))
    y = relay.nn.dense(x, w)
    yy = run_infer_type(y)


@tvm.testing.uses_gpu
def test_dense(executor_kind):
    for dtype in ["float16", "float32"]:
        # Dense accuracy for float16 is poor
        if dtype == "float16":
            continue
        n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
        x = relay.var("x", relay.TensorType((n, c, h, w), dtype))
        w = relay.var("w", relay.TensorType((2, w), dtype))
        y = relay.nn.dense(x, w, units=2)
        assert "units=2" in y.astext()
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((n, c, h, 2), dtype)

        n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), 2
        x = relay.var("x", relay.TensorType((n, c, h, w), dtype))
        wh, ww = te.size_var("wh"), te.size_var("ww")
        w = relay.var("w", relay.TensorType((ww, wh), dtype))
        y = relay.nn.dense(x, w)
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((n, c, h, ww), dtype)

        # test dynamic shape in inner
        m, k = 4, 2
        x = relay.var("x", relay.TensorType((m, k), dtype))
        k, nw = relay.Any(), 6
        w = relay.var("w", relay.TensorType((k, n), dtype))
        y = relay.nn.dense(x, w)
        yy = run_infer_type(y)

        n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), 2
        x = relay.var("x", relay.TensorType((n, c, h, w), dtype))
        w = relay.var("w", relay.IncompleteType())
        y = relay.nn.dense(x, w, units=2)
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((n, c, h, 2), dtype)

        x = relay.var("x", shape=(10, 5), dtype=dtype)
        w = relay.var("w", shape=(2, 5), dtype=dtype)
        z = relay.nn.dense(x, w)

        # Check result.
        func = relay.Function([x, w], z)
        x_data = np.random.rand(10, 5).astype(dtype)
        w_data = np.random.rand(2, 5).astype(dtype)
        ref_res = np.dot(x_data, w_data.T)

        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                x_data, w_data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
def test_dense_same_args_compile():
    for dtype in ["float32", "int8"]:
        x = relay.var("x", shape=(32, 64), dtype=dtype)
        out_dtype = "int32" if dtype == "int8" else "float32"
        f = relay.Function([x], relay.nn.dense(x, x, out_dtype=out_dtype))
        m = tvm.IRModule.from_expr(f)

        for target, _ in tvm.testing.enabled_targets():
            tvm.relay.build(m, target=target)


def test_dense_dtype():
    data_dtype = "uint8"
    weight_dtype = "int8"
    out_dtype = "uint8"
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), data_dtype))
    w = relay.var("w", relay.TensorType((2, w), weight_dtype))
    y = relay.nn.dense(x, w, units=2, out_dtype=out_dtype)
    assert "units=2" in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, h, 2), out_dtype)
    assert run_infer_type(yy.args[0]).checked_type.dtype == "uint8"
    assert run_infer_type(yy.args[1]).checked_type.dtype == "int8"


def test_bitserial_dense():
    m, k = te.size_var("m"), te.size_var("k")
    x = relay.var("x", relay.TensorType((m, k), "int16"))
    w = relay.var("w", relay.TensorType((k, 32), "int16"))
    y = relay.nn.bitserial_dense(x, w, units=32)
    "units=8" in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((m, 32), "int16")


@tvm.testing.requires_cascadelake
@pytest.mark.parametrize("m,n,k", [(32, 128, 96), (32, 128, 97)])
def test_dense_vnni(m, n, k):
    data_shape = (m, k)
    weight_shape = (n, k)

    for data_dtype in ["uint8", "int8"]:
        data = relay.var("data", shape=data_shape, dtype=data_dtype)
        weight = relay.var("weight", shape=weight_shape, dtype="int8")
        bias = relay.var("bias", shape=(weight_shape[0],), dtype="int32")
        dense = relay.nn.dense(data, weight, out_dtype="int32")
        out = relay.nn.bias_add(dense, bias)
        mod = tvm.IRModule.from_expr(out)

        target = "llvm -mcpu=cascadelake"
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target)

        asm = lib.lib.get_source("asm")
        assert "vpdpbusd" in asm

        dev = tvm.device(target, 0)
        runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

        a = np.random.uniform(1, 10, size=data_shape).astype(data_dtype)
        b = np.random.uniform(1, 10, size=weight_shape).astype("int8")
        c = np.random.uniform(1, 10, size=(weight_shape[0],)).astype("int32")

        runtime.set_input("data", a)
        runtime.set_input("weight", b)
        runtime.set_input("bias", c)
        runtime.run()

        out = runtime.get_output(0).numpy()
        ref = np.dot(a.astype("int32"), b.transpose().astype("int32")) + c

        np.testing.assert_equal(out, ref)


@pytest.mark.skip("Requires GFX10 AMDGPU")
def test_dense_rocm_sdot4():
    data_shape = (32, 96)
    weight_shape = (128, 96)

    data_dtype = "int8"
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype="int8")
    bias = relay.var("bias", shape=(weight_shape[0],), dtype="int32")
    dense = relay.nn.dense(data, weight, out_dtype="int32")
    out = relay.nn.bias_add(dense, bias)
    mod = tvm.IRModule.from_expr(out)

    target = "rocm -mattr=+dotprod"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)

    asm = lib.lib.imported_modules[0].get_source("asm")
    assert "v_dot4_i32_i8" in asm

    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    a = np.random.uniform(1, 10, size=data_shape).astype(data_dtype)
    b = np.random.uniform(1, 10, size=weight_shape).astype("int8")
    c = np.random.uniform(1, 10, size=(weight_shape[0],)).astype("int32")

    runtime.set_input("data", a)
    runtime.set_input("weight", b)
    runtime.set_input("bias", c)
    runtime.run()

    out = runtime.get_output(0).numpy()
    ref = np.dot(a.astype("int32"), b.transpose().astype("int32")) + c

    np.testing.assert_equal(out, ref)


def test_extern_concat_injective_fuse():
    # This is a subgraph from MobileBERT, which crashes compilation if buffers created in te.extern(...)
    # do not have their elem_offset explicitly set as a variable.

    # fmt: off
    mod = tvm.parser.fromtext(
        """
       #[version = "0.0.5"]
       def @main(%p0844: Tensor[(1, 384), int64], %p1652: Tensor[(2016, 128), float16]) {
        %1331 = cast(%p0844, dtype="int32");
        %1332 = take(%p1652, %1331, axis=0);
        %1333 = strided_slice(%1332, begin=[0, 1, 0], end=[1, 384, 128], strides=[1, 1, 1], axes=None);
        %1334 = strided_slice(%1332, begin=[0, 0, 0], end=[1, -1, 128], strides=[1, 1, 1], axes=None);
        %1335 = nn.pad(%1333, 0, pad_width=[[0, 0], [0, 1], [0, 0]]);
        %1336 = nn.pad(%1334, 0, pad_width=[[0, 0], [1, 0], [0, 0]]);
        %1337 = (%1335, %1332, %1336);
        %1338 = concatenate(%1337, axis=2);
        reshape(%1338, newshape=[-1, 384])
      }
    """
    )
    # fmt: on

    relay.build(mod, params={}, target="llvm")


if __name__ == "__main__":
    pytest.main([__file__])
