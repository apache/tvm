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
""" Support level10 operator test cases.
"""
import sys
import pytest

import numpy as np
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relay, te, topi
from tvm.relay import transform
from tvm.relay.testing import run_infer_type

executor_kind = tvm.testing.parameter("graph", "vm")


@tvm.testing.uses_gpu
def test_checkpoint(executor_kind):
    dtype = "float32"
    xs = [relay.var("x{}".format(i), dtype) for i in range(4)]
    f = relay.multiply(relay.add(xs[0], xs[1]), relay.add(xs[2], xs[3]))
    f_checkpoint = relay.annotation.checkpoint(f)

    func, func_checkpoint = relay.Function(xs, f), relay.Function(xs, f_checkpoint)
    f, f_checkpoint = run_infer_type(func), run_infer_type(func_checkpoint)
    assert f.checked_type == f_checkpoint.checked_type

    inputs = [np.random.uniform() for _ in range(len(xs))]
    for target, dev in tvm.testing.enabled_targets():
        f_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(f)(*inputs)
        f_checkpoint_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(
            f_checkpoint
        )(*inputs)
        tvm.testing.assert_allclose(f_res.numpy(), f_checkpoint_res.numpy(), 0, 0)


def test_checkpoint_alpha_equal():
    xs = [relay.var("x{}".format(i), relay.TensorType((1,), "float32")) for i in range(4)]
    f = relay.Function(
        xs,
        relay.annotation.checkpoint(
            relay.multiply(relay.add(xs[0], xs[1]), relay.add(xs[2], xs[3]))
        ),
    )
    df = transform.gradient(run_infer_type(f))

    # run PE and DCE
    with tvm.transform.PassContext(opt_level=3):
        # The expected output assumes DCE can elide 'dead writes' to references. At the time this unit test was
        # written DCE would elide all writes, which though unsound in general happens to work for this case. Preserve
        # that legacy behaviour here using 'ignore_impurity=True'.
        # TODO(mbs): Revisit once DCE supports dead reference writes.
        passes = [
            transform.PartialEvaluate(),
            transform.DeadCodeElimination(inline_once=True, ignore_impurity=True),
        ]
        mod = tvm.transform.Sequential(passes)(tvm.IRModule.from_expr(df))
        df = mod["main"]

    df_parsed = tvm.parser.parse_expr(
        """
        #[version = "0.0.5"]
        fn (%x: Tensor[(1), float32], %y: Tensor[(1), float32],
            %z: Tensor[(1), float32], %w: Tensor[(1), float32])
            ->  (Tensor[(1), float32],
                (Tensor[(1), float32], Tensor[(1), float32],
                 Tensor[(1), float32], Tensor[(1), float32])) {
            %0 = add(%x, %y);
            %1 = add(%z, %w);
            let %x1: Tensor[(1), float32] = multiply(%0, %1);
            let %x2: Tensor[(1), float32] = ones_like(%x1);
            let %x3: Tensor[(1), float32] = add(%x, %y);
            let %x4: Tensor[(1), float32] = add(%z, %w);
            %2 = zeros_like(%x3);
            %3 = multiply(%x2, %x4);
            %4 = collapse_sum_like(%3, %x3);
            let %x5: Tensor[(1), float32] = add(%2, %4);
            %5 = zeros_like(%x4);
            %6 = multiply(%x2, %x3);
            %7 = collapse_sum_like(%6, %x4);
            let %x6: Tensor[(1), float32] = add(%5, %7);
            %8 = zeros_like(%x);
            %9 = collapse_sum_like(%x5, %x);
            %10 = add(%8, %9);
            %11 = zeros_like(%y);
            %12 = collapse_sum_like(%x5, %y);
            %13 = add(%11, %12);
            %14 = zeros_like(%z);
            %15 = collapse_sum_like(%x6, %z);
            %16 = add(%14, %15);
            %17 = zeros_like(%w);
            %18 = collapse_sum_like(%x6, %w);
            %19 = add(%17, %18);
            %20 = (%10, %13, %16, %19);
            (%x1, %20)
        }
        """
    )

    tvm.ir.assert_structural_equal(df, df_parsed)


def test_checkpoint_alpha_equal_tuple():
    xs = [relay.var("x{}".format(i), relay.TensorType((1,), "float32")) for i in range(4)]
    f = relay.Function(
        xs,
        relay.annotation.checkpoint(
            relay.Tuple([relay.add(xs[0], xs[1]), relay.add(xs[2], xs[3])])
        ),
    )
    df = transform.gradient(run_infer_type(f))

    # run PE and DCE
    with tvm.transform.PassContext(opt_level=3):
        # See comment in test_checkpoint_alpha_equal above.
        # TODO(mbs): Revisit once DCE supports dead reference writes.
        passes = [
            transform.PartialEvaluate(),
            transform.DeadCodeElimination(inline_once=True, ignore_impurity=True),
        ]
        mod = tvm.transform.Sequential(passes)(tvm.IRModule.from_expr(df))
        df = mod["main"]

    df_parsed = tvm.parser.parse_expr(
        """
        #[version = "0.0.5"]
        fn (%x: Tensor[(1), float32], %y: Tensor[(1), float32],
            %z: Tensor[(1), float32], %w: Tensor[(1), float32])
            -> ((Tensor[(1), float32], Tensor[(1), float32]),
                (Tensor[(1), float32], Tensor[(1), float32],
                 Tensor[(1), float32], Tensor[(1), float32])) {
        let %x1: Tensor[(1), float32] = add(%x, %y) /* ty=Tensor[(1), float32] */;
        let %x2: Tensor[(1), float32] = add(%z, %w) /* ty=Tensor[(1), float32] */;
        let %x3: Tensor[(1), float32] = zeros_like(%x2) /* ty=Tensor[(1), float32] */;
        let %x4: Tensor[(1), float32] = ones_like(%x1) /* ty=Tensor[(1), float32] */;
        %0 = (%x1, %x2);
        %1 = zeros_like(%x) /* ty=Tensor[(1), float32] */;
        %2 = collapse_sum_like(%x4, %x) /* ty=Tensor[(1), float32] */;
        %3 = add(%1, %2) /* ty=Tensor[(1), float32] */;
        %4 = zeros_like(%y) /* ty=Tensor[(1), float32] */;
        %5 = collapse_sum_like(%x4, %y) /* ty=Tensor[(1), float32] */;
        %6 = add(%4, %5) /* ty=Tensor[(1), float32] */;
        %7 = zeros_like(%z) /* ty=Tensor[(1), float32] */;
        %8 = collapse_sum_like(%x3, %z) /* ty=Tensor[(1), float32] */;
        %9 = add(%7, %8) /* ty=Tensor[(1), float32] */;
        %10 = zeros_like(%w) /* ty=Tensor[(1), float32] */;
        %11 = collapse_sum_like(%x3, %w) /* ty=Tensor[(1), float32] */;
        %12 = add(%10, %11) /* ty=Tensor[(1), float32] */;
        %13 = (%3, %6, %9, %12);
        (%0, %13)
        }
        """
    )

    tvm.ir.assert_structural_equal(df, df_parsed)


@tvm.testing.uses_gpu
def test_collapse_sum_like(executor_kind):
    shape = (3, 4, 5, 6)
    shape_like = (4, 5, 6)
    dtype = "float32"
    x = relay.Var("x", relay.ty.TensorType(shape, dtype))
    y = relay.Var("y", relay.ty.TensorType(shape_like, dtype))
    z = relay.collapse_sum_like(x, y)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.ty.TensorType(shape_like, dtype)

    func = relay.Function([x, y], z)
    x = np.random.uniform(size=shape).astype(dtype)
    y = np.random.uniform(size=shape_like).astype(dtype)
    ref_res = np.sum(x, 0)
    for target, dev in tvm.testing.enabled_targets():
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x, y
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
def test_collapse_sum_to(executor_kind):
    shape = (3, 4, 5, 6)
    shape_to = (4, 5, 6)
    dtype = "float32"
    x = relay.Var("x", relay.ty.TensorType(shape, dtype))
    z = relay.collapse_sum_to(x, shape_to)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.ty.TensorType(shape_to, dtype)

    func = relay.Function([x], z)
    x = np.random.uniform(size=shape).astype(dtype)
    ref_res = np.sum(x, 0)
    for target, dev in tvm.testing.enabled_targets():
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(x)
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
def test_broadcast_to(executor_kind):
    shape = (4, 1, 6)
    shape_like = (3, 4, 5, 6)
    dtype = "float32"
    x = relay.Var("x", relay.ty.TensorType(shape, dtype))
    z = relay.broadcast_to(x, shape=shape_like)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.ty.TensorType(shape_like, dtype)

    func = relay.Function([x], z)
    x = np.random.uniform(size=shape).astype(dtype)
    ref_res = np.broadcast_to(x, shape_like)
    for target, dev in tvm.testing.enabled_targets():
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(x)
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
def test_broadcast_to_const_shape_int64(executor_kind):
    shape_like = relay.const(np.array([1, 5]), dtype="int64")
    x = relay.var("x", shape=(1,), dtype="int64")
    z = relay.broadcast_to(x, shape=shape_like)
    z = relay.sum(z, axis=0)

    f = relay.Function([x], z)

    x = np.random.randint(10, size=(1,), dtype="int64")
    ref_res = np.broadcast_to(x, (5,))
    for target, dev in tvm.testing.enabled_targets():
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(f)(x)
        tvm.testing.assert_allclose(op_res.numpy(), ref_res)


def test_broadcast_concat_shape_int64(executor_kind):
    x_shape = (1, 2, 1, 1)
    broadcast_shape = [1, 2, 2, 1]
    x = relay.var("data", relay.TensorType(x_shape, "float32"))
    broadcast_to = relay.op.broadcast_to(x, relay.const(broadcast_shape, dtype="int64"))
    concate = relay.op.concatenate((broadcast_to,), axis=0)

    f = relay.Function([x], concate)

    x = np.zeros(x_shape).astype("float32")
    ref_res = np.concatenate((np.broadcast_to(x, broadcast_shape),), axis=0)

    for target, dev in tvm.testing.enabled_targets():
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(f)(x)
        tvm.testing.assert_allclose(op_res.numpy(), ref_res)


def test_broadcast_pool2d_shape_int64(executor_kind):
    x_shape = (1, 3, 32, 32)
    out_shape = (2, 3, 32, 32)
    x = relay.var("data", shape=x_shape, dtype="float32")
    broadcast_to = relay.broadcast_to(x, shape=relay.const([2, 3, 32, 32], dtype="int64"))
    pool2d = relay.nn.max_pool2d(broadcast_to, pool_size=(3, 3), padding=(1, 1, 1, 1))
    sub = relay.subtract(broadcast_to, pool2d)

    f = relay.Function([x], sub)
    x = np.ones(x_shape).astype("float32")
    ref_res = np.zeros(out_shape).astype("float32")

    for target, dev in tvm.testing.enabled_targets():
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(f)(x)
        tvm.testing.assert_allclose(op_res.numpy(), ref_res)


@tvm.testing.uses_gpu
def test_broadcast_to_like(executor_kind):
    shape = (4, 1, 6)
    shape_like = (3, 4, 5, 6)
    dtype = "float32"
    x = relay.Var("x", relay.ty.TensorType(shape, dtype))
    y = relay.Var("y", relay.ty.TensorType(shape_like, dtype))
    z = relay.broadcast_to_like(x, y)

    zz = run_infer_type(z)
    assert zz.checked_type == relay.ty.TensorType(shape_like, dtype)

    func = relay.Function([x, y], z)
    x = np.random.uniform(size=shape).astype(dtype)
    y = np.random.uniform(size=shape_like).astype(dtype)
    ref_res = np.broadcast_to(x, shape_like)

    for target, dev in tvm.testing.enabled_targets():
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x, y
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


def np_slice_like(np_data, np_shape_like, axis=None):
    begin_idx = [0 for _ in np_data.shape]
    end_idx = list(np_data.shape)
    if axis:
        for i in axis:
            if i < 0:
                i = len(np_data.shape) + i
            end_idx[i] = np_shape_like.shape[i]
    else:
        for i in range(len(np_data.shape)):
            if i < len(np_shape_like.shape):
                end_idx[i] = np_shape_like.shape[i]
    slice_idx = []
    for b, e in zip(begin_idx, end_idx):
        slice_idx.append(slice(b, e))
    np_result = np_data[tuple(slice_idx)]
    return np_result


def verify_slice_like(executor_kind, data, slice_like, axes, output, dtype="float32"):
    x = relay.var("data", relay.TensorType(data, dtype))
    y = relay.var("slice_like", relay.TensorType(slice_like, dtype))
    z = relay.slice_like(x, y, axes)
    zz = run_infer_type(z)
    if axes:
        assert "axes" in z.astext()
    assert zz.checked_type == relay.ty.TensorType(output, dtype)

    if all(isinstance(v, int) == 0 for v in data) or all(
        isinstance(v, int) == 0 for v in slice_like
    ):
        return

    func = relay.Function([x, y], z)
    x_data = np.random.uniform(size=data).astype(dtype)
    y_data = np.random.uniform(size=slice_like).astype(dtype)
    ref_res = np_slice_like(x_data, y_data, axes)

    for target, dev in tvm.testing.enabled_targets():
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data, y_data
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
def test_slice_like(executor_kind):
    d1, d2, d3, d4 = te.var("d1"), te.var("d2"), te.var("d3"), te.var("d4")
    verify_slice_like(
        executor_kind, data=(d1, d2, d3), slice_like=(1, 2, 3), axes=None, output=(1, 2, 3)
    )
    verify_slice_like(
        executor_kind, data=(1, 2, 3), slice_like=(d1, d2, d3), axes=None, output=(d1, d2, d3)
    )
    verify_slice_like(
        executor_kind, data=(d2, d3, d4), slice_like=(d1, d2, d3), axes=(1, 2), output=(d2, d2, d3)
    )
    verify_slice_like(
        executor_kind, data=(3, 4, 5), slice_like=(1, 2, 3), axes=None, output=(1, 2, 3)
    )
    verify_slice_like(executor_kind, data=(3, 4, 5), slice_like=(1, 2), axes=None, output=(1, 2, 5))
    verify_slice_like(
        executor_kind, data=(3, 4, 5), slice_like=(1, 2, 3), axes=(1, 2), output=(3, 2, 3)
    )
    verify_slice_like(
        executor_kind, data=(3, 4, 5), slice_like=(1, 2, 3), axes=(-1, -3), output=(1, 4, 3)
    )
    verify_slice_like(
        executor_kind,
        data=(1, 3, 224, 224),
        slice_like=(1, 3, 112, 112),
        axes=(2, 3),
        output=(1, 3, 112, 112),
    )


@tvm.testing.uses_gpu
def test_reverse_reshape(executor_kind):
    def verify_reverse_reshape(executor_kind, shape, newshape, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.reverse_reshape(x, newshape=newshape)
        zz = run_infer_type(z)
        assert "newshape=" in z.astext()
        assert zz.checked_type == relay.ty.TensorType(oshape, "float32")

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        ref_res = np.reshape(x_data, oshape)
        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                x_data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_reverse_reshape(executor_kind, (2, 3, 4), (4, 0, 2), (4, 3, 2))
    verify_reverse_reshape(executor_kind, (2, 3, 4), (2, 0, 0), (2, 3, 4))
    verify_reverse_reshape(executor_kind, (2, 3, 4), (0, -1), (3, 8))
    verify_reverse_reshape(executor_kind, (2, 3, 4), (-1, 0), (6, 4))
    verify_reverse_reshape(executor_kind, (2, 3, 4), (0, -3), (2, 12))


def verify_batch_matmul_with_inputs(
    executor_kind, x, y, x_np, y_np, out_shape, dtype="float32", trans_x=False, trans_y=True
):
    z = relay.nn.batch_matmul(x, y, transpose_a=trans_x, transpose_b=trans_y)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.ty.TensorType(out_shape, dtype)

    input_vars = relay.analysis.free_vars(z)
    func = relay.Function(input_vars, z)
    z_np = tvm.topi.testing.batch_matmul(x_np, y_np, trans_x=trans_x, trans_y=trans_y)

    for target, dev in tvm.testing.enabled_targets():
        if len(input_vars) == 2:
            z = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                x_np, y_np
            )
        else:
            z = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(x_np)
        tvm.testing.assert_allclose(z.numpy(), z_np, rtol=1e-5, atol=1e-5)


def verify_batch_matmul(
    executor_kind, x_shape, y_shape, out_shape, dtype="float32", trans_x=False, trans_y=True
):
    x = relay.var("x", relay.TensorType(x_shape, dtype))
    y = relay.var("y", relay.TensorType(y_shape, dtype))
    x_np = np.random.uniform(size=x_shape).astype(dtype)
    y_np = np.random.uniform(size=y_shape).astype(dtype)
    verify_batch_matmul_with_inputs(
        executor_kind, x, y, x_np, y_np, out_shape, dtype, trans_x, trans_y
    )


@tvm.testing.uses_gpu
def test_batch_matmul(executor_kind):
    b, m, n, k = te.size_var("b"), te.size_var("m"), te.size_var("n"), te.size_var("k")
    x = relay.var("x", relay.TensorType((b, m, k), "float32"))
    y = relay.var("y", relay.TensorType((b, n, k), "float32"))
    z = relay.nn.batch_matmul(x, y)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((b, m, n), "float32")

    verify_batch_matmul(
        executor_kind, (1, 16, 32), (1, 16, 32), (1, 16, 16), trans_x=False, trans_y=True
    )
    verify_batch_matmul(
        executor_kind, (5, 16, 32), (5, 16, 32), (5, 16, 16), trans_x=False, trans_y=True
    )
    verify_batch_matmul(
        executor_kind, (5, 16, 32), (5, 20, 32), (5, 16, 20), trans_x=False, trans_y=True
    )
    verify_batch_matmul(
        executor_kind, (30, 16, 32), (30, 20, 32), (30, 16, 20), trans_x=False, trans_y=True
    )
    verify_batch_matmul(
        executor_kind, (1, 32, 16), (1, 16, 32), (1, 16, 16), trans_x=True, trans_y=True
    )
    verify_batch_matmul(
        executor_kind, (5, 16, 32), (5, 32, 16), (5, 16, 16), trans_x=False, trans_y=False
    )
    verify_batch_matmul(
        executor_kind, (5, 32, 16), (5, 32, 20), (5, 16, 20), trans_x=True, trans_y=False
    )

    x_np = np.random.randn(10, 27, 64).astype("float32")
    x = relay.var("x", shape=x_np.shape)
    verify_batch_matmul_with_inputs(executor_kind, x, x, x_np, x_np, (10, 27, 27))


@tvm.testing.requires_cascadelake
@pytest.mark.parametrize(
    "b,m,n,k",
    [
        (16, 32, 128, 96),
        (16, 32, 128, 97),
        (16, 32, 129, 96),
    ],
)
def test_batch_matmul_vnni(b, m, n, k):
    x_shape = (b, m, k)
    y_shape = (b, n, k)
    z_shape = (b, m, n)

    for lhs_dtype in ["uint8", "int8"]:
        x = relay.var("x", shape=x_shape, dtype=lhs_dtype)
        y = relay.var("y", shape=y_shape, dtype="int8")
        z = relay.var("z", shape=z_shape, dtype="int32")
        bmm = relay.nn.batch_matmul(x, y, out_dtype="int32")
        out = bmm + z
        mod = tvm.IRModule.from_expr(out)

        target = "llvm -mcpu=cascadelake"
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target)

        asm = lib.lib.get_source("asm")
        assert "vpdpbusd" in asm

        dev = tvm.device(target, 0)
        runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

        x_np = np.random.uniform(1, 10, size=x_shape).astype(lhs_dtype)
        y_np = np.random.uniform(1, 10, size=y_shape).astype("int8")
        z_np = np.random.uniform(1, 10, size=z_shape).astype("int32")

        runtime.set_input("x", x_np)
        runtime.set_input("y", y_np)
        runtime.set_input("z", z_np)
        runtime.run()

        out = runtime.get_output(0).numpy()
        ref = tvm.topi.testing.batch_matmul(x_np, y_np, out_dtype="int32") + z_np

        np.testing.assert_equal(out, ref)


@pytest.mark.skip("Requires GFX10 AMDGPU")
def test_batch_matmul_rocm_sdot4():
    x_shape = (16, 32, 96)
    y_shape = (16, 128, 96)

    lhs_dtype = "int8"
    x = relay.var("x", shape=x_shape, dtype=lhs_dtype)
    y = relay.var("y", shape=y_shape, dtype="int8")
    bmm = relay.nn.batch_matmul(x, y, out_dtype="int32")

    mod = tvm.IRModule.from_expr(bmm)

    target = "rocm -mattr=+dotprod"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)

    asm = lib.lib.imported_modules[0].get_source("asm")
    assert "v_dot4_i32_i8" in asm

    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    x_np = np.random.uniform(1, 10, size=x_shape).astype(lhs_dtype)
    y_np = np.random.uniform(1, 10, size=y_shape).astype("int8")

    runtime.set_input("x", x_np)
    runtime.set_input("y", y_np)
    runtime.run()

    out = runtime.get_output(0).numpy()
    ref = tvm.topi.testing.batch_matmul(x_np, y_np, out_dtype="int32")

    np.testing.assert_equal(out, ref)


@tvm.testing.uses_gpu
def test_shape_of():
    shape = (10, 5, 12)
    x = relay.var("x", shape=shape)
    func = relay.Function([x], relay.op.shape_of(x))
    func = run_infer_type(func)
    x_data = np.random.rand(*shape).astype("float32")
    for target, dev in tvm.testing.enabled_targets():
        # Because using graph executor, this op will be optimized after
        # constant folding pass, here we only test with interpreter
        for kind in ["vm"]:
            op_res = relay.create_executor(kind, device=dev, target=target).evaluate(func)(x_data)
            tvm.testing.assert_allclose(op_res.numpy(), np.array(shape).astype("int32"))


@tvm.testing.uses_gpu
def test_ndarray_size(executor_kind):
    def verify_ndarray_size(shape):
        x = relay.var("x", shape=shape)
        func = relay.Function([x], relay.op.ndarray_size(x))
        func = run_infer_type(func)

        x_data = np.random.uniform(size=shape).astype("float32")
        ref_res = np.size(x_data)
        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                x_data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res)

    verify_ndarray_size((2, 3, 5))
    verify_ndarray_size((2, 3, 5, 7))


def verify_adaptive_pool(dshape, out_size, pool_type, layout, dtype, opfunc):
    for shape_dtype in ["int32", "int64"]:
        x = relay.var("x", shape=[tvm.tir.IntImm(shape_dtype, x) for x in dshape], dtype=dtype)
        y = opfunc(x, out_size, layout)
        func = relay.Function([x], y)

        np_data = np.random.uniform(low=0, high=255, size=dshape).astype(dtype)
        np_out = tvm.topi.testing.adaptive_pool(np_data, out_size, pool_type, layout)

        for target, dev in tvm.testing.enabled_targets():
            relay_out = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                np_data
            )
            tvm.testing.assert_allclose(relay_out.numpy(), np_out, rtol=1e-5, atol=1e-5)


def verify_adaptive_pool1d(dshape, out_size, pool_type, layout="NCW", dtype="float32"):
    opfunc = relay.nn.adaptive_avg_pool1d if pool_type == "avg" else relay.nn.adaptive_max_pool1d
    verify_adaptive_pool(dshape, out_size, pool_type, layout, dtype, opfunc)


def verify_adaptive_pool2d(dshape, out_size, pool_type, layout="NCHW", dtype="float32"):
    opfunc = relay.nn.adaptive_avg_pool2d if pool_type == "avg" else relay.nn.adaptive_max_pool2d
    verify_adaptive_pool(dshape, out_size, pool_type, layout, dtype, opfunc)


def verify_adaptive_pool3d(dshape, out_size, pool_type, layout="NCDHW", dtype="float32"):
    opfunc = relay.nn.adaptive_avg_pool3d if pool_type == "avg" else relay.nn.adaptive_max_pool3d
    verify_adaptive_pool(dshape, out_size, pool_type, layout, dtype, opfunc)


@tvm.testing.uses_gpu
def test_adaptive_pool():
    verify_adaptive_pool1d((1, 9, 224), (1), "max")
    verify_adaptive_pool1d((1, 3, 224), (3), "avg")
    verify_adaptive_pool1d((1, 3, 224), (3), "avg", dtype="int32")
    verify_adaptive_pool1d((1, 14, 78), (13), "max")
    verify_adaptive_pool1d((1, 5, 97), (96), "avg")
    verify_adaptive_pool1d((1, 224, 3), (1), "max", layout="NWC")
    verify_adaptive_pool1d((1, 3, 224), (3), "avg", layout="NWC")
    verify_adaptive_pool2d((1, 9, 224, 224), (1, 1), "max")
    verify_adaptive_pool2d((1, 3, 224, 224), (2, 3), "avg")
    verify_adaptive_pool2d((1, 3, 224, 224), (2, 3), "avg", dtype="int32")
    verify_adaptive_pool2d((1, 14, 56, 78), (34, 13), "max")
    verify_adaptive_pool2d((1, 5, 46, 97), (4, 96), "avg")
    verify_adaptive_pool2d((1, 224, 224, 3), (1, 1), "max", layout="NHWC")
    verify_adaptive_pool2d((1, 3, 224, 224), (2, 3), "avg", layout="NHWC")
    verify_adaptive_pool3d((1, 16, 32, 32, 32), (1, 1, 1), "max", layout="NCDHW")
    verify_adaptive_pool3d((1, 16, 32, 32, 32), (1, 1, 1), "avg", layout="NCDHW")
    verify_adaptive_pool3d((1, 16, 32, 32, 32), (1, 1, 1), "avg", layout="NDHWC")
    verify_adaptive_pool3d((1, 16, 32, 32, 32), (1, 1, 1), "avg", layout="NCDHW", dtype="int32")
    verify_adaptive_pool3d((1, 16, 32, 32, 32), (1, 1, 1), "avg", layout="NDHWC", dtype="int32")
    verify_adaptive_pool3d((1, 16, 32, 32, 32), (2, 4, 4), "max", layout="NDHWC")


@tvm.testing.uses_gpu
def test_sequence_mask(executor_kind):
    def _verify(data_shape, mask_value, axis, dtype, itype):
        max_length = data_shape[axis]
        nbatch = data_shape[1 - axis]
        data = relay.var("data", relay.TensorType(data_shape, dtype))
        valid_length = relay.var("valid_length", relay.TensorType((nbatch,), itype))
        out = relay.sequence_mask(data, valid_length, mask_value, axis)
        checked = run_infer_type(out)
        assert checked.checked_type == relay.ty.TensorType(data_shape, dtype)
        func = relay.Function([data, valid_length], out)
        data_np = np.random.uniform(size=data_shape).astype(dtype)
        valid_length_np = np.random.randint(0, max_length, size=nbatch).astype(itype)
        gt_out_np = tvm.topi.testing.sequence_mask(data_np, valid_length_np, mask_value, axis)

        for target, dev in tvm.testing.enabled_targets():
            out_relay = relay.create_executor(executor_kind, device=dev, target=target).evaluate(
                func
            )(data_np, valid_length_np)
            tvm.testing.assert_allclose(out_relay.numpy(), gt_out_np)

    _verify((5, 10), 0.0, 1, "float32", "int32")
    _verify((2, 3, 5, 3), 0.0, 0, "float32", "int64")
    _verify((5, 8, 3), 0.1, 1, "float64", "float32")


@tvm.testing.uses_gpu
def test_one_hot(executor_kind):
    def _get_oshape(indices_shape, depth, axis):
        oshape = []
        true_axis = len(indices_shape) if axis == -1 else axis
        ndim = len(indices_shape) + 1
        indices_index = 0
        for i in range(0, ndim):
            if i == true_axis:
                oshape.append(depth)
            else:
                oshape.append(indices_shape[indices_index])
                indices_index += 1

        return oshape

    def _verify(indices_shape, depth, on_value, off_value, axis, dtype):
        indices = relay.var("indices", relay.TensorType(indices_shape, "int32"))
        on_value_const = relay.const(on_value)
        off_value_const = relay.const(off_value)
        out = relay.one_hot(indices, on_value_const, off_value_const, depth, axis, dtype)
        checked = run_infer_type(out)
        assert checked.checked_type == relay.ty.TensorType(
            _get_oshape(indices_shape, depth, axis), dtype
        )
        func = relay.Function([indices], out)
        indices_np = np.random.randint(0, depth, size=indices_shape).astype("int32")
        out_np = tvm.topi.testing.one_hot(indices_np, on_value, off_value, depth, axis, dtype)

        for target, dev in tvm.testing.enabled_targets():
            out_relay = relay.create_executor(executor_kind, device=dev, target=target).evaluate(
                func
            )(indices_np)
            tvm.testing.assert_allclose(out_relay.numpy(), out_np)

    _verify((3,), 3, 1, 0, -1, "int32")
    _verify((3,), 3, 1.0, 0.0, -1, "float32")
    _verify((2, 2), 5, 2, -2, 0, "int32")
    _verify((2, 2), 5, 0.5, -0.5, 1, "float32")
    _verify((3, 2, 4, 5), 6, 1, 0, 1, "int32")
    _verify((3, 2, 4, 5), 6, 1.0, 0.0, 0, "float32")


@tvm.testing.uses_gpu
def test_matrix_set_diag(executor_kind):
    def _verify(input_shape, diagonal_shape, dtype, k=0, align="RIGHT_LEFT"):
        input = relay.var("input", relay.TensorType(input_shape, dtype))
        diagonal = relay.var("diagonal", relay.TensorType(diagonal_shape, dtype))
        out = relay.matrix_set_diag(input, diagonal, k, align)

        in_type = run_infer_type(input)
        out_type = run_infer_type(out)
        assert in_type.checked_type == out_type.checked_type

        func = relay.Function([input, diagonal], out)
        input_np = np.random.randint(-100, 100, size=input_shape).astype(dtype)
        diagonal_np = np.random.randint(-100, 100, size=diagonal_shape).astype(dtype)
        out_np = tvm.topi.testing.matrix_set_diag(input_np, diagonal_np, k, align)

        for target, dev in tvm.testing.enabled_targets():
            out_relay = relay.create_executor(executor_kind, device=dev, target=target).evaluate(
                func
            )(input_np, diagonal_np)
            tvm.testing.assert_allclose(out_relay.numpy(), out_np)

    _verify((2, 2), (2,), "float32")
    _verify((4, 3, 3), (4, 3), "int32")
    _verify((2, 3, 4), (2, 3), "float32", 1)
    _verify((2, 3, 4), (2, 4, 3), "int32", (-1, 2), "LEFT_RIGHT")
    _verify((2, 3, 4), (2, 4, 3), "int32", (-1, 2), "LEFT_LEFT")
    _verify((2, 3, 4), (2, 4, 3), "int32", (-1, 2), "RIGHT_RIGHT")


@tvm.testing.parametrize_targets
def test_nll_loss(executor_kind, dev, target):
    def _get_oshape(target_shape, reduction):
        if reduction == "none":
            return target_shape
        else:
            return []

    def _verify(prediction_shape, reduction="mean", ignore_index=-100, dtype="float32"):
        C = prediction_shape[1]
        target_shape = prediction_shape[:1] + prediction_shape[2:]

        predictions = relay.var("predictions", relay.TensorType(prediction_shape, dtype))
        targets = relay.var("targets", relay.TensorType(target_shape, "int32"))
        weights = relay.var("weights", relay.TensorType((C,), dtype))
        out = relay.nn.nll_loss(predictions, targets, weights, reduction, ignore_index)
        checked = run_infer_type(out)
        assert checked.checked_type == relay.ty.TensorType(
            _get_oshape(target_shape, reduction), dtype
        )
        func = relay.Function([predictions, targets, weights], out)
        predictions_np = np.random.uniform(size=prediction_shape).astype(dtype)
        targets_np = np.random.randint(0, C, target_shape).astype("int32")
        weights_np = np.random.uniform(size=(C,)).astype(dtype)
        out_np = tvm.topi.testing.nll_loss(
            predictions_np, targets_np, weights_np, reduction, ignore_index
        )

        out_relay = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            predictions_np, targets_np, weights_np
        )
        tvm.testing.assert_allclose(out_relay.numpy(), out_np, rtol=1e-6, atol=1e-6)

    _verify((10, 5))
    _verify((10, 5, 2, 2))
    _verify((10, 5), reduction="sum")
    _verify((10, 5), reduction="none")
    _verify((10, 5), ignore_index=3)
    _verify((10, 5), dtype="float64")


if __name__ == "__main__":
    tvm.testing.main()
