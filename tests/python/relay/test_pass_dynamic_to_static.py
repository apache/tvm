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
from tvm import te
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_infer_type, create_workload
import tvm.topi.testing
import tvm.testing


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)

    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def verify_func(func, data, ref_res, rtol=1e-5, atol=1e-7):
    assert isinstance(data, list)
    for target, ctx in tvm.testing.enabled_targets():
        for kind in ["graph", "vm", "debug"]:
            mod = tvm.ir.IRModule.from_expr(func)
            intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
            op_res = intrp.evaluate()(*data)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=rtol, atol=atol)


@tvm.testing.uses_gpu
def test_dynamic_to_static_reshape():
    def verify_reshape(shape, newshape, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(newshape, "float32"))
        z = relay.reshape(x, relay.shape_of(y))
        func = run_infer_type(relay.Function([x, y], z))
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("reshape")
        assert "newshape=" in zz.astext()
        assert zz.checked_type == relay.ty.TensorType(oshape, "float32")

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=newshape).astype("float32")
        ref_res = np.reshape(x_data, oshape)
        verify_func(func2, [x_data, y_data], ref_res)

    verify_reshape((2, 3, 4), (8, 3), (8, 3))
    verify_reshape((4, 7), (2, 7, 2), (2, 7, 2))


@tvm.testing.uses_gpu
def test_dynamic_to_static_double_reshape():
    def verify_reshape(shape, newshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(newshape, "float32"))
        z = relay.reshape(x, relay.shape_of(y))
        z = relay.reshape(z, relay.shape_of(x))
        func = run_infer_type(relay.Function([x, y], z))
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("reshape")
        assert "newshape=" in zz.astext()
        assert zz.checked_type == relay.ty.TensorType(shape, "float32")

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=newshape).astype("float32")
        verify_func(func2, [x_data, y_data], x_data)

    verify_reshape((2, 3, 4), (8, 3))
    verify_reshape((4, 7), (2, 7, 2))


@tvm.testing.uses_gpu
def test_dynamic_to_static_quad_reshape():
    def verify_reshape(shape, newshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(newshape, "float32"))
        z1 = relay.reshape(x, relay.shape_of(y))
        z2 = relay.reshape(z1, relay.shape_of(x))
        z3 = relay.reshape(z2, relay.shape_of(z1))
        z4 = relay.reshape(z3, relay.shape_of(z2))
        func = run_infer_type(relay.Function([x, y], z4))
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("reshape")
        assert "newshape=" in zz.astext()
        assert zz.checked_type == relay.ty.TensorType(shape, "float32")

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=newshape).astype("float32")
        verify_func(func2, [x_data, y_data], x_data)

    verify_reshape((2, 3, 4), (8, 3))
    verify_reshape((4, 7), (2, 7, 2))


@tvm.testing.uses_gpu
def test_dynamic_to_static_tile():
    def verify_tile(shape, reps, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(reps, "float32"))
        z = relay.tile(x, relay.shape_of(y))
        func = run_infer_type(relay.Function([x, y], z))
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("tile")
        assert zz.checked_type == relay.ty.TensorType(oshape, "float32")

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=reps).astype("float32")
        ref_res = np.tile(x_data, reps)
        verify_func(func2, [x_data, y_data], ref_res)

    verify_tile((2, 3, 4), (2, 1, 5), (4, 3, 20))
    verify_tile((4, 7), (4, 2), (16, 14))


@tvm.testing.uses_gpu
def test_dynamic_to_static_topk():
    def verify_topk(k, axis, ret_type, is_ascend, dtype):
        shape = (20, 100)
        x = relay.var("x", relay.TensorType(shape, "float32"))
        k_var = relay.const(k)
        out = relay.topk(x, k_var, axis, ret_type, is_ascend, dtype)
        if isinstance(out, relay.expr.TupleWrapper):
            out = out.astuple()
        func = relay.Function([x], out)

        np_data = np.random.uniform(size=shape).astype("float32")
        if is_ascend:
            np_indices = np.argsort(np_data, axis=axis)
        else:
            np_indices = np.argsort(-np_data, axis=axis)
        kk = k if k >= 1 else shape[axis]
        if axis == 0:
            np_indices = np_indices[:kk, :]
            np_values = np.zeros(np_indices.shape).astype("float32")
            for i in range(shape[1]):
                np_values[:, i] = np_data[np_indices[:, i], i]
        else:
            np_indices = np_indices[:, :kk]
            np_values = np.zeros(np_indices.shape).astype("float32")
            for i in range(shape[0]):
                np_values[i, :] = np_data[i, np_indices[i, :]]
        np_indices = np_indices.astype(dtype)

        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())
        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("topk")

        for target, ctx in tvm.testing.enabled_targets():
            if "llvm" not in target:
                continue
            for kind in ["graph", "vm", "debug"]:
                mod = tvm.ir.IRModule.from_expr(func2)
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(np_data)
                if ret_type == "both":
                    tvm.testing.assert_allclose(op_res[0].asnumpy(), np_values)
                    tvm.testing.assert_allclose(op_res[1].asnumpy(), np_indices)
                elif ret_type == "values":
                    tvm.testing.assert_allclose(op_res.asnumpy(), np_values)
                else:
                    tvm.testing.assert_allclose(op_res.asnumpy(), np_indices)

    np.random.seed(0)
    for k in [0, 1, 5]:
        for axis in [0, -1, 1]:
            for ret_type in ["both", "values", "indices"]:
                verify_topk(k, axis, ret_type, True, "int64")
                verify_topk(k, axis, ret_type, False, "float32")


@tvm.testing.uses_gpu
def test_dynamic_to_static_broadcast_to():
    def verify_broadcast_to(shape, broadcast_shape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(broadcast_shape, "float32"))
        z = relay.broadcast_to(x, shape=relay.shape_of(y))

        func = run_infer_type(relay.Function([x, y], z))
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("broadcast_to")
        assert zz.checked_type == relay.ty.TensorType(broadcast_shape, "float32")

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=broadcast_shape).astype("float32")

        ref_res = np.broadcast_to(x_data, y_data.shape)
        verify_func(func2, [x_data, y_data], ref_res)

    verify_broadcast_to((3, 1), (3, 3))


@tvm.testing.uses_gpu
def test_dynamic_to_static_zeros_ones():
    def verify_ones_zeros(shape, dtype):
        for op, ref in [(relay.zeros, np.zeros), (relay.ones, np.ones)]:
            x = relay.var("x", relay.TensorType(shape, dtype))
            y = op(relay.shape_of(x), dtype)

            func = run_infer_type(relay.Function([x], y))
            func2 = run_opt_pass(
                run_opt_pass(func, transform.DynamicToStatic()), transform.InferType()
            )

            zz = func2.body
            assert isinstance(zz, relay.Constant)
            assert zz.checked_type == relay.ty.TensorType(shape, dtype)

            x_data = np.random.uniform(low=1, high=1, size=shape)
            ref_res = ref(x_data.shape)
            verify_func(func2, [x_data], ref_res)

    verify_ones_zeros((1, 2, 3), "int64")
    verify_ones_zeros((9, 8, 3, 4), "float32")


@tvm.testing.uses_gpu
def test_dynamic_to_static_resize():
    def verify_resize(shape, scale, method, layout):
        if layout == "NHWC":
            size = (shape[1] * scale, shape[2] * scale)
        else:
            size = (shape[2] * scale, shape[3] * scale)

        x = relay.var("x", relay.TensorType(shape, "float32"))
        size_var = relay.const(np.array(size).astype("float32"))
        coord_trans = "asymmetric" if method == "nearest_neighbor" else "align_corners"
        z = relay.image.resize(
            x, size_var, layout, method, coordinate_transformation_mode=coord_trans
        )

        func = run_infer_type(relay.Function([x], z))
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("image.resize")

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")

        if method == "bilinear":
            ref_res = tvm.topi.testing.bilinear_resize_python(x_data, size, layout)
        else:
            ref_res = tvm.topi.testing.upsampling_python(x_data, (scale, scale), layout)
        verify_func(func2, [x_data], ref_res, rtol=1e-4, atol=1e-6)

    for method in ["bilinear", "nearest_neighbor"]:
        for layout in ["NCHW", "NHWC"]:
            verify_resize((1, 4, 4, 4), 2, method, layout)


@tvm.testing.uses_gpu
def test_dynamic_to_static_one_hot():
    def _verify(indices_shape, depth, on_value, off_value, axis, dtype):
        indices = relay.var("indices", relay.TensorType(indices_shape, "int32"))
        depth_var = relay.const(depth)
        on_value_const = relay.const(on_value)
        off_value_const = relay.const(off_value)
        out = relay.one_hot(indices, on_value_const, off_value_const, depth_var, axis, dtype)
        func = relay.Function([indices], out)

        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("one_hot")

        indices_np = np.random.randint(0, depth, size=indices_shape).astype("int32")
        out_np = tvm.topi.testing.one_hot(indices_np, on_value, off_value, depth, axis, dtype)
        verify_func(func2, [indices_np], out_np)

    _verify((3,), 3, 1, 0, -1, "int32")
    _verify((3,), 3, 1.0, 0.0, -1, "float32")
    _verify((2, 2), 5, 2, -2, 0, "int32")
    _verify((2, 2), 5, 0.5, -0.5, 1, "float32")
    _verify((3, 2, 4, 5), 6, 1, 0, 1, "int32")
    _verify((3, 2, 4, 5), 6, 1.0, 0.0, 0, "float32")


@tvm.testing.uses_gpu
def test_dynamic_to_static_full():
    def verify_full(fill_value, fill_shape, dtype):
        x = relay.var("x", relay.scalar_type(dtype))
        y = relay.var("y", relay.TensorType(fill_shape, "int64"))
        z = relay.full(x, relay.shape_of(y), dtype)

        func = run_infer_type(relay.Function([x, y], z))
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("full")

        ref_res = np.full(fill_shape, fill_value).astype(dtype)
        y_data = np.random.uniform(low=-1, high=1, size=fill_shape).astype("int64")
        verify_func(func2, [fill_value, y_data], ref_res)

    verify_full(4, (1, 2, 3, 4), "int32")
    verify_full(4.0, (1, 2, 8, 10), "float32")


def test_dynamic_to_static_upsampling():
    def verify_upsampling(data_shape, scale_h_val, scale_w_val, dtype):
        x = relay.var("x", relay.TensorType(data_shape, dtype))
        scale_h = relay.const(scale_h_val)
        scale_w = relay.const(scale_w_val)
        z = relay.nn.upsampling(x, scale_h, scale_w)

        func = run_infer_type(relay.Function([x], z))
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("nn.upsampling")

        x_data = np.random.uniform(size=data_shape).astype(dtype)
        ref_res = tvm.topi.testing.upsampling_python(x_data, (scale_h_val, scale_w_val), "NCHW")
        verify_func(func2, [x_data], ref_res)

    verify_upsampling((1, 16, 32, 32), 2, 2, "int8")
    verify_upsampling((1, 16, 32, 32), 4, 4, "int32")


def test_dynamic_to_static_upsampling3d():
    def verify_upsampling3d(data_shape, scale_d_val, scale_h_val, scale_w_val, dtype):
        x = relay.var("x", relay.TensorType(data_shape, dtype))
        scale_d = relay.const(scale_d_val)
        scale_h = relay.const(scale_h_val)
        scale_w = relay.const(scale_w_val)

        z = relay.nn.upsampling3d(x, scale_d, scale_h, scale_w)

        func = run_infer_type(relay.Function([x], z))
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("nn.upsampling3d")

        x_data = np.random.uniform(size=data_shape).astype(dtype)
        ref_res = tvm.topi.testing.upsampling3d_python(
            x_data, (scale_d_val, scale_h_val, scale_w_val), "NCDHW"
        )
        verify_func(func2, [x_data], ref_res)

    verify_upsampling3d((1, 1, 1, 1, 1), 2, 3, 4, "int8")
    verify_upsampling3d((5, 7, 8, 10, 32), 3, 2, 2, "int8")
    verify_upsampling3d((1, 4, 2, 5, 3), 5, 4, 3, "int32")


def test_dynamic_to_static_pad():
    def verify_pad(data_shape, pad_width, pad_val, dtype):
        x = relay.var("x", relay.TensorType(data_shape, dtype))
        z = relay.nn.pad(x, relay.const(np.array(pad_width)), pad_val)
        func = run_infer_type(relay.Function([x], z))
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())
        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("nn.pad")

        x_data = np.random.uniform(size=data_shape).astype(dtype)
        ref_res = np.pad(
            x_data, pad_width, "constant", constant_values=(((pad_val,) * 2),) * len(data_shape)
        )
        verify_func(func2, [x_data], ref_res)

    verify_pad((4, 10, 7, 7), ((1, 1), (2, 2), (3, 3), (4, 4)), 2.0, "int32")
    verify_pad((2, 7), ((1, 4), (2, 2)), 4.0, "float64")


def test_dynamic_to_static_strided_slice():
    def verify(dshape, begin, end, strides, output, slice_mode="end", test_ref=True, dtype="int32"):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        ndim = len(dshape)
        begin = begin if begin else [0] * ndim
        end = end if end else list(dshape)
        if strides:
            if len(strides) == 1:
                strides = strides * ndim
        else:
            strides = [1] * ndim

        # target numpy result
        x_data = np.random.uniform(size=dshape).astype("float32")
        ref_res = tvm.topi.testing.strided_slice_python(x_data, begin, end, strides, slice_mode)
        data = [x_data, np.array(begin), np.array(end)]

        begin = relay.const(begin, dtype=dtype)
        end = relay.const(end, dtype=dtype)

        if strides:
            data.append(np.array(strides))
            strides = relay.const(strides, dtype=dtype)
            z = relay.strided_slice(x, begin=begin, end=end, strides=strides, slice_mode=slice_mode)
        else:
            z = relay.strided_slice(x, begin=begin, end=end, slice_mode=slice_mode)
        func = relay.Function([x], z)

        func = run_infer_type(func)
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())
        assert isinstance(func2.body, relay.Call)
        assert func2.body.op == relay.op.get("strided_slice")
        verify_func(func2, [x_data], ref_res)

    verify((1, 3, 10, 10), [0, 0, 0, 0], [1, 3, 10, 10], [1], (0, 3, 10, 10), dtype="int64")
    verify(
        (1, 224, 224, 3),
        [0, 20, 20, 0],
        [1, 140, 140, 3],
        [1, 1, 1, 1],
        (1, 120, 120, 3),
        dtype="int64",
    )
    verify((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], (1, 3, 3), dtype="int16")
    verify((3, 4, 3), [0, 0, 0], [4, -5, 4], [1, -1, 2], (3, 1, 2))
    verify((3, 4, 3), [1, 1, 0], [4, 4, 3], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 1000, 3], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 4, 4], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 4, 3], None, (2, 3, 3))
    verify((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1], (1, 4, 3))
    verify((3, 4, 3), [1, -1, 0], [2, -3, 3], [1, -1, 1], (1, 2, 3))
    verify(
        (3, 4, 3), [1, 0, 0], [3, -1, 3], [1, 1, 1], (2, 4, 3), slice_mode="size", test_ref=False
    )
    verify((3, 4, 3), [1, 0, 0], [-1, 2, 3], [1, 1, 1], (2, 2, 3), slice_mode="size", test_ref=True)


if __name__ == "__main__":
    test_dynamic_to_static_reshape()
    test_dynamic_to_static_double_reshape()
    test_dynamic_to_static_quad_reshape()
    test_dynamic_to_static_tile()
    test_dynamic_to_static_topk()
    test_dynamic_to_static_broadcast_to()
    test_dynamic_to_static_zeros_ones()
    test_dynamic_to_static_resize()
    test_dynamic_to_static_one_hot()
    test_dynamic_to_static_full()
    test_dynamic_to_static_upsampling()
    test_dynamic_to_static_pad()
    test_dynamic_to_static_strided_slice()
