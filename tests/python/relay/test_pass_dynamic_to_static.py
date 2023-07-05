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
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_infer_type, create_workload
import tvm.topi.testing
import tvm.testing


def run_opt_pass(expr, opt_pass, params=None):
    assert isinstance(opt_pass, tvm.transform.Pass)

    mod = tvm.IRModule.from_expr(expr)
    if params is not None:
        mod["main"] = bind_params_by_name(mod["main"], params)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def verify_func(func, data, ref_res, rtol=1e-5, atol=1e-7):
    assert isinstance(data, list)
    for target, dev in tvm.testing.enabled_targets():
        for kind in ["graph", "vm", "debug"]:
            mod = tvm.ir.IRModule.from_expr(func)
            op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                *data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=rtol, atol=atol)


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
def test_dynamic_to_static_squeeze():
    def verify_squeeze(shape, axis, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(axis, "float32"))
        z = relay.squeeze(x, relay.shape_of(y))
        func = run_infer_type(relay.Function([x, y], z))
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("squeeze")
        assert "axis=" in zz.astext()
        assert zz.checked_type == relay.ty.TensorType(oshape, "float32")

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=axis).astype("float32")
        ref_res = np.squeeze(x_data, axis)
        verify_func(func2, [x_data, y_data], ref_res)

    verify_squeeze((1, 3, 4, 1), (0,), (3, 4, 1))
    verify_squeeze((1, 3, 4, 1), (3,), (1, 3, 4))
    verify_squeeze((1, 3, 4, 1), (0, 3), (3, 4))


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
        k_var = relay.var("k", relay.TensorType((), "int32"))
        out = relay.topk(x, k_var, axis, ret_type, is_ascend, dtype)
        if isinstance(out, relay.expr.TupleWrapper):
            out = out.astuple()
        func = relay.Function([x, k_var], out)
        params = {"k": k}

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

        func2 = run_opt_pass(
            run_opt_pass(func, transform.DynamicToStatic(), params), transform.InferType()
        )
        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("topk")

        for target, dev in tvm.testing.enabled_targets():
            if "llvm" not in target:
                continue
            for kind in ["graph", "vm", "debug"]:
                mod = tvm.ir.IRModule.from_expr(func2)
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    np_data
                )
                if ret_type == "both":
                    tvm.testing.assert_allclose(op_res[0].numpy(), np_values)
                    tvm.testing.assert_allclose(op_res[1].numpy(), np_indices)
                elif ret_type == "values":
                    tvm.testing.assert_allclose(op_res.numpy(), np_values)
                else:
                    tvm.testing.assert_allclose(op_res.numpy(), np_indices)

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
                run_opt_pass(func, transform.DynamicToStatic()),
                transform.InferType(),
            )

            zz = func2.body
            assert zz.checked_type == relay.ty.TensorType(shape, dtype)

            x_data = np.random.uniform(low=1, high=1, size=shape)
            ref_res = ref(x_data.shape)
            verify_func(func2, [x_data], ref_res)

    verify_ones_zeros((1, 2, 3), "int64")
    verify_ones_zeros((9, 8, 3, 4), "float32")


@tvm.testing.uses_gpu
def test_dynamic_to_static_resize2d():
    def verify_resize(shape, scale, method, layout):
        if layout == "NHWC":
            size = (shape[1] * scale, shape[2] * scale)
        else:
            size = (shape[2] * scale, shape[3] * scale)

        x = relay.var("x", relay.TensorType(shape, "float32"))
        size_var = relay.var("size", relay.TensorType((len(size),), "float32"))
        coord_trans = "asymmetric" if method == "nearest_neighbor" else "align_corners"
        z = relay.image.resize2d(
            x, size_var, None, layout, method, coordinate_transformation_mode=coord_trans
        )
        params = {"size": np.array(size).astype("float32")}

        func = run_infer_type(relay.Function([x, size_var], z))
        func2 = run_opt_pass(
            run_opt_pass(func, transform.DynamicToStatic(), params), transform.InferType()
        )

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("image.resize2d")

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        ref_res = tvm.topi.testing.resize2d_python(
            x_data, (scale, scale), layout, method, coord_trans
        )

    for method in ["linear", "nearest_neighbor"]:
        for layout in ["NCHW", "NHWC"]:
            verify_resize((1, 4, 4, 4), 2, method, layout)


@tvm.testing.uses_gpu
def test_dynamic_to_static_one_hot():
    def _verify(indices_shape, depth, on_value, off_value, axis, dtype):
        indices = relay.var("indices", relay.TensorType(indices_shape, "int32"))
        depth_var = relay.const(depth)
        on_value_var = relay.var("on_value", relay.TensorType((), "int32"))
        off_value_var = relay.var("off_value", relay.TensorType((), "int32"))
        out = relay.one_hot(indices, on_value_var, off_value_var, depth_var, axis, dtype)
        params = {
            "on_value": on_value,
            "off_value": off_value,
        }

        func = relay.Function([indices, on_value_var, off_value_var], out)
        func2 = run_opt_pass(
            run_opt_pass(func, transform.DynamicToStatic(), params), transform.InferType()
        )

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
        scale_h = relay.var("scale_h", relay.TensorType((), "float32"))
        scale_w = relay.var("scale_w", relay.TensorType((), "float32"))
        z = relay.nn.upsampling(x, scale_h, scale_w)
        params = {
            "scale_h": scale_h_val,
            "scale_w": scale_w_val,
        }

        func = run_infer_type(relay.Function([x, scale_h, scale_w], z))
        func2 = run_opt_pass(
            run_opt_pass(func, transform.DynamicToStatic(), params), transform.InferType()
        )

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("nn.upsampling")

        x_data = np.random.uniform(size=data_shape).astype(dtype)
        ref_res = tvm.topi.testing.resize2d_python(
            x_data, (scale_h_val, scale_w_val), "NCHW", "nearest_neighbor", "asymmetric"
        )
        verify_func(func2, [x_data], ref_res)

    verify_upsampling((1, 16, 32, 32), 2, 2, "int8")
    verify_upsampling((1, 16, 32, 32), 4, 4, "int32")


def test_dynamic_to_static_upsampling3d():
    def verify_upsampling3d(data_shape, scale_d_val, scale_h_val, scale_w_val, dtype):
        x = relay.var("x", relay.TensorType(data_shape, dtype))
        scale_d = relay.var("scale_d", relay.TensorType((), "float32"))
        scale_h = relay.var("scale_h", relay.TensorType((), "float32"))
        scale_w = relay.var("scale_w", relay.TensorType((), "float32"))

        z = relay.nn.upsampling3d(x, scale_d, scale_h, scale_w)
        params = {
            "scale_d": scale_d_val,
            "scale_h": scale_h_val,
            "scale_w": scale_w_val,
        }

        func = run_infer_type(relay.Function([x, scale_d, scale_h, scale_w], z))
        func2 = run_opt_pass(
            run_opt_pass(func, transform.DynamicToStatic(), params), transform.InferType()
        )

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("nn.upsampling3d")

        x_data = np.random.uniform(size=data_shape).astype(dtype)
        ref_res = tvm.topi.testing.resize3d_python(
            x_data,
            (scale_d_val, scale_h_val, scale_w_val),
            "NCDHW",
            "nearest_neighbor",
            "asymmetric",
        )
        verify_func(func2, [x_data], ref_res)

    verify_upsampling3d((1, 1, 1, 1, 1), 2, 3, 4, "int8")
    verify_upsampling3d((5, 7, 8, 10, 32), 3, 2, 2, "int8")
    verify_upsampling3d((1, 4, 2, 5, 3), 5, 4, 3, "int32")


def test_dynamic_to_static_pad():
    def verify_pad(data_shape, pad_width_val, pad_val, dtype):
        x = relay.var("x", relay.TensorType(data_shape, dtype))
        pad_width = relay.var(
            "pad_width", relay.TensorType((len(pad_width_val), len(pad_width_val[0])), "int32")
        )
        z = relay.nn.pad(x, pad_width, pad_val)
        func = run_infer_type(relay.Function([x, pad_width], z))
        params = {"pad_width": np.array(pad_width_val)}
        func2 = run_opt_pass(
            run_opt_pass(func, transform.DynamicToStatic(), params), transform.InferType()
        )
        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("nn.pad")

        x_data = np.random.uniform(size=data_shape).astype(dtype)
        ref_res = np.pad(
            x_data, pad_width_val, "constant", constant_values=(((pad_val,) * 2),) * len(data_shape)
        )
        verify_func(func2, [x_data], ref_res)

    verify_pad((4, 10, 7, 7), ((1, 1), (2, 2), (3, 3), (4, 4)), 2.0, "int32")
    verify_pad((2, 7), ((1, 4), (2, 2)), 4.0, "float64")


def test_dynamic_to_static_strided_slice():
    def verify(
        dshape,
        begin_val,
        end_val,
        strides_val,
        output,
        slice_mode="end",
        test_ref=True,
        dtype="int32",
    ):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        ndim = len(dshape)
        begin_val = begin_val if begin_val else [0] * ndim
        end_val = end_val if end_val else list(dshape)
        if strides_val:
            if len(strides_val) == 1:
                strides_val = strides_val * ndim
        else:
            strides_val = [1] * ndim

        # target numpy result
        x_data = np.random.uniform(size=dshape).astype("float32")
        ref_res = tvm.topi.testing.strided_slice_python(
            x_data, begin_val, end_val, strides_val, slice_mode
        )
        data = [x_data, np.array(begin_val), np.array(end_val)]

        begin = relay.var("begin", relay.TensorType((len(begin_val),), dtype))
        end = relay.var("end", relay.TensorType((len(end_val),), dtype))

        func_params = [x, begin, end]
        if strides_val:
            data.append(np.array(strides_val))
            strides = relay.var("strides", relay.TensorType((len(strides_val),), dtype))
            z = relay.strided_slice(x, begin=begin, end=end, strides=strides, slice_mode=slice_mode)
            func_params.append(strides)
        else:
            z = relay.strided_slice(x, begin=begin, end=end, slice_mode=slice_mode)
        func = relay.Function(func_params, z)
        params = {"begin": begin_val, "end": end_val, "strides": strides_val}

        func = run_infer_type(func)
        func2 = run_opt_pass(
            run_opt_pass(func, transform.DynamicToStatic(), params), transform.InferType()
        )
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


@tvm.testing.uses_gpu
def test_dyn_to_static_sparse_to_dense():
    def verify_sparse_to_dense(sparse_indices, sparse_values, default_value, output_shape, xpected):
        sparse_indices_data = np.array(sparse_indices)
        sparse_values_data = np.array(sparse_values)
        default_value_data = np.array(default_value)
        output_shape_data = np.array(output_shape)

        a = relay.var(
            "a", relay.TensorType(sparse_indices_data.shape, str(sparse_indices_data.dtype))
        )
        b = relay.var(
            "b", relay.TensorType(sparse_values_data.shape, str(sparse_values_data.dtype))
        )
        output_shape_const = relay.const(output_shape_data)

        if default_value is None:
            args = [a, b]
            d = relay.sparse_to_dense(a, output_shape_const, b)
        else:
            c = relay.var(
                "c", relay.TensorType(default_value_data.shape, str(default_value_data.dtype))
            )
            args = [a, b, c]
            d = relay.sparse_to_dense(a, output_shape_const, b, c)

        zz = run_infer_type(d)
        assert len(zz.checked_type.shape) == len(output_shape)

        func = relay.Function(args, d)

        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())
        assert isinstance(func2.body, relay.Call)
        assert func2.body.op == relay.op.get("sparse_to_dense")

        if default_value is None:
            arguments = [sparse_indices_data, sparse_values_data]
        else:
            arguments = [sparse_indices_data, sparse_values_data, default_value_data]

        verify_func(func2, arguments, xpected)

    verify_sparse_to_dense(1, 3, 0, [5], [0, 3, 0, 0, 0])  # scalar
    verify_sparse_to_dense([0, 1, 4], [3, 3, 3], 0, [5], [3, 3, 0, 0, 3])  # vector
    verify_sparse_to_dense(
        [[0, 0], [1, 2]], [1, 2], 0, [3, 4], [[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]
    )  # nXd
    verify_sparse_to_dense(
        [[0, 0, 0], [1, 2, 3]],
        [1, 2],
        4,
        [2, 3, 4],
        [[[1, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]], [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 2]]],
    )  # nXd
    verify_sparse_to_dense(
        [0, 1, 4], [3.1, 3.1, 3.1], 3.5, [5], [3.1, 3.1, 3.5, 3.5, 3.1]
    )  # floats
    verify_sparse_to_dense(1, 3, None, [5], [0, 3, 0, 0, 0])  # default value not specified


@tvm.testing.uses_gpu
def test_dynamic_to_static_dynamic_rank():
    def verify_full(fill_value, fill_shape, dtype):
        x = relay.var("x", relay.scalar_type(dtype))
        y = relay.var("y", relay.TensorType(fill_shape, "int64"))
        shape = relay.shape_of(y)
        shape = relay.strided_slice(shape, [0], relay.shape_of(shape))
        z = relay.full(x, shape, dtype)

        func = relay.Function([x, y], z)
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("full")

        ref_res = np.full(fill_shape, fill_value).astype(dtype)
        y_data = np.random.uniform(low=-1, high=1, size=fill_shape).astype("int64")
        verify_func(func2, [fill_value, y_data], ref_res)

    verify_full(4, (1, 2, 3, 4), "int32")
    verify_full(4.0, (1, 2, 8, 10), "float32")


@tvm.testing.uses_gpu
def test_dynamic_to_static_dynamic_if():
    x = relay.var("x", relay.TensorType((2, 2), "int64"))
    cond = relay.const(1)
    iff = relay.If(cond, relay.reshape(x, [1, 4]), relay.reshape(x, (4, 1)))

    func = relay.Function([x], iff)
    func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

    zz = func2.body
    assert isinstance(zz, relay.Call)
    assert zz.op == relay.op.get("reshape")
    x_data = np.random.uniform(low=-1, high=1, size=(2, 2)).astype("int64")
    verify_func(func2, [x_data], x_data.reshape(1, 4))


if __name__ == "__main__":
    tvm.testing.main()
