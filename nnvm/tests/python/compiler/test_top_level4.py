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
import math
import numpy as np
import tvm
from tvm.contrib import graph_runtime
import topi
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list
from nnvm.testing.check_computation import check_function

def verify_transpose(dshape, axes):
    x = sym.Variable("x")
    if axes:
        y = sym.transpose(x, axes=axes)
    else:
        y = sym.transpose(x)
    y = y + 1
    dtype = "float32"
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = graph_runtime.create(graph, lib, ctx)
        # set input
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        m.run(x=data)
        out_np = np.transpose(data.asnumpy(), axes=axes) + 1
        out = m.get_output(0, tvm.nd.empty(out_np.shape))
        tvm.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)

def verify_reduce_explicit(dshape, data, result, fsym, oshape=None, otype='float32', **kwargs):
    """ Verify reduce operations by comparign its result with `result` """
    x = sym.Variable("x")
    y = fsym(x + 0, **kwargs)
    for target, ctx in ctx_list():
        # TODO(yuruofei): remove when cuda reduce schedule is done
        if target == 'cuda' and fsym == sym.mean:
            continue
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = graph_runtime.create(graph, lib, ctx)
        # set input
        m.run(x=data)
        # oshape set to None means do not test the shape-correctness
        oshape = result.shape if isinstance(result, np.ndarray) else (1,) if oshape is None else oshape
        out = m.get_output(0, tvm.nd.empty(oshape, dtype=otype))
        if isinstance(result, np.ndarray):
            np.testing.assert_equal(out.asnumpy().shape, result.shape)
            tvm.testing.assert_allclose(out.asnumpy(), result, atol=1e-5, rtol=1e-5)
        else:
            tvm_out = out.asnumpy()
            assert abs(result - tvm_out) <= (1e-5 + 1e-5 * abs(tvm_out))

def verify_reduce(dshape, fnp, fsym, oshape=None, otype='float32', **kwargs):
    """ Verify reduce operations by generating data at random and calling numpy
    version as reference """
    data = np.random.uniform(size=dshape).astype(otype)
    result = fnp(data + 0, **kwargs)
    verify_reduce_explicit(dshape, data, result, fsym, oshape=oshape, otype=otype, **kwargs)

def verify_collapse(dshape, target_shape, fnp):
    x = sym.Variable("x", shape=dshape)
    t = sym.Variable("t", shape=target_shape)
    y = sym.collapse_sum(x, t)
    dtype = "float32"
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target,
                                            {"x": dshape, "t": target_shape})
        m = graph_runtime.create(graph, lib, ctx)
        data = np.random.uniform(size=dshape).astype(dtype)
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(target_shape))
        out_np = fnp(data)
        tvm.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)


def test_transpose():
    verify_transpose((2, 3, 4), (0, 2, 1))
    verify_transpose((2, 3, 4), None)


def test_reduce():

    def _with_keepdims(func):
        """ Wrapper around numpy's argmax/argmin with `keepdims` argument supported """
        def wrapper(data, axis=None, keepdims=False):
            if not keepdims:
                return func(data, axis=axis)
            else:
                if axis is not None:
                    out_shape = list(data.shape)
                    out_shape[axis] = 1
                else:
                    out_shape = [1 for _ in range(len(data.shape))]
                return func(data, axis=axis).reshape(out_shape)
        return wrapper

    verify_reduce((2, 3, 4), np.max, sym.max, axis=1, keepdims=True)
    verify_reduce((4, 4, 3), np.min, sym.min, keepdims=True)
    verify_reduce((4, 4, 3), np.sum, sym.sum, axis=(0, 2))
    verify_reduce((4, 4, 3), np.sum, sym.sum)
    verify_reduce((128, 24, 128), np.mean, sym.mean, axis=(0, 1), keepdims=False)
    verify_reduce((128, 24, 128), np.mean, sym.mean, axis=(0, 2), keepdims=False)
    verify_reduce((128, 24, 128), np.mean, sym.mean, axis=(0, 1), keepdims=True)
    verify_reduce((128, 24, 128), np.mean, sym.mean, axis=(0, 2), keepdims=True)
    verify_reduce((128, 24, 128), np.mean, sym.mean, keepdims=True)
    verify_reduce((128, 24, 128), np.mean, sym.mean, keepdims=False)
    verify_reduce((128, 24, 128), np.mean, sym.mean, axis=(0, 1, 2), keepdims=True)

    data = np.array([[[1,2],[3,4]],[[3,44],[5,6]]], dtype=np.float32)
    verify_reduce_explicit([2,2,2], data, np.array([[1,1],[1,0]]), sym.argmax, otype='int32', axis=[0,2], exclude=True)
    verify_reduce_explicit([2,2,2], data, np.array([[0,0],[0,1]]), sym.argmin, otype='int32', axis=[0,2], exclude=True)
    shape = [4, 4, 3]
    for axis in [None, 0, 1, 2]:
        for keepdims in [True,False]:
            kwargs = { 'keepdims':keepdims }
            if axis is None:
                # FIXME: NNVM doesn't support setting `axis=None` explicitly.
                kwargs.update({'oshape': [1,1,1] if keepdims else [1] })
            else:
                kwargs.update({'axis': axis})
                kwargs.update({'oshape': shape[:axis]+[1]+shape[axis+1:] if keepdims else shape[:axis]+shape[axis+1:]})

            verify_reduce(shape, _with_keepdims(np.argmax), sym.argmax, otype='int32', **kwargs)
            verify_reduce(shape, _with_keepdims(np.argmin), sym.argmin, otype='int32', **kwargs)


def test_collapse():
    verify_collapse((2, 3, 4), (1,), lambda x: x.sum())
    verify_collapse((2, 3, 4), (1, 1, 1), lambda x: x.sum(keepdims=True))
    verify_collapse((2, 3, 4), (1, 1), lambda x: x.sum().reshape(1, 1))
    verify_collapse((2, 3, 4), (1, 4), lambda x: x.reshape(-1, 4).sum(0, keepdims=True))
    verify_collapse((2, 3, 4), (3, 4), lambda x: x.sum(0))
    verify_collapse((2, 3, 4), (1, 3, 4), lambda x: x.sum(0, keepdims=True))
    verify_collapse((2, 3, 4), (1, 1, 4), lambda x: x.sum((0, 1), keepdims=True))
    verify_collapse((2, 3, 4), (2, 1, 4), lambda x: x.sum(1, keepdims=True))
    verify_collapse((2, 3, 4), (2, 1, 1), lambda x: x.sum((1, 2), keepdims=True))
    verify_collapse((2, 3, 4), (2, 3, 1), lambda x: x.sum(2, keepdims=True))
    verify_collapse((2, 3, 4), (2, 3, 4), lambda x: x)


def verify_flip(ishape, axis):
    x = sym.Variable("x")
    y = sym.flip(x, axis=axis) + 1
    dtype = "float32"
    x_np = np.random.uniform(size=ishape).astype(dtype)
    res = np.flip(x_np, axis) + 1

    for target, ctx in ctx_list():
        # set input
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": ishape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(x=x_np)
        out = m.get_output(0, tvm.nd.empty(res.shape))
        tvm.testing.assert_allclose(out.asnumpy(), res, atol=1e-5, rtol=1e-5)


def test_flip():
    verify_flip((3, 4, 3), 1)
    verify_flip((3, 4, 3), 0)
    verify_flip((3, 4, 3), 2)
    verify_flip((3, 4, 3), -1)
    verify_flip((3, 4, 3), -3)
    verify_flip((3, 4, 3), -2)


def verify_reshape(dshape, oshape):
    x = sym.Variable("x")
    y = sym.reshape(x, shape=oshape)
    y = y + 1
    dtype = "float32"
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = graph_runtime.create(graph, lib, ctx)
        # set input
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        m.run(x=data)
        out_np = data.asnumpy().reshape(oshape) + 1
        out = m.get_output(0, tvm.nd.empty(out_np.shape))
        tvm.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)


def test_reshape():
    verify_reshape((2, 3, 4), (-1, 2, 1))
    verify_reshape((2, 3, 4), (8, 3))
    verify_reshape((4, 7), (2, 7, 2))


def test_clip():
    x = sym.Variable("x")
    a_min=0.2
    a_max=0.75
    y = sym.clip(x, a_min=a_min, a_max=a_max)

    def forward(x):
        return np.clip(x, a_min=a_min, a_max=a_max)

    def backward(head_grads, x):
        mask1 = np.greater_equal(x, a_min).astype("float")
        mask2 = np.less_equal(x, a_max).astype("float")
        return [head_grads * mask1 * mask2]

    shape = {'x': (3, 4, 5)}
    check_function(y, forward, backward, shape=shape)


def test_broadcast():
    a = sym.Variable("a")
    b = sym.Variable("b")
    shape = {'a': (3, 4, 5), 'b': (1, 5)}

    def _collapse(g):
        return g.reshape(-1, shape['b'][-1]).sum(0, keepdims=True)

    y = sym.broadcast_add(a, b)
    def _backward_add(head_grads, a, b):
        da = head_grads
        db = _collapse(head_grads)
        return da, db
    check_function(y, lambda a, b: a + b, _backward_add, shape=shape)

    y = sym.broadcast_sub(a, b)
    def _backward_sub(head_grads, a, b):
        da = head_grads
        db = -_collapse(head_grads)
        return da, db
    check_function(y, lambda a, b: a - b, _backward_sub, shape=shape)

    y = sym.broadcast_mul(a, b)
    def _backward_mul(head_grads, a, b):
        da = head_grads * b
        db = _collapse(head_grads * a)
        return da, db
    check_function(y, lambda a, b: a * b, _backward_mul, shape=shape)

    y = sym.broadcast_div(a, b)
    def _backward_div(head_grads, a, b):
        da = head_grads / b
        db = _collapse(- head_grads * a / b**2)
        return da, db
    # We avoid computing numerical derivatives too close to zero here
    check_function(y, lambda a, b: a / b, _backward_div, shape=shape, numerical_grads=False)
    check_function(y, lambda a, b: a / b, _backward_div, shape=shape,
                   in_range={'b': (0.1, 20)})

    y = sym.broadcast_mod(a, b)
    check_function(y,
                   lambda a, b: np.mod(a, b),
                   in_range={'a': (0.001, 100), 'b': (1, 100)}, dtype='int32', shape=shape)

    y = sym.broadcast_max(a, b)
    check_function(y, lambda a, b: np.maximum(a, b), shape=shape)

    y = sym.broadcast_min(a, b)
    check_function(y, lambda a, b: np.minimum(a, b), shape=shape)

    y = sym.broadcast_pow(a, b)
    check_function(y,
                   lambda a, b: np.power(a, b),
                   in_range={'a': (0.001, 100), 'b': (0.001, 2)}, shape=shape)

    y = sym.broadcast_left_shift(a, b)
    check_function(y, lambda a, b: a << b, dtype='int32', shape=shape)

    y = sym.broadcast_right_shift(a, b)
    check_function(y, lambda a, b: a >> b, dtype='int32', shape=shape)

    y = sym.broadcast_greater(a, b)
    check_function(y, lambda a, b: np.greater(a, b), shape=shape)

    y = sym.broadcast_less(a, b)
    check_function(y, lambda a, b: np.less(a, b), shape=shape)

    y = sym.broadcast_equal(a, b)
    check_function(y, lambda a, b: np.equal(a, b),
                   in_range={'a': (-2, 2), 'b': (-2, 2)}, dtype='int32', shape=shape)

    y = sym.broadcast_not_equal(a, b)
    check_function(y, lambda a, b: np.not_equal(a, b),
                   in_range={'a': (-2, 2), 'b': (-2, 2)}, dtype='int32', shape=shape)

    y = sym.broadcast_greater_equal(a, b)
    check_function(y, lambda a, b: np.greater_equal(a, b),
                   in_range={'a': (-3, 3), 'b': (-3, 3)}, dtype='int32', shape=shape)

    y = sym.broadcast_less_equal(a, b)
    check_function(y, lambda a, b: np.less_equal(a, b),
                   in_range={'a': (-3, 3), 'b': (-3, 3)}, dtype='int32', shape=shape)

def test_greater():
    l = sym.Variable("l")
    r = sym.Variable("r")
    y = sym.greater(l, r)

    def forward(l, r):
        return np.greater(l, r).astype("float32")

    def backward(head_grads, l, r):
        return {'l': np.zeros_like(l)}

    shape = {'l': (3, 4, 5), 'r': (3, 4, 5)}
    check_function(y, forward, backward, shape=shape)


def test_less():
    l = sym.Variable("l")
    r = sym.Variable("r")
    y = sym.less(l, r)

    def forward(l, r):
        return np.less(l, r).astype("float32")

    def backward(head_grads, l, r):
        return {'l': np.zeros_like(l)}

    shape = {'l': (3, 4, 5), 'r': (3, 4, 5)}
    check_function(y, forward, backward, shape=shape)


def test_reshape_like():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.reshape_like(x, y)

    def forward(x, y):
        return np.reshape(x, y.shape)

    def backward(head_grads, x, y):
        return [np.reshape(head_grads, x.shape),
                np.zeros_like(y)]

    shape = {'x': (3, 4, 5), 'y': (5, 4, 3)}
    check_function(z, forward, backward, shape=shape)


def verify_expand_like(in_shape, out_shape, axis, exclude):
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.expand_like(x, y, axis=axis, exclude=exclude)

    def forward(x, y):
        odim = len(out_shape)

        if len(x.shape) == len(y.shape):
            return np.broadcast_to(x, y.shape)

        if x.shape == (1,) and len(y.shape) == odim:
            x = np.reshape(x, ())

        real_axis = [i if i >= 0 else i + odim for i in axis]
        real_axis = sorted(real_axis)
        if exclude:
            real_axis = list(set(range(odim)) - set(real_axis))
        for i in real_axis:
            x = np.expand_dims(x, i).astype(x.dtype)
        for i in real_axis:
            x = np.concatenate([x]*out_shape[i], axis=i).astype(x.dtype)

        return x

    def backward(head_grads, x, y):
        odim = len(out_shape)

        keepdims = len(x.shape) == len(y.shape)

        if x.shape == (1,) and len(y.shape) == odim:
            x = np.reshape(x, ())

        real_axis = [i if i >= 0 else i + odim for i in axis]
        real_axis = sorted(real_axis)
        if exclude:
            real_axis = list(set(range(odim)) - set(real_axis))
        return [np.sum(head_grads, axis=tuple(real_axis), keepdims=keepdims),
                np.zeros_like(y)]


    shape = {'x': in_shape, 'y': out_shape}
    check_function(z, forward, backward, shape=shape)


def test_expand_like():
    verify_expand_like((3,), (3, 2), [1], False)
    verify_expand_like((2,), (2, 3), [1], False)
    verify_expand_like((3, 4), (3, 5, 4), [1], False)
    verify_expand_like((5, 7), (5, 6, 7, 8), [0, 2], True)
    verify_expand_like((2, 3), (2, 3), [], False)
    verify_expand_like((1,), (2, 3), [0, 1], False)
    verify_expand_like((1, 1), (2, 3), [0, 1], False)
    verify_expand_like((2, 1), (2, 3), [1], False)
    verify_expand_like((1, 3), (2, 3), [0], False)


def verify_elemwise_sum(num_args):
    s = [sym.Variable("input" + str(i)) for i in range(num_args)]
    y = sym.elemwise_sum(*s, num_args=num_args)

    def forward(**inputs):
        return np.sum(np.array(list(inputs.values())), axis=0)

    def backward(head_grads, **inputs):
        return [head_grads] * num_args

    shape = {s[i]: (3, 4, 5) for i in range(num_args)}
    check_function(y, forward, backward, shape=shape)


def test_elemwise_sum():
    verify_elemwise_sum(1)
    verify_elemwise_sum(5)
    verify_elemwise_sum(7)


def test_block_grad():
    x = sym.Variable("x")
    y = sym.block_grad(x)

    def forward(x):
        return x

    def backward(head_grads, x):
        return [np.zeros_like(head_grads)]


    shape = {'x': (3, 4, 5)}
    # Numerical grad checking would fail for this function
    check_function(y, forward, backward, shape=shape, numerical_grads=False)


def test_full():
    shape = (3, 4, 5)
    value = 7
    dtype = "float32"
    for target, ctx in ctx_list():
        data = sym.Variable("data", dtype=dtype)
        # full_like
        s = sym.full_like(data=data, fill_value=value, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target, {"data": shape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(data=np.random.uniform(size=shape).astype(dtype))
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        tvm.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=value, dtype=dtype),
            atol=1e-5, rtol=1e-5)
        # ones_like
        s = sym.ones_like(data=data, fill_value=value, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target, {"data": shape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(data=np.random.uniform(size=shape).astype(dtype))
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        tvm.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=1, dtype=dtype),
            atol=1e-5, rtol=1e-5)
        # zeros_like
        s = sym.zeros_like(data=data, fill_value=value, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target, {"data": shape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(data=np.random.uniform(size=shape).astype(dtype))
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        tvm.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=0, dtype=dtype),
            atol=1e-5, rtol=1e-5)
        # full
        s = sym.full(shape=shape, dtype=dtype, fill_value=value, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target)
        m = graph_runtime.create(graph, lib, ctx)
        m.run()
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        tvm.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=value, dtype=dtype),
            atol=1e-5, rtol=1e-5)
        # ones
        s = sym.ones(shape=shape, dtype=dtype, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target)
        m = graph_runtime.create(graph, lib, ctx)
        m.run()
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        tvm.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=1, dtype=dtype),
            atol=1e-5, rtol=1e-5)
        # zeros
        s = sym.zeros(shape=shape, dtype=dtype, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target)
        m = graph_runtime.create(graph, lib, ctx)
        m.run()
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        tvm.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=0, dtype=dtype),
            atol=1e-5, rtol=1e-5)

def verify_multibox_prior(dshape, sizes=(1,), ratios=(1,), steps=(-1, -1),
                          offsets=(0.5, 0.5), clip=False):
    data = sym.Variable("data")
    out = sym.multibox_prior(data=data, sizes=sizes, ratios=ratios, steps=steps,
                             offsets=offsets, clip=clip)

    in_height = dshape[2]
    in_width = dshape[3]
    num_sizes = len(sizes)
    num_ratios = len(ratios)
    size_ratio_concat = sizes + ratios
    steps_h = steps[0] if steps[0] > 0 else 1.0 / in_height
    steps_w = steps[1] if steps[1] > 0 else 1.0 / in_width
    offset_h = offsets[0]
    offset_w = offsets[1]

    oshape = (1, in_height * in_width * (num_sizes + num_ratios - 1), 4)
    dtype = "float32"
    np_out = np.zeros(oshape).astype(dtype)

    for i in range(in_height):
        center_h = (i + offset_h) * steps_h
        for j in range(in_width):
            center_w = (j + offset_w) * steps_w
            for k in range(num_sizes + num_ratios - 1):
                w = size_ratio_concat[k] * in_height / in_width / 2.0 if k < num_sizes else \
                    size_ratio_concat[0] * in_height / in_width * math.sqrt(size_ratio_concat[k + 1]) / 2.0
                h = size_ratio_concat[k] / 2.0 if k < num_sizes else \
                    size_ratio_concat[0] / math.sqrt(size_ratio_concat[k + 1]) / 2.0
                count = i * in_width * (num_sizes + num_ratios - 1) + j * (num_sizes + num_ratios - 1) + k
                np_out[0][count][0] = center_w - w
                np_out[0][count][1] = center_h - h
                np_out[0][count][2] = center_w + w
                np_out[0][count][3] = center_h + h
    if clip:
        np_out = np.clip(np_out, 0, 1)

    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(out, target, {"data": dshape})
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input("data", np.random.uniform(size=dshape).astype(dtype))
        m.run()
        tvm_out = m.get_output(0, tvm.nd.empty(np_out.shape, dtype))
        tvm.testing.assert_allclose(tvm_out.asnumpy(), np_out, atol=1e-5, rtol=1e-5)

def test_multibox_prior():
    verify_multibox_prior((1, 3, 50, 50))
    verify_multibox_prior((1, 3, 224, 224), sizes=(0.5, 0.25, 0.1), ratios=(1, 2, 0.5))
    verify_multibox_prior((1, 32, 32, 32), sizes=(0.5, 0.25), ratios=(1, 2), steps=(2, 2), clip=True)

def test_multibox_transform_loc():
    batch_size = 1
    num_anchors = 3
    num_classes = 3
    cls_prob = sym.Variable("cls_prob")
    loc_preds = sym.Variable("loc_preds")
    anchors = sym.Variable("anchors")
    transform_loc_data, valid_count = sym.multibox_transform_loc(cls_prob=cls_prob, loc_pred=loc_preds,
                                                                 anchor=anchors)
    out = sym.non_max_suppression(data=transform_loc_data, valid_count=valid_count, return_indices=False)

    # Manually create test case
    np_cls_prob = np.array([[[0.2, 0.5, 0.3], [0.25, 0.3, 0.45], [0.7, 0.1, 0.2]]])
    np_loc_preds = np.array([[0.1, -0.2, 0.3, 0.2, 0.2, 0.4, 0.5, -0.3, 0.7, -0.2, -0.4, -0.8]])
    np_anchors = np.array([[[-0.1, -0.1, 0.1, 0.1], [-0.2, -0.2, 0.2, 0.2], [1.2, 1.2, 1.5, 1.5]]])

    expected_np_out = np.array([[[1, 0.69999999, 0, 0, 0.10818365, 0.10008108],
                                 [0, 0.44999999, 1, 1, 1, 1],
                                 [0, 0.30000001, 0, 0, 0.22903419, 0.20435292]]])

    dtype = "float32"
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(out, target, {"cls_prob": (batch_size, num_anchors, num_classes),
                                                          "loc_preds": (batch_size, num_anchors * 4),
                                                          "anchors": (1, num_anchors, 4)})
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input(**{"cls_prob": np_cls_prob.astype(dtype), "loc_preds": np_loc_preds.astype(dtype), "anchors": np_anchors.astype(dtype)})
        m.run()
        tvm_out = m.get_output(0, tvm.nd.empty(expected_np_out.shape, dtype))
        tvm.testing.assert_allclose(tvm_out.asnumpy(), expected_np_out, atol=1e-5, rtol=1e-5)

def test_non_max_suppression():
    dshape = (1, 5, 6)
    data = sym.Variable("data")
    valid_count = sym.Variable("valid_count", dtype="int32")
    iou_threshold = 0.7
    force_suppress = True
    top_k = 2
    out = sym.non_max_suppression(data=data, valid_count=valid_count, return_indices=False,
                                  iou_threshold=iou_threshold, force_suppress=force_suppress, top_k=top_k)

    np_data = np.array([[[0, 0.8, 1, 20, 25, 45], [1, 0.7, 30, 60, 50, 80],
                         [0, 0.4, 4, 21, 19, 40], [2, 0.9, 35, 61, 52, 79],
                         [1, 0.5, 100, 60, 70, 110]]]).astype("float32")
    np_valid_count = np.array([4]).astype("int32")
    np_result = np.array([[[2, 0.9, 35, 61, 52, 79], [0, 0.8, 1, 20, 25, 45],
                           [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1, -1]]])

    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(out, target, {"data": dshape, "valid_count": (dshape[0],)},
                                            dtype={"data": "float32", "valid_count": "int32"})
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input(**{"data": np_data, "valid_count": np_valid_count})
        m.run()
        tvm_out = m.get_output(0, tvm.nd.empty(np_result.shape, "float32"))
        tvm.testing.assert_allclose(tvm_out.asnumpy(), np_result, atol=1e-5, rtol=1e-5)

def np_slice_like(np_data, np_shape_like, axis=[]):
    begin_idx = [0 for _ in np_data.shape]
    end_idx = list(np_data.shape)
    if len(axis) > 0:
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
    np_result = np_data[slice_idx]
    return np_result

def verify_slice_like(np_data, np_shape_like, axis=[]):
    dtype = "float32"
    np_data = np_data.astype(dtype)
    np_shape_like = np_shape_like.astype(dtype)
    np_result = np_slice_like(np_data, np_shape_like, axis)
    data1 = sym.Variable("data1")
    data2 = sym.Variable("data2")
    net = sym.slice_like(data=data1, slice_like=data2, axis=axis)
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(net, target, {"data1": np_data.shape,
                                                          "data2": np_shape_like.shape})
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input(**{"data1": np_data, "data2": np_shape_like})
        m.run()
        out = m.get_output(0, tvm.nd.empty(np_result.shape, dtype))
        tvm.testing.assert_allclose(out.asnumpy(), np_result, atol=1e-5, rtol=1e-5)

def test_slice_like():
    np_data = np.random.uniform(size=(3, 4, 5))
    np_shape_like = np.random.uniform(size=(1, 2, 3))
    verify_slice_like(np_data, np_shape_like)
    np_data = np.random.uniform(size=(3, 4, 5))
    np_shape_like = np.random.uniform(size=(1, 2))
    verify_slice_like(np_data, np_shape_like)
    np_data = np.random.uniform(size=(3, 4, 5))
    np_shape_like = np.random.uniform(size=(1, 2, 3))
    axis = (1, 2)
    verify_slice_like(np_data, np_shape_like, axis)
    np_data = np.random.uniform(size=(3, 4, 5))
    np_shape_like = np.random.uniform(size=(1, 2, 3))
    axis = (-1, -3)
    verify_slice_like(np_data, np_shape_like, axis)
    np_data = np.random.uniform(size=(1, 3, 224, 224))
    np_shape_like = np.random.uniform(size=(1, 3, 112, 112))
    axis = (2, 3)
    verify_slice_like(np_data, np_shape_like, axis)

def verify_where(condition, x, y):
    dtype = "float32"
    if len(condition.shape) == 1:
        np_out = np.array([xv if c else yv for (c,xv,yv) in zip(condition,x,y)])
    else:
        np_out = np.where(condition, x, y)
    cond_var = sym.Variable("condition")
    x_var = sym.Variable("x")
    y_var = sym.Variable("y")
    net = sym.where(cond_var, x_var, y_var)
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(net, target, {"condition": condition.shape,
                                                          "x": x.shape, "y": y.shape})
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input(**{"condition": condition, "x": x, "y": y})
        m.run()
        out = m.get_output(0, tvm.nd.empty(x.shape, dtype))
        tvm.testing.assert_allclose(out.asnumpy(), np_out, atol=1e-5, rtol=1e-5)

def test_where():
    shape = (13, 8, 224, 224, 6)
    condition = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
    x = np.random.uniform(size=shape).astype("float32")
    y = np.random.uniform(size=shape).astype("float32")
    verify_where(condition, x, y)
    condition = np.random.uniform(low=-1, high=1, size=(shape[0],)).astype("float32")
    x = np.random.uniform(size=shape).astype("float32")
    y = np.random.uniform(size=shape).astype("float32")
    verify_where(condition, x, y)

def test_argmax():
    dshape = (204800, 2)
    oshape = (1, 320, 640)

    dtype = "float32"
    x = sym.Variable("x", shape=dshape, dtype=dtype)
    x = sym.reshape(x, shape=(1, 320, 640, 2))
    x = sym.transpose(x, axes=(0, 3, 1, 2))
    y = sym.argmax(x, axis=1)
    target_str = "llvm"
    target = tvm.target.create(target_str)
    ctx = tvm.context(target_str, 0)
    with nnvm.compiler.build_config(opt_level=2):
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
    m = graph_runtime.create(graph, lib, ctx)
    data = np.random.uniform(size=dshape).astype(dtype)
    m.run(x=data)
    np_reshape = np.reshape(data, (1, 320, 640, 2))
    np_transpose = np.transpose(np_reshape, axes=(0, 3, 1, 2))
    np_argmax = np.argmax(np_transpose, axis=1)
    out = m.get_output(0)
    np.testing.assert_allclose(out.asnumpy(), np_argmax, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    test_reshape()
    test_broadcast()
    test_reduce()
    test_collapse()
    test_transpose()
    test_clip()
    test_greater()
    test_less()
    test_reshape_like()
    test_expand_like()
    test_elemwise_sum()
    test_block_grad()
    test_full()
    test_flip()
    test_multibox_prior()
    test_multibox_transform_loc()
    test_non_max_suppression()
    test_slice_like()
    test_where()
    test_argmax()
    print(nnvm.compiler.engine.dump())
