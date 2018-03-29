import numpy as np
import tvm
from tvm.contrib import graph_runtime
import topi
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list


def helper(symbol, inputs, dtype,
           np_forward, np_backward=None, need_input=True, need_head_grads=True):
    ishapes = {}
    input_syms = []
    np_inputs = {}
    for (name, shape, s) in inputs:
        ishapes.update({name: shape})
        np_inputs.update({name: np.random.uniform(size=shape).astype(dtype)})
        input_syms.append(s)

    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(symbol, target, ishapes)
        m = graph_runtime.create(graph, lib, ctx)
        m.run(**np_inputs)
        y_np = np_forward(**np_inputs)
        out = m.get_output(0, tvm.nd.empty(y_np.shape, dtype))
        np.testing.assert_allclose(out.asnumpy(), y_np, atol=1e-5, rtol=1e-5)
        # backward
        if np_backward:
            graph._set_symbol_list_attr("grad_ys", symbol)
            graph._set_symbol_list_attr("grad_xs", input_syms)
            graph._set_symbol_list_attr("grad_ys_out_grad", sym.Variable("head_grads", shape=y_np.shape))
            graph = graph.apply("Gradient")
            ishapes.update({"head_grads": y_np.shape})
            graph, lib, _ = nnvm.compiler.build(graph, target, ishapes)
            m = graph_runtime.create(graph, lib, ctx)
            head_grads = np.random.uniform(size=y_np.shape).astype(dtype)
            y_np = np_backward(head_grads=head_grads, **np_inputs)
            b_inputs = {}
            if need_input:
                b_inputs.update(np_inputs)
            if need_head_grads:
                b_inputs.update({"head_grads":head_grads})
            m.run(**b_inputs)
            for i in range(len(y_np)):
                out = m.get_output(i, tvm.nd.empty(y_np[i].shape, dtype))
                np.testing.assert_allclose(out.asnumpy(), y_np[i], atol=1e-5, rtol=1e-5)


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
        np.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)


def verify_reduce(dshape, fnp, fsym, **kwargs):
    x = sym.Variable("x")
    y = fsym(x + 1, **kwargs)
    dtype = "float32"
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = graph_runtime.create(graph, lib, ctx)
        # set input
        data = np.random.uniform(size=dshape).astype(dtype)
        out_np = fnp(data + 1, **kwargs)
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(out_np.shape))
        np.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)


def test_tranpose():
    verify_transpose((2, 3, 4), (0, 2, 1))
    verify_transpose((2, 3, 4), None)


def test_reduce():
    verify_reduce((2, 3, 4), np.max, sym.max, axis=1, keepdims=True)
    verify_reduce((4, 4, 3), np.min, sym.min, keepdims=True)
    verify_reduce((4, 4, 3), np.sum, sym.sum, axis=(0, 2))


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
        np.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)


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


    dtype = "float32"
    inputs = [('x', (3, 4, 5), x)]
    helper(y, inputs, dtype, forward, backward)


def test_greater():
    l = sym.Variable("l")
    r = sym.Variable("r")
    y = sym.greater(l, r)

    def forward(l, r):
        return np.greater(l, r).astype("float32")

    def backward(head_grads, l, r):
        return [np.zeros_like(l)]


    dtype = "float32"
    inputs = [('l', (3, 4, 5), l),
              ('r', (3, 4, 5), r)]
    helper(y, inputs, dtype, forward, backward, need_head_grads=False)


def test_less():
    l = sym.Variable("l")
    r = sym.Variable("r")
    y = sym.less(l, r)

    def forward(l, r):
        return np.less(l, r).astype("float32")

    def backward(head_grads, l, r):
        return [np.zeros_like(l)]


    dtype = "float32"
    inputs = [('l', (3, 4, 5), l),
              ('r', (3, 4, 5), r)]
    helper(y, inputs, dtype, forward, backward, need_head_grads=False)


def test_reshape_like():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.reshape_like(x, y)

    def forward(x, y):
        return np.reshape(x, y.shape)

    def backward(head_grads, x, y):
        return [np.reshape(head_grads, x.shape),
                np.zeros_like(y)]


    dtype = "float32"
    inputs = [('x', (3, 4, 5), x),
              ('y', (5, 4, 3), y)]
    helper(z, inputs, dtype, forward, backward)


def verify_expand_like(in_shape, out_shape, axis, exclude):
    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.expand_like(x, y, axis=axis, exclude=exclude)

    def forward(x, y):
        odim = len(out_shape)
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
        real_axis = [i if i >= 0 else i + odim for i in axis]
        real_axis = sorted(real_axis)
        if exclude:
            real_axis = list(set(range(odim)) - set(real_axis))
        return [np.sum(head_grads, axis=tuple(real_axis)),
                np.zeros_like(y)]


    dtype = "float32"
    inputs = [('x', in_shape, x),
              ('y', out_shape, y)]
    helper(z, inputs, dtype, forward, backward, need_input=False)


def test_expand_like():
    verify_expand_like((3,), (3, 2), [1], False)
    verify_expand_like((2,), (2, 3), [1], False)
    verify_expand_like((3, 4), (3, 5, 4), [1], False)
    verify_expand_like((5, 7), (5, 6, 7, 8), [0, 2], True)


def verify_elemwise_sum(num_args):
    s = [sym.Variable("input" + str(i)) for i in range(num_args)]
    y = sym.elemwise_sum(*s, num_args=num_args)

    def forward(**inputs):
        return np.sum(np.array(list(inputs.values())), axis=0)

    def backward(head_grads, **inputs):
        return [head_grads] * num_args

    dtype = "float32"
    inputs = [("input" + str(i), (3, 4, 5), s[i])
              for i in range(num_args)]
    helper(y, inputs, dtype, forward, backward, need_input=False)


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


    dtype = "float32"
    inputs = [('x', (3, 4, 5), x)]
    helper(y, inputs, dtype, forward, backward, need_head_grads=False)


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
        np.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=value, dtype=dtype),
            atol=1e-5, rtol=1e-5)
        # ones_like
        s = sym.ones_like(data=data, fill_value=value, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target, {"data": shape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(data=np.random.uniform(size=shape).astype(dtype))
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        np.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=1, dtype=dtype),
            atol=1e-5, rtol=1e-5)
        # zeros_like
        s = sym.zeros_like(data=data, fill_value=value, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target, {"data": shape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(data=np.random.uniform(size=shape).astype(dtype))
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        np.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=0, dtype=dtype),
            atol=1e-5, rtol=1e-5)
        # full
        s = sym.full(shape=shape, dtype=dtype, fill_value=value, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target)
        m = graph_runtime.create(graph, lib, ctx)
        m.run()
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        np.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=value, dtype=dtype),
            atol=1e-5, rtol=1e-5)
        # ones
        s = sym.ones(shape=shape, dtype=dtype, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target)
        m = graph_runtime.create(graph, lib, ctx)
        m.run()
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        np.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=1, dtype=dtype),
            atol=1e-5, rtol=1e-5)
        # zeros
        s = sym.zeros(shape=shape, dtype=dtype, name="s")
        graph, lib, _ = nnvm.compiler.build(s, target)
        m = graph_runtime.create(graph, lib, ctx)
        m.run()
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype))
        np.testing.assert_allclose(
            out.asnumpy(),
            np.full(shape, fill_value=0, dtype=dtype),
            atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_reshape()
    test_reduce()
    test_tranpose()
    test_clip()
    test_greater()
    test_less()
    test_reshape_like()
    test_expand_like()
    test_elemwise_sum()
    test_block_grad()
    test_full()
    print(nnvm.compiler.engine.dump())
