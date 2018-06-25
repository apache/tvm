import math
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
        np.testing.assert_allclose(out.asnumpy(), res, atol=1e-5, rtol=1e-5)

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

    target = "llvm"
    ctx = tvm.cpu()
    graph, lib, _ = nnvm.compiler.build(out, target, {"data": dshape})
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input("data", np.random.uniform(size=dshape).astype(dtype))
    m.run()
    out = m.get_output(0, tvm.nd.empty(np_out.shape, dtype))
    np.testing.assert_allclose(out.asnumpy(), np_out, atol=1e-5, rtol=1e-5)

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
    out = sym.nms(data=transform_loc_data, valid_count=valid_count)

    # Manually create test case
    np_cls_prob = np.array([[[0.2, 0.5, 0.3], [0.25, 0.3, 0.45], [0.7, 0.1, 0.2]]])
    np_loc_preds = np.array([[0.1, -0.2, 0.3, 0.2, 0.2, 0.4, 0.5, -0.3, 0.7, -0.2, -0.4, -0.8]])
    np_anchors = np.array([[[-0.1, -0.1, 0.1, 0.1], [-0.2, -0.2, 0.2, 0.2], [1.2, 1.2, 1.5, 1.5]]])

    expected_np_out = np.array([[[1, 0.69999999, 0, 0, 0.10818365, 0.10008108],
                                 [0, 0.44999999, 1, 1, 1, 1],
                                 [0, 0.30000001, 0, 0, 0.22903419, 0.20435292]]])

    target = "llvm"
    dtype = "float32"
    ctx = tvm.cpu()
    graph, lib, _ = nnvm.compiler.build(out, target, {"cls_prob": (batch_size, num_anchors, num_classes),
                                                      "loc_preds": (batch_size, num_anchors * 4),
                                                      "anchors": (1, num_anchors, 4)})
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**{"cls_prob": np_cls_prob.astype(dtype), "loc_preds": np_loc_preds.astype(dtype), "anchors": np_anchors.astype(dtype)})
    m.run()
    out = m.get_output(0, tvm.nd.empty(expected_np_out.shape, dtype))
    np.testing.assert_allclose(out.asnumpy(), expected_np_out, atol=1e-5, rtol=1e-5)

def test_nms():
    dshape = (1, 5, 6)
    data = sym.Variable("data")
    valid_count = sym.Variable("valid_count", dtype="int32")
    nms_threshold = 0.7
    force_suppress = True
    nms_topk = 2
    out = sym.nms(data=data, valid_count=valid_count, nms_threshold=nms_threshold,
                  force_suppress=force_suppress, nms_topk=nms_topk)

    np_data = np.array([[[0, 0.8, 1, 20, 25, 45], [1, 0.7, 30, 60, 50, 80],
                         [0, 0.4, 4, 21, 19, 40], [2, 0.9, 35, 61, 52, 79],
                         [1, 0.5, 100, 60, 70, 110]]]).astype("float32")
    np_valid_count = np.array([4]).astype("int32")
    np_result = np.array([[[2, 0.9, 35, 61, 52, 79], [0, 0.8, 1, 20, 25, 45],
                           [0, 0.4, 4, 21, 19, 40], [-1, 0.9, 35, 61, 52, 79],
                           [-1, -1, -1, -1, -1, -1]]])

    target = "llvm"
    ctx = tvm.cpu()
    graph, lib, _ = nnvm.compiler.build(out, target, {"data": dshape, "valid_count": (dshape[0],)},
                                        dtype={"data": "float32", "valid_count": "int32"})
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**{"data": np_data, "valid_count": np_valid_count})
    m.run()
    out = m.get_output(0, tvm.nd.empty(np_result.shape, "float32"))
    np.testing.assert_allclose(out.asnumpy(), np_result, atol=1e-5, rtol=1e-5)



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
    test_flip()
    test_multibox_prior()
    test_multibox_transform_loc()
    test_nms()
    print(nnvm.compiler.engine.dump())
