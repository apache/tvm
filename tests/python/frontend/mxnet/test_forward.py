import numpy as np

import tvm
from tvm.contrib import graph_runtime
from tvm.relay.testing.config import ctx_list
from tvm import relay
import mxnet as mx

from mxnet import gluon
from mxnet.gluon.model_zoo import vision
import model_zoo


def verify_mxnet_frontend_impl(mx_symbol,
                               data_shape=(1, 3, 224, 224),
                               out_shape=(1, 1000),
                               gluon_impl=False,
                               name=None,
                               dtype='float32'):
    """Use name different from test to avoid let nose pick it up"""
    if gluon_impl:
        def get_gluon_output(name, x):
            net = vision.get_model(name)
            net.collect_params().initialize(mx.init.Xavier())
            net_sym = gluon.nn.SymbolBlock(outputs=net(mx.sym.var('data')),
                                           inputs=mx.sym.var('data'),
                                           params=net.collect_params())
            out = net_sym(mx.nd.array(x.astype(dtype))).asnumpy()
            return out, net_sym
    else:
        def get_mxnet_output(symbol, x, dtype='float32'):
            from collections import namedtuple
            Batch = namedtuple('Batch', ['data'])
            mod = mx.mod.Module(symbol, label_names=None)
            mod.bind(data_shapes=[('data', x.shape)], for_training=False)
            mod.init_params()
            mod.forward(Batch([mx.nd.array(x.astype(dtype))]))
            out = mod.get_outputs()[0].asnumpy()
            args, auxs = mod.get_params()
            return out, args, auxs

    def get_tvm_output(symbol, x, args, auxs, target, ctx, dtype='float32'):
        shape_dict = {"data": x.shape}
        if gluon_impl:
            new_sym, params = relay.frontend.from_mxnet(symbol, shape_dict)
        else:
            new_sym, params = relay.frontend.from_mxnet(symbol,
                                                        shape_dict,
                                                        arg_params=args,
                                                        aux_params=auxs)
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(new_sym, target, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input("data", tvm.nd.array(x.astype(dtype)))
        m.set_input(**params)
        m.run()
        # get outputs
        out = m.get_output(0, tvm.nd.empty(out_shape, dtype))
        return out.asnumpy()

    # random input
    x = np.random.uniform(size=data_shape)
    if gluon_impl:
        gluon_out, gluon_sym = get_gluon_output(name, x)
        for target, ctx in ctx_list():
            tvm_out = get_tvm_output(gluon_sym, x, None, None, target, ctx, dtype)
            tvm.testing.assert_allclose(gluon_out, tvm_out, rtol=1e-5, atol=1e-5)
    else:
        mx_out, args, auxs = get_mxnet_output(mx_symbol, x, dtype)
        assert "data" not in args
        for target, ctx in ctx_list():
            tvm_out = get_tvm_output(mx_symbol, x, args, auxs, target, ctx, dtype)
            tvm.testing.assert_allclose(mx_out, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_mlp():
    mlp = model_zoo.mx_mlp()
    verify_mxnet_frontend_impl(mlp,
                               data_shape=(1, 1, 28, 28),
                               out_shape=(1, 10))

def test_forward_vgg():
    for n in [11]:
        mx_sym = model_zoo.mx_vgg(n)
        verify_mxnet_frontend_impl(mx_sym)

def test_forward_resnet():
    for n in [18]:
        mx_sym = model_zoo.mx_resnet(18)
        verify_mxnet_frontend_impl(mx_sym)

def test_forward_elu():
    data = mx.sym.var('data')
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.LeakyReLU(data, act_type='elu')
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))

def test_forward_rrelu():
    data = mx.sym.var('data')
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.LeakyReLU(data, act_type='rrelu', lower_bound=0.3, upper_bound=0.7)
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))

def test_forward_prelu():
    data = mx.sym.var('data')
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.LeakyReLU(data, act_type='prelu')
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))

def test_forward_softrelu():
    data = mx.sym.var('data')
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.Activation(data, act_type='softrelu')
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))

def test_forward_fc_flatten():
    # test flatten=True option in mxnet 0.11.1
    data = mx.sym.var('data')
    try:
        mx_sym = mx.sym.FullyConnected(data, num_hidden=100, flatten=True)
        verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 100))
        mx_sym = mx.sym.FullyConnected(mx.sym.Flatten(data), num_hidden=100, flatten=False)
        verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 100))
    except:
        pass

def test_forward_clip():
    data = mx.sym.var('data')
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicity
    mx_sym = mx.sym.clip(data, a_min=0, a_max=1)
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))

def test_forward_split():
    data = mx.sym.var('data')
    mx_sym = mx.sym.split(data, axis=1, num_outputs=4, squeeze_axis=False)
    verify_mxnet_frontend_impl(mx_sym, (1, 4, 2, 1), (1, 1, 2, 1))

def test_forward_split_squeeze():
    data = mx.sym.var('data')
    mx_sym = mx.sym.split(data, axis=1, num_outputs=4, squeeze_axis=True)
    verify_mxnet_frontend_impl(mx_sym, (1, 4, 2, 1), (1, 2, 1))

def test_forward_expand_dims():
    data = mx.sym.var('data')
    mx_sym = mx.sym.expand_dims(data, axis=1)
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 1, 3, 4))

def test_forward_pooling():
    data = mx.sym.var('data')
    mx_sym = mx.sym.Pooling(data, kernel=(3, 3), pad=(1, 1), pool_type='avg')
    verify_mxnet_frontend_impl(mx_sym, (1, 20, 8, 8), (1, 20, 8, 8))

    mx_sym = mx.sym.Pooling(data, kernel=(3, 3), pad=(1, 1), pool_type='max')
    verify_mxnet_frontend_impl(mx_sym, (1, 20, 8, 8), (1, 20, 8, 8))

def test_forward_lrn():
    data = mx.sym.var('data')
    mx_sym = mx.sym.LRN(data, alpha=2, beta=2, knorm=1, nsize=5)
    verify_mxnet_frontend_impl(mx_sym, (1, 10, 24, 24), (1, 10, 24, 24))

def test_forward_ones():
    data = mx.sym.var('data')
    ones = mx.sym.ones(shape=(2, 3, 4), dtype='float32')
    mx_sym = mx.sym.elemwise_add(data, ones)
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 3, 4))

def test_forward_zeros():
    data = mx.sym.var('data')
    zeros = mx.sym.zeros(shape=(2, 3, 4), dtype='float32')
    mx_sym = mx.sym.elemwise_add(data, zeros)
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 3, 4))

def test_forward_ones_like():
    data = mx.sym.var('data')
    mx_sym = mx.sym.ones_like(data, dtype='float32')
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 3, 4))

def test_forward_zeros_like():
    data = mx.sym.var('data')
    mx_sym = mx.sym.zeros_like(data, dtype='float32')
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 3, 4))

def test_forward_argmax():
    data = mx.sym.var('data')
    mx_sym = mx.sym.argmax(data, axis=1)
    verify_mxnet_frontend_impl(mx_sym, (5, 3), (5,))

def test_forward_argmin():
    data = mx.sym.var('data')
    mx_sym = mx.sym.argmin(data, axis=0)
    verify_mxnet_frontend_impl(mx_sym, (5, 4), (4,))

def test_forward_slice():
    data = mx.sym.var('data')
    mx_sym = mx.sym.slice(data, begin=(0, 1), end=(2, 4))
    verify_mxnet_frontend_impl(mx_sym, (3, 4), (2, 3))
    mx_sym = mx.sym.slice(data, begin=(-1, 1), end=(-3, 4), step=(-1, 2))
    verify_mxnet_frontend_impl(mx_sym, (3, 4), (2, 2))

def test_forward_where():
    cond = mx.sym.var('cond')
    x = mx.sym.var('x')
    y = mx.sym.var('y')
    dshape = (2, 2)
    dtype = 'float32'
    mx_sym = mx.sym.where(cond, x, y)
    np_cond = np.array([[0, 1], [-1, 0]]).astype(dtype)
    np_x = np.random.uniform(size=dshape).astype(dtype)
    np_y = np.random.uniform(size=dshape).astype(dtype)
    mx_cond = mx.nd.array(np_cond)
    mx_x = mx.nd.array(np_x)
    mx_y = mx.nd.array(np_y)
    shapes = {'cond': dshape, 'x': dshape, 'y': dshape}
    mod = mx.mod.Module(mx_sym, label_names=None, data_names=['cond', 'x', 'y'])
    mod.bind(data_shapes=shapes.items(), for_training=False)
    mod.init_params()
    args, auxs = mod.get_params()
    mx_out = mx.nd.where(mx_cond, mx_x, mx_y).asnumpy()

    new_sym, _ = relay.frontend.from_mxnet(mx_sym, shapes, args, auxs)
    for target, ctx in ctx_list():
        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, ctx=ctx, target=target)
            op_res = intrp.evaluate(new_sym)(np_cond, np_x, np_y)
            tvm.testing.assert_allclose(op_res.asnumpy(), mx_out)


def test_forward_arange():
    def _mx_symbol(F, start, stop, step):
        if start is None and step is None:
            sym = F.arange(stop)
        elif start is None:
            sym = F.arange(stop, step=step)
        elif step is None:
            sym = F.arange(start, stop)
        else:
            sym = F.arange(start, stop, step)
        return sym

    def verify(start, stop, step):
        ref_res = _mx_symbol(mx.nd, start, stop, step).asnumpy()
        mx_sym = _mx_symbol(mx.sym, start, stop, step)
        new_sym, _ = relay.frontend.from_mxnet(mx_sym, {})
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(new_sym)()
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)
    verify(0, 20, None)
    verify(0, 20, 2)
    verify(1, 20, None)
    verify(1, 20, 2)
    verify(1, 20, 1.5)
    verify(1, 20.5, None)
    verify(1, 20, 3)
    verify(20, 1, -1)
    verify(20, 1, -1.5)


if __name__ == '__main__':
    test_forward_mlp()
    test_forward_vgg()
    test_forward_resnet()
    test_forward_elu()
    test_forward_rrelu()
    test_forward_prelu()
    test_forward_softrelu()
    test_forward_fc_flatten()
    test_forward_clip()
    test_forward_split()
    test_forward_split_squeeze()
    test_forward_expand_dims()
    test_forward_pooling()
    test_forward_lrn()
    test_forward_ones()
    test_forward_zeros()
    test_forward_ones_like()
    test_forward_zeros_like()
    test_forward_argmax()
    test_forward_argmin()
    test_forward_where()
    test_forward_arange()
