import numpy as np

import topi
import tvm
from tvm.contrib import graph_runtime
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list
from nnvm import frontend
import mxnet as mx
import model_zoo


def test_mxnet_frontend_impl(mx_symbol, data_shape=(1, 3, 224, 224), out_shape=(1, 1000)):
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
        new_sym, params = frontend.from_mxnet(symbol, args, auxs)
        dshape = x.shape
        shape_dict = {'data': dshape}
        graph, lib, params = nnvm.compiler.build(new_sym, target, shape_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input("data", tvm.nd.array(x.astype(dtype)))
        m.set_input(**params)
        m.run()
        # get outputs
        out = m.get_output(0, tvm.nd.empty(out_shape, dtype))
        return out.asnumpy()

    # random input
    dtype = 'float32'
    x = np.random.uniform(size=data_shape)
    mx_out, args, auxs = get_mxnet_output(mx_symbol, x, dtype)
    assert "data" not in args
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(mx_symbol, x, args, auxs, target, ctx, dtype)
        np.testing.assert_allclose(mx_out, tvm_out, rtol=1e-5)

def test_forward_mlp():
    mlp = model_zoo.mx_mlp
    test_mxnet_frontend_impl(mlp)

def test_forward_vgg():
    for n in [11]:
        mx_sym = model_zoo.mx_vgg[n]
        test_mxnet_frontend_impl(mx_sym)

def test_forward_resnet():
    for n in [18]:
        mx_sym = model_zoo.mx_resnet[n]
        test_mxnet_frontend_impl(mx_sym)

if __name__ == '__main__':
    test_forward_mlp()
    test_forward_vgg()
    test_forward_resnet()
