import numpy as np

import topi
import tvm
import nnvm.symbol as sym
import nnvm.compiler
import nnvm.runtime
from nnvm import frontend
import mxnet as mx
import model_zoo

USE_GPU=True

def default_target():
    if USE_GPU:
        return 'cuda'
    else:
        return 'llvm'

def default_ctx():
    if USE_GPU:
        return tvm.gpu(0)
    else:
        return tvm.cpu(0)

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

    def get_tvm_output(symbol, x, args, auxs, dtype='float32'):
        dshape = x.shape
        shape_dict = {'data': dshape}
        for k, v in args.items():
            shape_dict[k] = v.shape
        for k, v in auxs.items():
            shape_dict[k] = v.shape
        graph, lib, _ = nnvm.compiler.build(symbol, default_target(), shape_dict)
        m = nnvm.runtime.create(graph, lib, default_ctx())
        # get member functions
        set_input, run, get_output = m['set_input'], m['run'], m['get_output']
        # set inputs
        set_input('data', tvm.nd.array(x.astype(dtype)))
        for k, v in args.items():
            set_input(k, tvm.nd.array(v.asnumpy().astype(dtype)))
        for k, v in auxs.items():
            set_input(k, tvm.nd.array(v.asnumpy().astype(dtype)))
        # execute
        run()
        # get outputs
        out = tvm.nd.empty(out_shape, dtype)
        get_output(0, out)
        return out.asnumpy()

    # random input
    dtype = 'float32'
    x = np.random.uniform(size=data_shape)
    mx_out, args, auxs = get_mxnet_output(mx_symbol, x, dtype)
    new_sym = frontend.from_mxnet(mx_symbol)
    tvm_out = get_tvm_output(new_sym, x, args, auxs, dtype)
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
