import numpy as np
import nnvm
import tvm
from tvm.contrib import graph_runtime
from nnvm.testing.config import ctx_list
import onnx
from model_zoo import super_resolution, squeezenet1_1, lenet, resnet18_1_0

def verify_onnx_forward_impl(graph_file, data_shape, out_shape):
    import caffe2.python.onnx.backend
    def get_caffe2_output(model, x, dtype='float32'):
        prepared_backend = caffe2.python.onnx.backend.prepare(model)
        W = {model.graph.input[0].name: x.astype(dtype)}
        c2_out = prepared_backend.run(W)[0]
        return c2_out

    def get_tvm_output(model, x, target, ctx, dtype='float32'):
        new_sym, params = nnvm.frontend.from_onnx(model)
        shape_dict = {'input_0': x.shape}
        graph, lib, params = nnvm.compiler.build(new_sym, target, shape_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input('input_0', tvm.nd.array(x.astype(dtype)))
        m.set_input(**params)
        m.run()
        # get outputs
        out = m.get_output(0, tvm.nd.empty(out_shape, dtype))
        return out.asnumpy()

    dtype = 'float32'
    x = np.random.uniform(size=data_shape)
    model = onnx.load(graph_file)
    c2_out = get_caffe2_output(model, x, dtype)
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, x, target, ctx, dtype)
        np.testing.assert_allclose(c2_out, tvm_out, rtol=1e-5, atol=1e-5)

def verify_super_resolution_example():
    verify_onnx_forward_impl(super_resolution, (1, 1, 224, 224), (1, 1, 672, 672))

def verify_squeezenet1_1():
    verify_onnx_forward_impl(squeezenet1_1, (1, 3, 224, 224), (1, 1000))

def verify_lenet():
    verify_onnx_forward_impl(lenet, (1, 1, 28, 28), (1, 10))

def verify_resnet18():
    verify_onnx_forward_impl(resnet18_1_0, (1, 3, 224, 224), (1, 1000))

if __name__ == '__main__':
    # verify_super_resolution_example()
    # verify_squeezenet1_1()
    # verify_lenet()
    verify_resnet18()
