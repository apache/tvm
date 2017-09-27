import numpy as np
import nnvm
import tvm
from tvm.contrib import graph_runtime
from nnvm.testing.config import ctx_list
import onnx
from model_zoo import super_resolution

def verify_onnx_forward_impl(graph_file, data_shape, out_shape):
    import onnx_caffe2.backend
    def get_caffe2_output(graph, x, dtype='float32'):
        prepared_backend = onnx_caffe2.backend.prepare(graph)
        W = {graph.input[-1]: x.astype(dtype)}
        c2_out = prepared_backend.run(W)[0]
        return c2_out

    def get_tvm_output(graph, x, target, ctx, dtype='float32'):
        new_sym, params = nnvm.frontend.from_onnx(graph)
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
    graph = onnx.load(graph_file)
    c2_out = get_caffe2_output(graph, x, dtype)
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(graph, x, target, ctx, dtype)
        np.testing.assert_allclose(c2_out, tvm_out, rtol=1e-5, atol=1e-5)

def verify_super_resolution_example():
    verify_onnx_forward_impl(super_resolution[0], (1, 1, 224, 224), (1, 1, 672, 672))

if __name__ == '__main__':
    verify_super_resolution_example()
