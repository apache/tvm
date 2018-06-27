import numpy as np
import nnvm
import tvm
from tvm.contrib import graph_runtime
from nnvm.testing.config import ctx_list
import onnx
from model_zoo import super_resolution, squeezenet1_1, lenet, resnet18_1_0
from onnx import helper, TensorProto

def get_tvm_output(model, x, target, ctx, out_shape, dtype='float32'):
    new_sym, params = nnvm.frontend.from_onnx(model)
    input_name = model.graph.input[0].name
    shape_dict = {input_name: x.shape}
    dtype_dict = {input_name: dtype}
    graph, lib, params = nnvm.compiler.build(new_sym, target, shape_dict, dtype_dict, params=params)
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input(input_name, tvm.nd.array(x.astype(dtype)))
    m.set_input(**params)
    m.run()
    # get outputs
    out = m.get_output(0, tvm.nd.empty(out_shape, dtype))
    return out.asnumpy()


def verify_onnx_forward_impl(graph_file, data_shape, out_shape):
    import caffe2.python.onnx.backend
    def get_caffe2_output(model, x, dtype='float32'):
        prepared_backend = caffe2.python.onnx.backend.prepare(model)
        W = {model.graph.input[0].name: x.astype(dtype)}
        c2_out = prepared_backend.run(W)[0]
        return c2_out

    dtype = 'float32'
    x = np.random.uniform(size=data_shape)
    model = onnx.load(graph_file)
    c2_out = get_caffe2_output(model, x, dtype)
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, x, target, ctx, out_shape, dtype)
        np.testing.assert_allclose(c2_out, tvm_out, rtol=1e-5, atol=1e-5)

def verify_super_resolution_example():
    verify_onnx_forward_impl(super_resolution, (1, 1, 224, 224), (1, 1, 672, 672))

def verify_squeezenet1_1():
    verify_onnx_forward_impl(squeezenet1_1, (1, 3, 224, 224), (1, 1000))

def verify_lenet():
    verify_onnx_forward_impl(lenet, (1, 1, 28, 28), (1, 10))

def verify_resnet18():
    verify_onnx_forward_impl(resnet18_1_0, (1, 3, 224, 224), (1, 1000))


def test_reshape():
    in_shape = (4, 3, 3, 4)
    ref_shape = (3, 4, 4, 3)

    ref_array = np.array(ref_shape)
    ref_node = onnx.helper.make_node('Constant',
                                 inputs=[],
                                 outputs=['ref_in'],
                                 value=onnx.helper.make_tensor(name = 'const_tensor',
                                                               data_type = onnx.TensorProto.INT32,
                                                               dims = ref_array.shape,
                                                               vals = ref_array.flatten().astype(int)))
    reshape_node = helper.make_node("Reshape", ["in", "ref_in"], ["out"])

    graph = helper.make_graph([ref_node, reshape_node],
                              "reshape_test",
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(ref_shape))])

    model = helper.make_model(graph, producer_name='reshape_test')

    for target, ctx in ctx_list():
        x = np.random.uniform(size=in_shape)
        tvm_out = get_tvm_output(model, x, target, ctx, ref_shape, 'float32')

    np.testing.assert_allclose(ref_shape, tvm_out.shape)

def test_reshape_like():
    in_shape = (4, 3, 3, 4)
    ref_shape = (3, 4, 4, 3)

    ref_array = np.random.uniform(size=ref_shape).astype('float32')
    ref_node = onnx.helper.make_node('Constant',
                                 inputs=[],
                                 outputs=['ref_in'],
                                 value=onnx.helper.make_tensor(name = 'const_tensor',
                                                               data_type = onnx.TensorProto.FLOAT,
                                                               dims = ref_array.shape,
                                                               vals = ref_array.flatten().astype(float)))
    copy_node = helper.make_node("Identity", ["ref_in"], ["copy_in"])
    reshape_node = helper.make_node("Reshape", ["in", "copy_in"], ["out"])

    graph = helper.make_graph([ref_node, copy_node, reshape_node],
                              "reshape_like_test",
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(ref_shape))])

    model = helper.make_model(graph, producer_name='reshape_like_test')

    for target, ctx in ctx_list():
        x = np.random.uniform(size=in_shape)
        tvm_out = get_tvm_output(model, x, target, ctx, ref_shape, 'float32')

    np.testing.assert_allclose(ref_shape, tvm_out.shape)

def test_squeeze():
    in_shape = (1, 3, 1, 3, 1, 1)
    out_shape = (3, 3)
    y = helper.make_node("Squeeze", ['in'], ['out'])

    graph = helper.make_graph([y],
                              'squeeze_test',
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='squeeze_test')

    for target, ctx in ctx_list():
        x = np.random.uniform(size=in_shape)
        tvm_out = get_tvm_output(model, x, target, ctx, out_shape, 'float32')

    np.testing.assert_allclose(out_shape, tvm_out.shape)

def test_unsqueeze():
    in_shape = (3, 3)
    axis = (0, 3, 4)
    out_shape = (1, 3, 3, 1, 1)
    y = helper.make_node("Unsqueeze", ['in'], ['out'], axes=list(axis))

    graph = helper.make_graph([y],
                              'squeeze_test',
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='squeeze_test')

    for target, ctx in ctx_list():
        x = np.random.uniform(size=in_shape)
        tvm_out = get_tvm_output(model, x, target, ctx, out_shape, 'float32')

    np.testing.assert_allclose(out_shape, tvm_out.shape)

if __name__ == '__main__':
    # verify_super_resolution_example()
    # verify_squeezenet1_1()
    # verify_lenet()
    verify_resnet18()
    test_reshape()
    test_reshape_like()
    test_squeeze()
    test_unsqueeze()
