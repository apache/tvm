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


def get_caffe2_output(model, x, dtype='float32'):
    import caffe2.python.onnx.backend
    prepared_backend = caffe2.python.onnx.backend.prepare(model)
    W = {model.graph.input[0].name: x.astype(dtype)}
    c2_out = prepared_backend.run(W)[0]
    return c2_out


def verify_onnx_forward_impl(graph_file, data_shape, out_shape):
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

def _test_power_iteration(x_shape, y_shape):
    if isinstance(y_shape, int):
        y_shape = [y_shape]

    x = np.random.uniform(size=x_shape).astype(np.float32)
    y = np.random.uniform(size=y_shape).astype(np.float32)

    np_res = np.power(x, y).astype(np.float32)

    res = helper.make_node("Pow", ['x', 'y'], ['out'])

    graph = helper.make_graph([res],
                              'power_test',
                              inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
                                        helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(np_res.shape))])

    model = helper.make_model(graph, producer_name='power_test')

    for target, ctx in ctx_list():
        new_sym, params = nnvm.frontend.from_onnx(model)

        input_name = model.graph.input[0].name
        input_name1 = model.graph.input[1].name
        shape_dict = {input_name: x.shape, input_name1: y.shape}
        dtype_dict = {input_name: x.dtype, input_name1: y.dtype}

        graph, lib, params = nnvm.compiler.build(new_sym, target, shape_dict, dtype_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input(input_name, tvm.nd.array(x))
        m.set_input(input_name1, tvm.nd.array(y))
        m.set_input(**params)
        m.run()
        # get outputs
        tvm_out = m.get_output(0, tvm.nd.empty(np_res.shape, np_res.dtype))

        np.testing.assert_allclose(np_res, tvm_out.asnumpy(), rtol=1e-5, atol=1e-5)

def test_power():
    _test_power_iteration((1, 3), (1))
    _test_power_iteration((2, 3), (2, 3))
    _test_power_iteration((2, 3), (1, 3))

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

def _test_slice_iteration(indata, outdata, starts, ends, axes=None):
    if axes:
        y = helper.make_node("Slice", ['in'], ['out'], axes=axes, starts=starts, ends=ends)
    else:
        y = helper.make_node("Slice", ['in'], ['out'], starts=starts, ends=ends)

    graph = helper.make_graph([y],
                              'slice_test',
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))])

    model = helper.make_model(graph, producer_name='slice_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, indata, target, ctx, outdata.shape, 'float32')

    np.testing.assert_allclose(outdata, tvm_out)

def test_slice():
    x = np.random.randn(20, 10, 5).astype(np.float32)
    _test_slice_iteration(x, x[0:3, 0:10], (0, 0), (3, 10), (0, 1))
    _test_slice_iteration(x, x[:, :, 3:4], (0, 0, 3), (20, 10, 4))
    _test_slice_iteration(x, x[:, 1:1000], (1), (1000), (1))
    _test_slice_iteration(x, x[:, 0:-1], (0), (-1), (1))

def _test_onnx_op_elementwise(inshape, outfunc, npargs, dtype, opname, kwargs):
    indata = np.random.uniform(size=(2, 4, 5, 6)).astype(dtype)
    outdata = outfunc(indata, **npargs)

    y = helper.make_node(opname, ['in'], ['out'], **kwargs)

    graph = helper.make_graph([y],
                              opname+'_test',
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))])

    model = helper.make_model(graph, producer_name=opname+'_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, indata, target, ctx, outdata.shape, dtype)

    np.testing.assert_allclose(outdata, tvm_out)

def test_floor():
    _test_onnx_op_elementwise((2, 4, 5, 6), np.floor, {}, 'float32', 'Floor', {})

def test_ceil():
    _test_onnx_op_elementwise((2, 4, 5, 6), np.ceil, {}, 'float32', 'Ceil', {})

def test_clip():
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              np.clip,
                              {'a_min': -1.0, 'a_max': 1.0},
                              'float32',
                              'Clip',
                              {'min': -1.0, 'max': 1.0})

def test_matmul():
    a_shape = (4, 3)
    b_shape = (3, 4)
    out_shape = (4, 4)

    a_array = np.random.uniform(size=a_shape).astype('float32')
    b_array = np.random.uniform(size=b_shape).astype('float32')

    mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])

    graph = helper.make_graph([mul_node],
                              "matmul_test",
                              inputs = [helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
                                        helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='matmul_test')

    for target, ctx in ctx_list():
        new_sym, params = nnvm.frontend.from_onnx(model)

        input_name = model.graph.input[0].name
        input_name1 = model.graph.input[1].name
        shape_dict = {input_name: a_array.shape, input_name1: b_array.shape}
        dtype_dict = {input_name: 'float32', input_name1: 'float32'}

        graph, lib, params = nnvm.compiler.build(new_sym, target, shape_dict, dtype_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input(input_name, tvm.nd.array(a_array.astype('float32')))
        m.set_input(input_name1, tvm.nd.array(b_array.astype('float32')))
        m.set_input(**params)
        m.run()
        # get outputs
        tvm_out = m.get_output(0, tvm.nd.empty(out_shape, 'float32'))

        np.testing.assert_allclose(np.matmul(a_array, b_array), tvm_out.asnumpy(), rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    # verify_super_resolution_example()
    # verify_squeezenet1_1()
    # verify_lenet()
    verify_resnet18()
    test_reshape()
    test_reshape_like()
    test_power()
    test_squeeze()
    test_unsqueeze()
    test_slice()
    test_floor()
    test_ceil()
    test_clip()
    test_matmul()
