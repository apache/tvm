import numpy as np
import math
import nnvm
import topi
import topi.testing
import tvm
from tvm.contrib import graph_runtime
from nnvm.testing.config import ctx_list
import onnx
from model_zoo import super_resolution, squeezenet1_1, lenet, resnet18_1_0
from onnx import helper, TensorProto

def get_tvm_output(graph_def, input_data, target, ctx, output_shape, output_dtype='float32'):
    """ Generic function to execute and get tvm output"""

    sym, params = nnvm.frontend.from_onnx(graph_def)
    target = 'llvm'
    if isinstance(input_data, list):
        input_names = {}
        shape_dict = {}
        dtype_dict = {}
        for i, _ in enumerate(input_data):
            input_names[i] = graph_def.graph.input[i].name
            shape_dict[input_names[i]] = input_data[i].shape
            dtype_dict[input_names[i]] = input_data[i].dtype
    else:
        input_names = graph_def.graph.input[0].name
        shape_dict = {input_names: input_data.shape}
        dtype_dict = {input_names: input_data.dtype}

    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict,
                                             dtype=dtype_dict, params=params)

    ctx = tvm.cpu(0)
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    if isinstance(input_data, list):
        for i, e in enumerate(input_names):
            m.set_input(input_names[i], tvm.nd.array(input_data[i].astype(input_data[i].dtype)))
    else:
        m.set_input(input_names, tvm.nd.array(input_data.astype(input_data.dtype)))

    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    if isinstance(output_shape, list) and isinstance(output_dtype, list):
        tvm_output_list = []
        for i, s in enumerate(output_shape):
            tvm_output = m.get_output(i, tvm.nd.empty((s), output_dtype[i]))
            tvm_output_list.append(tvm_output.asnumpy())
        return tvm_output_list
    else:
        tvm_output = m.get_output(0, tvm.nd.empty((output_shape), output_dtype))
        return tvm_output.asnumpy()

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
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(ref_shape))])

    model = helper.make_model(graph, producer_name='reshape_test')

    for target, ctx in ctx_list():
        x = np.random.uniform(size=in_shape).astype('int32')
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
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(ref_shape))])

    model = helper.make_model(graph, producer_name='reshape_like_test')

    for target, ctx in ctx_list():
        x = np.random.uniform(size=in_shape).astype('float32')
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
                              inputs = [helper.make_tensor_value_info("x",
                                            TensorProto.FLOAT, list(x_shape)),
                                        helper.make_tensor_value_info("y",
                                            TensorProto.FLOAT, list(y_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(np_res.shape))])

    model = helper.make_model(graph, producer_name='power_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [x, y], target, ctx, np_res.shape)
        np.testing.assert_allclose(np_res, tvm_out, rtol=1e-5, atol=1e-5)

def test_power():
    _test_power_iteration((1, 3), (1))
    _test_power_iteration((2, 3), (2, 3))
    _test_power_iteration((2, 3), (1, 3))

def test_squeeze():
    in_shape = (1, 3, 1, 3, 1, 1)
    out_shape = (3, 3)
    y = helper.make_node("Squeeze", ['in'], ['out'], axes=[0, 2, 4, 5])

    graph = helper.make_graph([y],
                              'squeeze_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='squeeze_test')

    for target, ctx in ctx_list():
        x = np.random.uniform(size=in_shape).astype('float32')
        tvm_out = get_tvm_output(model, x, target, ctx, out_shape, 'float32')

    np.testing.assert_allclose(out_shape, tvm_out.shape)

def test_unsqueeze():
    in_shape = (3, 3)
    axis = (0, 3, 4)
    out_shape = (1, 3, 3, 1, 1)
    y = helper.make_node("Unsqueeze", ['in'], ['out'], axes=list(axis))

    graph = helper.make_graph([y],
                              'squeeze_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='squeeze_test')

    for target, ctx in ctx_list():
        x = np.random.uniform(size=in_shape).astype('float32')
        tvm_out = get_tvm_output(model, x, target, ctx, out_shape, 'float32')

    np.testing.assert_allclose(out_shape, tvm_out.shape)

def verify_gather(in_shape, indices, axis, dtype):
    x = np.random.uniform(size=in_shape).astype(dtype)
    indices = np.array(indices, dtype="int32")
    out_np = np.take(x, indices, axis=axis)

    y = helper.make_node("Gather", ['in', 'indices'], ['out'], axis=axis)

    graph = helper.make_graph([y],
                              'gather_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape)),
                                        helper.make_tensor_value_info("indices",
                                            TensorProto.INT32, list(indices.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(out_np.shape))])
    model = helper.make_model(graph, producer_name='gather_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [x, indices], target, ctx, out_np.shape)
        np.testing.assert_allclose(out_np, tvm_out)

def test_gather():
    verify_gather((4,), [1], 0, 'int32')
    verify_gather((1,4), [0], 0, 'int32')
    verify_gather((4,), [[[1,0],[0,1]]], 0, 'float32')
    verify_gather((2,2), [[[1,0],[0,1]]], 1, 'int32')
    verify_gather((3,3,3), [[[1,0]]], -1, 'int32')
    verify_gather((4,3,5,6), [[2,1,0,0]], 0, 'float32')

def _test_slice_iteration(indata, outdata, starts, ends, axes=None):
    if axes:
        y = helper.make_node("Slice", ['in'], ['out'], axes=axes, starts=starts, ends=ends)
    else:
        y = helper.make_node("Slice", ['in'], ['out'], starts=starts, ends=ends)

    graph = helper.make_graph([y],
                              'slice_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(outdata.shape))])

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
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(outdata.shape))])

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

    a_array = np.random.uniform(size=a_shape).astype('float32')
    b_array = np.random.uniform(size=b_shape).astype('float32')
    out_np = np.matmul(a_array, b_array)

    mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])

    graph = helper.make_graph([mul_node],
                              "matmul_test",
                              inputs = [helper.make_tensor_value_info("a",
                                            TensorProto.FLOAT, list(a_shape)),
                                        helper.make_tensor_value_info("b",
                                            TensorProto.FLOAT, list(b_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(out_np.shape))])

    model = helper.make_model(graph, producer_name='matmul_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_array, b_array], target, ctx, out_np.shape)
        np.testing.assert_allclose(out_np, tvm_out, rtol=1e-5, atol=1e-5)

def verify_lrn(shape, nsize, dtype, alpha=None, beta=None, bias=None):
    in_array = np.random.uniform(size=shape).astype(dtype)

    if alpha == None and beta == None and bias==None:
        alpha = 0.0001
        beta = 0.75
        bias = 1.0
        node = onnx.helper.make_node('LRN', inputs=['in'], outputs=['out'], size=nsize)
    else:
        node = onnx.helper.make_node('LRN', inputs=['in'], outputs=['out'], alpha=alpha,
                                     beta=beta, bias=bias, size=nsize)

    graph = helper.make_graph([node],
                              "lrn_test",
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(shape))])
    model = helper.make_model(graph, producer_name='lrn_test')

    def _get_python_lrn():
        square_sum = np.zeros(shape).astype(dtype)
        for n, c, h, w in np.ndindex(in_array.shape):
            square_sum[n, c, h, w] = sum(in_array[n,
                                         max(0, c - int(math.floor((nsize - 1) / 2))): \
                                             min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
                                         h,
                                         w] ** 2)
        py_out = in_array / ((bias + (alpha / nsize) * square_sum) ** beta)
        return py_out

    for target, ctx in ctx_list():
        new_sym, params = nnvm.frontend.from_onnx(model)

        input_name = model.graph.input[0].name
        shape_dict = {input_name: in_array.shape}
        dtype_dict = {input_name: dtype}
        graph, lib, params = nnvm.compiler.build(new_sym, target,
                                                 shape_dict, dtype_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input(input_name, tvm.nd.array(in_array.astype(dtype)))
        m.set_input(**params)
        m.run()
        # get outputs
        tvm_out = m.get_output(0, tvm.nd.empty(shape, dtype))
        py_out = _get_python_lrn()
        np.testing.assert_allclose(py_out, tvm_out.asnumpy(), rtol=1e-5, atol=1e-5)

def test_lrn():
    verify_lrn((5, 5, 5, 5), 3, 'float32')
    verify_lrn((5, 5, 5, 5), 3, 'float32', alpha=0.0002, beta=0.5, bias=2.0)

def _test_upsample_nearest():
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3*scale, 3*scale)
    y = helper.make_node("Upsample", ['in'], ['out'], mode='nearest', scales=[1.0, 1.0, 2.0, 2.0])

    in_array = np.random.uniform(size=in_shape).astype(np.float32)
    out_array = topi.testing.upsampling_python(in_array, scale, "NCHW")

    graph = helper.make_graph([y],
                              'upsample_nearest_test',
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='upsample_nearest_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, in_array, target, ctx, out_shape, 'float32')
        np.testing.assert_allclose(out_array, tvm_out)

def _test_upsample_bilinear():
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3*scale, 3*scale)
    y = helper.make_node("Upsample", ['in'], ['out'], mode='linear', scales=[1.0, 1.0, 2.0, 2.0])

    in_array = np.random.uniform(size=in_shape).astype(np.float32)
    out_array = topi.testing.bilinear_resize_python(in_array, (3*scale, 3*scale), "NCHW")

    graph = helper.make_graph([y],
                              'upsample_bilinear_test',
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='upsample_bilinear_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, in_array, target, ctx, out_shape, 'float32')
        np.testing.assert_allclose(out_array, tvm_out, rtol=1e-5, atol=1e-5)

def test_upsample():
    _test_upsample_nearest()
    _test_upsample_bilinear()

def _test_softmax(inshape, axis):
    opname = 'Softmax'
    indata = np.random.uniform(size=inshape).astype(np.float32)
    outshape = inshape
    outdata = topi.testing.softmax_python(indata)
    if isinstance(axis, int):
        y = helper.make_node(opname, ['in'], ['out'], axis = axis)
    elif axis is None:
        y = helper.make_node(opname, ['in'], ['out'])

    graph = helper.make_graph([y],
                              opname+'_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(outdata.shape))])

    model = helper.make_model(graph, producer_name=opname+'_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, indata, target, ctx, outshape, 'float32')
        np.testing.assert_allclose(outdata, tvm_out, rtol=1e-5, atol=1e-5)

def test_softmax():
    _test_softmax((1, 10), None)
    _test_softmax((1, 10), 1)

def verify_min(input_dim):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.min((a_np1, a_np2, a_np3), axis=0)

    min_node = helper.make_node("Min", ["a_np1", "a_np2", "a_np3"], ["out"])

    graph = helper.make_graph([min_node],
                              "Min_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np2",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np3",
                                            TensorProto.FLOAT, list(input_dim))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='Min_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1, a_np2, a_np3], target, ctx, b_np.shape)
        np.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_min():
    verify_min((1, 3, 20, 20))
    verify_min((20, 20))

def verify_max(input_dim):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.max((a_np1, a_np2, a_np3), axis=0)

    max_node = helper.make_node("Max", ["a_np1", "a_np2", "a_np3"], ["out"])

    graph = helper.make_graph([max_node],
                              "Max_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np2",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np3",
                                            TensorProto.FLOAT, list(input_dim))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='Max_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1, a_np2, a_np3], target, ctx, b_np.shape)
        np.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_max():
    verify_max((1, 3, 20, 20))
    verify_max((20, 20))

def verify_mean(input_dim):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.mean((a_np1, a_np2, a_np3), axis=0)

    mean_node = helper.make_node("Mean", ["a_np1", "a_np2", "a_np3"], ["out"])

    graph = helper.make_graph([mean_node],
                              "Mean_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np2",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np3",
                                            TensorProto.FLOAT, list(input_dim))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='Mean_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1, a_np2, a_np3], target, ctx, b_np.shape)
        np.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_mean():
    verify_mean((1, 3, 20, 20))
    verify_mean((20, 20))

def verify_hardsigmoid(input_dim, alpha, beta):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.clip(a_np1 * alpha + beta, 0, 1)

    hardsigmoid_node = helper.make_node("HardSigmoid", ["a_np1"], ["out"], alpha=alpha, beta=beta)

    graph = helper.make_graph([hardsigmoid_node],
                              "HardSigmoid_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.FLOAT, list(input_dim))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='HardSigmoid_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1], target, ctx, b_np.shape)
        np.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_hardsigmoid():
    verify_hardsigmoid((1, 3, 20, 20), 0.5, 0.6)
    verify_hardsigmoid((20, 20), 0.3, 0.4)

def verify_argmin(input_dim, axis=None, keepdims=None):
    def _argmin_numpy(data, axis=0, keepdims=True):
        result = np.argmin(data, axis=axis)
        if (keepdims == 1):
            result = np.expand_dims(result, axis)
        return result.astype(data.dtype)

    a_np1 = np.random.uniform(-10, 10, input_dim).astype(np.int32)
    if keepdims is None and axis is None:
        b_np = _argmin_numpy(a_np1)
        node = onnx.helper.make_node('ArgMin',
                                     inputs=['a_np1'],
                                     outputs=['out'])
    elif axis is None:
        b_np = _argmin_numpy(a_np1, keepdims=keepdims)
        node = onnx.helper.make_node('ArgMin',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     keepdims=keepdims)
    elif keepdims is None:
        b_np = _argmin_numpy(a_np1, axis=axis)
        node = onnx.helper.make_node('ArgMin',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     axis=axis)
    else:
        b_np = _argmin_numpy(a_np1, axis=axis, keepdims=keepdims)
        node = onnx.helper.make_node('ArgMin',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     axis=axis,
                                     keepdims=keepdims)
    graph = helper.make_graph([node],
                              "argmin_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.INT32, list(a_np1.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.INT32, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='argmin_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1], target, ctx, b_np.shape, b_np.dtype)
        np.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def verify_argmax(input_dim, axis=None, keepdims=None):
    def _argmax_numpy(data, axis=0, keepdims=True):
        result = np.argmax(data, axis=axis)
        if (keepdims == 1):
            result = np.expand_dims(result, axis)
        return result.astype(data.dtype)

    a_np1 = np.random.uniform(-10, 10, input_dim).astype(np.int32)

    if keepdims is None and axis is None:
        b_np = _argmax_numpy(a_np1)
        node = onnx.helper.make_node('ArgMax',
                                     inputs=['a_np1'],
                                     outputs=['out'])
    elif axis is None:
        b_np = _argmax_numpy(a_np1, keepdims=keepdims)
        node = onnx.helper.make_node('ArgMax',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     keepdims=keepdims)
    elif keepdims is None:
        b_np = _argmax_numpy(a_np1, axis=axis)
        node = onnx.helper.make_node('ArgMax',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     axis=axis)
    else:
        b_np = _argmax_numpy(a_np1, axis=axis, keepdims=keepdims)
        node = onnx.helper.make_node('ArgMax',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     axis=axis,
                                     keepdims=keepdims)

    graph = helper.make_graph([node],
                              "argmax_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.INT32, list(a_np1.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.INT32, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='argmax_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1], target, ctx, b_np.shape, b_np.dtype)
        np.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_arg_min_max():
    '''Verify argmin and argmax'''
    verify_argmin([3,4,4])
    verify_argmax([3,4,4])
    verify_argmin([3,4,4], axis=1)
    verify_argmax([3,4,4], axis=0)
    verify_argmin([3,4,4], keepdims=0)
    verify_argmax([3,4,4], keepdims=1)
    for axis in [0,1,2]:
        for keepdims in [True,False]:
            verify_argmin([3,4,4], axis, keepdims)
            verify_argmax([3,4,4], axis, keepdims)

def verify_constantfill(is_shape, input_dim, out_dim, value, dtype, **kwargs):
    input_a = np.random.uniform(size=input_dim).astype(dtype)
    out = np.empty(shape=out_dim, dtype=dtype)
    out.fill(value)

    if is_shape == True:
        fill_node = helper.make_node("ConstantFill", [], ["out"], shape=input_dim, value=value, **kwargs)
    else:
        fill_node = helper.make_node("ConstantFill", ["input_a"], ["out"], value=value, dtype=dtype, **kwargs)

    graph = helper.make_graph([fill_node],
                              "fill_test",
                              inputs = [helper.make_tensor_value_info("input_a",
                                            TensorProto.FLOAT, list(input_dim))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(out.shape))])

    model = helper.make_model(graph, producer_name='fill_test')

    for target, ctx in ctx_list():
        if is_shape == True:
            tvm_out = get_tvm_output(model, [], target, ctx, out.shape)
        else:
            tvm_out = get_tvm_output(model, [input_a], target, ctx, out.shape)

        np.testing.assert_allclose(out, tvm_out, rtol=1e-5, atol=1e-5)

def test_constantfill():
    verify_constantfill(True, (2, 3, 4, 5), (2, 3, 4, 5), 10, 'float32')
    verify_constantfill(False, (2, 3, 4, 5), (2, 3, 4, 5), 10, 'float32')
    verify_constantfill(True, (2, 3, 4, 5), (2, 3, 4, 5, 4, 5, 6), 10, 'float32', extra_shape=(4, 5, 6))

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
    test_gather()
    test_lrn()
    test_upsample()
    test_forward_min()
    test_forward_max()
    test_forward_mean()
    test_forward_hardsigmoid()
    test_forward_arg_min_max()
    test_softmax()
    test_constantfill()
