import numpy as np

from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import datatypes

import tvm
from tvm.contrib import graph_runtime
import topi
import topi.testing
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list
from nnvm import frontend
import coremltools as cm
import model_zoo

def get_tvm_output(symbol, x, params, target, ctx,
                   out_shape=(1000,), input_name='image', dtype='float32'):
    shape_dict = {input_name : x.shape}
    with nnvm.compiler.build_config(opt_level=3):
        graph, lib, params = nnvm.compiler.build(symbol, target, shape_dict, params=params)
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input(input_name, tvm.nd.array(x.astype(dtype)))
    m.set_input(**params)
    m.run()
    # get outputs
    out = m.get_output(0, tvm.nd.empty(out_shape, dtype))
    return out.asnumpy()

def test_model_checkonly(model_file, model_name=''):
    model = cm.models.MLModel(model_file)
    sym, params = nnvm.frontend.from_coreml(model)
    x = model_zoo.get_cat_image()
    for target, ctx in ctx_list():
        tvm_output = get_tvm_output(sym, x, params, target, ctx)
        print(target, ctx, model_name, 'prediction id: ', np.argmax(tvm_output.flat))

def test_mobilenet_checkonly():
    model_file = model_zoo.get_mobilenet()
    test_model_checkonly(model_file, 'mobilenet')

def test_resnet50_checkonly():
    model_file = model_zoo.get_resnet50()
    test_model_checkonly(model_file, 'resnet50')

def run_tvm_graph(graph_def, input_data, input_name, output_shape, output_dtype='float32'):
    """ Generic function to compile on nnvm and execute on tvm """

    sym, params = nnvm.frontend.from_coreml(graph_def)
    target = 'llvm'
    if isinstance(input_data, list):
        shape_dict = {}
        dtype_dict = {}
        for i, e in enumerate(input_name):
            shape_dict[e] = input_data[i].shape
            dtype_dict[e] = input_data[i].dtype
    else:
        shape_dict = {input_name: input_data.shape}
        dtype_dict = {input_name: input_data.dtype}

    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict,
                                             dtype=dtype_dict, params=params)

    ctx = tvm.cpu(0)
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    if isinstance(input_data, list):
        for i, e in enumerate(input_name):
            m.set_input(e, tvm.nd.array(input_data[i].astype(input_data[i].dtype)))
    else:
        m.set_input(input_name, tvm.nd.array(input_data.astype(input_data.dtype)))

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

def verify_AddLayerParams(input_dim, alpha=2):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.add(a_np1, a_np2) + alpha
    inputs = [('input1', datatypes.Array(*input_dim)),
              ('input2', datatypes.Array(*input_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(name='Add',
                            alpha=alpha,
                            input_names=['input1', 'input2'],
                            output_name='output',
                            mode='ADD')
    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model,
                           [a_np1, a_np2],
                           ['input1', 'input2'],
                           b_np.shape,
                           dtype)
        np.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_AddLayerParams():
    verify_AddLayerParams((1, 2, 2), 0)
    verify_AddLayerParams((1, 2, 2), 1)
    verify_AddLayerParams((1, 3, 3), 2)

def verify_MultiplyLayerParams(input_dim, alpha):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.multiply(a_np1, a_np2) * alpha
    inputs = [('input1', datatypes.Array(*input_dim)),
              ('input2', datatypes.Array(*input_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(name='Mul',
                            alpha=alpha,
                            input_names=['input1', 'input2'],
                            output_name='output',
                            mode='MULTIPLY')
    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model,
                           [a_np1, a_np2],
                           ['input1', 'input2'],
                           b_np.shape,
                           dtype)
        np.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_MultiplyLayerParams():
    verify_MultiplyLayerParams((1, 2, 2), 0)
    verify_MultiplyLayerParams((1, 2, 2), 1)
    verify_MultiplyLayerParams((1, 3, 3), 2)

def verify_ConcatLayerParams(input1_dim, input2_dim):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input1_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input2_dim).astype(dtype)

    b_np = np.concatenate((a_np1, a_np2), axis=1)
    inputs = [('input1', datatypes.Array(*input1_dim)),
              ('input2', datatypes.Array(*input2_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(name='Concate',
                            input_names=['input1', 'input2'],
                            output_name='output',
                            mode='CONCAT')
    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model,
                           [a_np1, a_np2],
                           ['input1', 'input2'],
                           b_np.shape,
                           dtype)
        np.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_ConcatLayerParams():
    verify_ConcatLayerParams((1, 1, 2, 2), (1, 2, 2, 2))
    verify_ConcatLayerParams((1, 2, 4, 4), (1, 3, 4, 4))

def verify_UpsampleLayerParams(input_dim, scale, mode):
    dtype = "float32"

    a_np = np.full(input_dim, 1, dtype=dtype)
    if mode == 'NN':
        b_np = topi.testing.upsampling_python(a_np, scale)
    else:
        new_h = input_dim[2] * scale
        new_w = input_dim[3] * scale
        b_np = topi.testing.bilinear_resize_python(a_np, (new_h, new_w), 'NCHW')

    input = [('input', datatypes.Array(*input_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(input, output)
    builder.add_upsample(name='Upsample',
                         scaling_factor_h=scale,
                         scaling_factor_w=scale,
                         mode=mode,
                         input_name='input',
                         output_name='output')

    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model, a_np, 'input', b_np.shape, dtype)
        np.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_UpsampleLayerParams():
    verify_UpsampleLayerParams((1, 16, 32, 32), 2, 'NN')
    verify_UpsampleLayerParams((1, 4, 6, 6), 3, 'BILINEAR')

def verify_l2_normalize(input_dim, eps):
    dtype = "float32"

    a_np = np.random.uniform(size=input_dim).astype(dtype)
    b_np = topi.testing.l2_normalize_python(a_np, eps, 1)

    input = [('input', datatypes.Array(*input_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(input, output)
    builder.add_l2_normalize(name='L2', epsilon=eps, input_name='input', output_name='output')

    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model, a_np, 'input', b_np.shape, dtype)
        np.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_l2_normalize():
    verify_l2_normalize((1, 3, 20, 20), 0.001)

def verify_lrn(input_dim, size, bias, alpha, beta):
    dtype = "float32"
    axis=1
    a_np = np.random.uniform(size=input_dim).astype(dtype)
    b_np = topi.testing.lrn_python(a_np, size, axis, bias, alpha, beta)

    input = [('input', datatypes.Array(*input_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(input, output)
    builder.add_lrn(name='LRN',
                    input_name='input',
                    output_name='output',
                    alpha=alpha,
                    beta=beta,
                    k=bias,
                    local_size=size)

    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model, a_np, 'input', b_np.shape, dtype)
        np.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_lrn():
    verify_lrn((1, 3, 10, 20), 3, 1.0, 1.0, 0.5)

def verify_average(input_dim1, input_dim2, axis=0):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim1).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim2).astype(dtype)

    b_np = np.mean((a_np1, a_np2), axis=axis)

    inputs = [('input1', datatypes.Array(*input_dim1)),
              ('input2', datatypes.Array(*input_dim2))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(name='MEAN',
                            input_names=['input1', 'input2'],
                            output_name='output',
                            mode='AVE')
    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model,
                           [a_np1, a_np2],
                           ['input1', 'input2'],
                           b_np.shape,
                           dtype)
        np.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_average():
    verify_average((1, 3, 20, 20), (1, 3, 20, 20))
    verify_average((3, 20, 20), (1, 3, 20, 20))
    verify_average((20, 20), (1, 3, 20, 20))

def verify_max(input_dim):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.max((a_np1, a_np2, a_np3), axis=0)

    inputs = [('input1', datatypes.Array(*input_dim)),
              ('input2', datatypes.Array(*input_dim)),
              ('input3', datatypes.Array(*input_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(name='Max',
                            input_names=['input1', 'input2', 'input3'],
                            output_name='output',
                            mode='MAX')
    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model,
                           [a_np1, a_np2, a_np3],
                           ['input1', 'input2', 'input3'],
                           b_np.shape,
                           dtype)
        np.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_max():
    verify_max((1, 3, 20, 20))
    verify_max((20, 20))

def verify_min(input_dim):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.min((a_np1, a_np2, a_np3), axis=0)

    inputs = [('input1', datatypes.Array(*input_dim)),
              ('input2', datatypes.Array(*input_dim)),
              ('input3', datatypes.Array(*input_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(name='Min',
                            input_names=['input1', 'input2', 'input3'],
                            output_name='output',
                            mode='MIN')
    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model,
                           [a_np1, a_np2, a_np3],
                           ['input1', 'input2', 'input3'],
                           b_np.shape,
                           dtype)
        np.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_min():
    verify_min((1, 3, 20, 20))
    verify_min((20, 20))

if __name__ == '__main__':
    test_mobilenet_checkonly()
    test_resnet50_checkonly()
    test_forward_AddLayerParams()
    test_forward_ConcatLayerParams()
    test_forward_MultiplyLayerParams()
    test_forward_UpsampleLayerParams()
    test_forward_l2_normalize()
    test_forward_lrn()
    test_forward_average()
    test_forward_max()
    test_forward_min()
