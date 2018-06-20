import numpy as np

import topi
import tvm
from tvm.contrib import graph_runtime
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

if __name__ == '__main__':
    test_mobilenet_checkonly()
    test_resnet50_checkonly()
