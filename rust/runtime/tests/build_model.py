"""Builds a simple NNVM graph for testing."""

from os import path as osp

import nnvm
from nnvm import sym
from nnvm.compiler import graph_util
from nnvm.testing import init
import numpy as np
import tvm

CWD = osp.dirname(osp.abspath(osp.expanduser(__file__)))


def _get_model(dshape):
    data = sym.Variable('data', shape=dshape)
    fc1 = sym.dense(data, units=dshape[-1]*2, use_bias=True)
    left, right = sym.split(fc1, indices_or_sections=2, axis=1)
    return sym.Group(((left + 1), (right - 1)))


def _init_params(graph, input_shapes, initializer=init.Xavier(), seed=10):
    if isinstance(graph, sym.Symbol):
        graph = nnvm.graph.create(graph)
    ishapes, _ = graph_util.infer_shape(graph, **input_shapes)
    param_shapes = dict(zip(graph.index.input_names, ishapes))
    np.random.seed(seed)
    params = {}
    for param, shape in param_shapes.items():
        if param in {'data', 'label'} or not shape:
            continue
        init_value = np.empty(shape).astype('float32')
        initializer(param, init_value)
        params[param] = tvm.nd.array(init_value)
    return params

def main():
    dshape = (32, 16)
    net = _get_model(dshape)
    ishape_dict = {'data': dshape}
    params = _init_params(net, ishape_dict)
    graph, lib, params = nnvm.compiler.build(net, 'llvm',
                                             shape=ishape_dict,
                                             params=params,
                                             dtype='float32')

    with open(osp.join(CWD, 'graph.json'), 'w') as f_resnet:
        f_resnet.write(graph.json())
    with open(osp.join(CWD, 'graph.params'), 'wb') as f_params:
        f_params.write(nnvm.compiler.save_param_dict(params))

if __name__ == '__main__':
    main()
