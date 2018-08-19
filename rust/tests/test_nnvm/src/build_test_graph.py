#!/usr/bin/env python3

"""Builds a simple NNVM graph for testing."""

from os import path as osp
import sys

import nnvm
from nnvm import sym
from nnvm.compiler import graph_util
from nnvm.testing import init
import numpy as np
import tvm


def _get_model(dshape):
    data = sym.Variable('data', shape=dshape)
    fc = sym.dense(data, units=dshape[-1]*2, use_bias=True)
    left, right = sym.split(fc, indices_or_sections=2, axis=1)
    return sym.Group(((left + 1), (right - 1), fc))


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

        init_value = np.arange(np.product(shape), 0, -1).reshape(*shape).astype('float32')
        if param.endswith('_bias'):
            params[param] = tvm.nd.array(init_value)
            continue

        init_value = np.empty(shape).astype('float32')
        initializer(param, init_value)
        # init_value /= init_value.sum() + 1e-10
        params[param] = tvm.nd.array(init_value)
    return params

def main():
    dshape = (4, 8)
    net = _get_model(dshape)
    ishape_dict = {'data': dshape}
    params = _init_params(net, ishape_dict)
    graph, lib, params = nnvm.compiler.build(net, 'llvm --system-lib',
                                             shape=ishape_dict,
                                             params=params,
                                             dtype='float32')

    out_dir = sys.argv[1]
    lib.save(osp.join(sys.argv[1], 'graph.o'))
    with open(osp.join(out_dir, 'graph.json'), 'w') as f_resnet:
        f_resnet.write(graph.json())
    with open(osp.join(out_dir, 'graph.params'), 'wb') as f_params:
        f_params.write(nnvm.compiler.save_param_dict(params))

if __name__ == '__main__':
    main()
