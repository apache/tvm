"""Helper utility to create common workload for testing."""
from __future__ import absolute_import as _abs

import numpy as np
import tvm
from ..compiler import graph_util
from ..import graph


def create_workload(net, batch_size, image_shape=(3, 224, 224), dtype="float32"):
    """Helper function to create benchmark workload for input network

    Parameters
    ----------
    net : nnvm.Symbol
        The selected network symbol to use

    batch_size : int
        The batch size used in the model

    image_shape : tuple, optional
        The input image shape

    dtype : str, optional
        The data type

    Returns
    -------
    net : nnvm.Symbol
        The computational graph

    params : dict of str to NDArray
        The parameters.
    """
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    params = {}
    g = graph.create(net)
    input_shapes, _ = graph_util.infer_shape(g, data=data_shape)
    shape_dict = dict(zip(g.index.input_names, input_shapes))
    for k, v in shape_dict.items():
        if k == "data":
            continue
        # Specially generate non-negative parameters.
        if k.endswith("gamma"):
            init = np.random.uniform(0.9, 1, size=v)
        elif k.endswith("var"):
            init = np.random.uniform(0.9, 1, size=v)
        else:
            init = np.random.uniform(-0.1, 0.1, size=v)
        params[k] = tvm.nd.array(init.astype(dtype), ctx=tvm.cpu(0))
    return net, params
