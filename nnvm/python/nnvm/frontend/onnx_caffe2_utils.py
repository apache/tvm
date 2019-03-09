"""Util functions shared by the ONNX and Caffe2 frontends."""
from __future__ import absolute_import as _abs
from nnvm import graph as _graph
from nnvm.compiler import graph_util


def dimension_picker(prefix, surfix=''):
    def _impl(attr):
        kernel = attr['kernel_shape']
        if len(kernel) == 2:
            return prefix + '2d' + surfix
        raise NotImplementedError("Only 2d kernel supported.")

    return _impl


def dimension_constraint():
    def _dim_check(attrs):
        if len(attrs['kernel_shape']) == 2:
            return True
        return False

    return _dim_check, "Only 2d kernel supported."


def infer_channels(inputs, params, transpose=False):
    """A hack for getting 'channels' or 'units' since caffe2 don't provide
    these attributes. We check the shape of weights provided to get the number.
    """
    g = _graph.create(inputs)
    shape_dict = {k: v.shape for k, v in params.items()}
    _, out_shapes = graph_util.infer_shape(g, **shape_dict)
    channels = out_shapes[0][0] if not transpose else out_shapes[0][1]
    return channels


def revert_caffe2_pad(pads):
    """Caffe2 require two times the normal padding."""
    if len(pads) == 4:
        pads = pads[:2]
    elif len(pads) == 2:
        pass
    else:
        raise ValueError("Invalid caffe2 type padding: {}".format(pads))
    return pads
