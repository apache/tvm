"""
DenseNet, load model from gluon model zoo

Reference:
Huang, Gao, et al. "Densely Connected Convolutional Networks." CVPR 2017
"""

from .utils import create_workload
from ..frontend.mxnet import _from_mxnet_impl

def get_workload(batch_size, num_classes=1000, num_layers=121, dtype="float32"):
    """Get benchmark workload for mobilenet

    Parameters
    ----------
    batch_size : int
        The batch size used in the model

    num_classes : int, optional
        Number of classes

    num_layers : int, optional
        The number of layers

    dtype : str, optional
        The data type

    Returns
    -------
    net : nnvm.Symbol
        The computational graph

    params : dict of str to NDArray
        The parameters.
    """
    import mxnet as mx
    from mxnet.gluon.model_zoo.vision import get_model

    image_shape = (1, 3, 224, 224)

    block = get_model('densenet%d' % num_layers, classes=num_classes, pretrained=False)

    data = mx.sym.Variable('data')
    sym = block(data)
    sym = mx.sym.SoftmaxOutput(sym)

    net = _from_mxnet_impl(sym, {})

    return create_workload(net, batch_size, image_shape[1:], dtype)
