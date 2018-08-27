"""
MobileNetV2, load model from gluon model zoo

Reference:
Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation
https://arxiv.org/abs/1801.04381
"""

from .utils import create_workload
from ..frontend.mxnet import _from_mxnet_impl

def get_workload(batch_size, num_classes=1000, multiplier=1.0, dtype="float32"):
    """Get benchmark workload for mobilenet

    Parameters
    ----------
    batch_size : int
        The batch size used in the model

    num_classes : int, optional
        Number of classes

    multiplier : tuple, optional
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
    import mxnet as mx
    from mxnet.gluon.model_zoo.vision.mobilenet import MobileNetV2

    image_shape = (1, 3, 224, 224)

    block = MobileNetV2(multiplier=multiplier, classes=num_classes)

    data = mx.sym.Variable('data')
    sym = block(data)
    sym = mx.sym.SoftmaxOutput(sym)

    net = _from_mxnet_impl(sym, {})

    return create_workload(net, batch_size, image_shape[1:], dtype)
