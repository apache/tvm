"""Utility for benchmark"""

import sys
import nnvm

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network
    
    Parameters
    ----------
    name: str
        The name of the network, can be 'resnet-18', 'resnet-50', 'vgg-16', 'inception_v3', 'mobilenet', ...
    batch_size:
        batch size

    Returns
    -------
    net: nnvm.symbol
        The NNVM symbol of network definition
    params: dict
        The random parameters for benchmark
    input_shape: tuple
        The shape of input tensor
    output_shape: tuple
        The shape of output tensor
    """
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if name == 'mobilenet':
        net, params = nnvm.testing.mobilenet.get_workload(batch_size=batch_size)
    elif name == 'mobilenet_v2':
        net, params = nnvm.testing.mobilenet_v2.get_workload(batch_size=batch_size)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        net, params = nnvm.testing.inception_v3.get_workload(batch_size=batch_size)
    elif "resnet" in name:
        n_layer = int(name.split('-')[1])
        net, params = nnvm.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        net, params = nnvm.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size)
    elif "densenet" in name:
        n_layer = int(name.split('-')[1])
        net, params = nnvm.testing.densenet.get_workload(num_layers=n_layer, batch_size=batch_size)
    elif "squeezenet" in name:
        version = name.split("_v")[1]
        net, params = nnvm.testing.squeezenet.get_workload(batch_size=batch_size, version=version)
    elif name == 'custom':
        # an example for custom network
        from nnvm.testing import utils
        net = nnvm.sym.Variable('data')
        net = nnvm.sym.conv2d(net, channels=4, kernel_size=(3,3), padding=(1,1))
        net = nnvm.sym.flatten(net)
        net = nnvm.sym.dense(net, units=1000)
        net, params = utils.create_workload(net, batch_size, (3, 224, 224))
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        net, params = nnvm.frontend.from_mxnet(block)
        net = nnvm.sym.softmax(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return net, params, input_shape, output_shape

def print_progress(msg):
    """print progress message
    
    Parameters
    ----------
    msg: str
        The message to print
    """
    sys.stdout.write(msg + "\r")
    sys.stdout.flush()
