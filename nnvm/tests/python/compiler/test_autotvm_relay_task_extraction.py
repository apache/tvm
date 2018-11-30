"""Test task extraction for autotvm"""
import tvm.relay.testing
from tvm import relay
from tvm import autotvm

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)

    if name == 'resnet-18':
        net, params = relay.testing.resnet.get_workload(num_layers=18, batch_size=batch_size)
    elif name == 'mobilenet':
        net, params = relay.testing.mobilenet.get_workload(batch_size=batch_size)
    elif name == 'dcgan':
        net, params = relay.testing.dcgan.get_workload(batch_size=batch_size)
        input_shape = (batch_size, 100)
    else:
        raise ValueError("Unsupported network: " + name)

    return net, params, input_shape

def test_task_extraction():
    target = 'llvm'

    net, params, input_shape = get_network('resnet-18', batch_size=1)
    tasks = autotvm.task.extract_from_program(net, target=target,
                                            params=params,
                                            ops=(relay.op.nn.conv2d,))
    assert len(tasks) == 12

    net, params, input_shape = get_network('resnet-18', batch_size=1)
    tasks = autotvm.task.extract_from_program(net, target=target,
                                            params=params,
                                            ops=(relay.op.nn.dense,))
    assert len(tasks) == 1

    net, params, input_shape = get_network('resnet-18', batch_size=1)
    tasks = autotvm.task.extract_from_program(net, target=target,
                                            params=params,
                                            ops=(relay.op.nn.conv2d, relay.op.nn.dense))
    assert len(tasks) == 13

    net, params, input_shape = get_network('mobilenet', batch_size=1)
    tasks = autotvm.task.extract_from_program(net, target=target,
                                            params=params,
                                            ops=(relay.op.nn.conv2d, relay.op.nn.dense))
    assert len(tasks) == 20

    net, params, input_shape = get_network('dcgan', batch_size=1)
    tasks = autotvm.task.extract_from_program(net, target=target,
                                            params=params,
                                            ops=(relay.op.nn.conv2d_transpose,))
    assert len(tasks) == 4

if __name__ == '__main__':
    test_task_extraction()
