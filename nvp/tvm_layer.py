import tvm
from tvm import relay
from tvm.relay.testing import init

def gen_module(layer):
    if layer == 'relu':
        data = relay.var("data", shape=[1, 100], dtype="float32")
        Trelay = tvm.relay.nn.relu(data)
    elif layer == 'leaky_relu':
        data = relay.var("data", shape=[1, 100], dtype="float32")
        Trelay = tvm.relay.nn.leaky_relu(data)
    elif layer == 'softmax':
        data = relay.var("data", shape=[1, 100], dtype="float32")
        Trelay = tvm.relay.nn.softmax(data)
    elif layer == 'conv2d':
        data = relay.var("data", shape=[1, 10, 5, 5], dtype="float32") #NCHW
        weight = relay.var("weight", shape=[7, 10, 3, 3], dtype="float32") #OIHW
        Trelay = tvm.relay.nn.conv2d(data, weight, strides=(1,1), padding=(0,0))
    elif layer == 'dwconv_3x3':
        data = relay.var("data", shape=[1, 10, 5, 5], dtype="float32") #NCHW8c
        weight = relay.var("weight", shape=[7, 10, 3, 3], dtype="float32") #OIHW
        Trelay = tvm.relay.nn.conv2d(data, weight, strides=(1,1), padding=(0,0))
    else:
        raise NotImplementedError("Currently NOT implemented: %s" %(layer))

    args = relay.analysis.free_vars(Trelay)   # tvm.relay.Var
    mod = tvm.IRModule.from_expr(Trelay)
    func = relay.Function(args, Trelay)
    # mod = tvm.IRModule.from_expr(func)
    mod, params = init.create_workload(func)
    return mod
