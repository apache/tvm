import tvm
from tvm import relay
from tvm.relay.testing import init

def gen_module(layer):
    if layer == 'relu':
        (cw, oh, ow) = (32, 10, 12)
        data = relay.var("data", shape=[1, cw*oh*ow], dtype="float32")
        Trelay = tvm.relay.nn.relu(data)
    elif layer == 'leaky_relu':
        data = relay.var("data", shape=[1, 100], dtype="float32")
        Trelay = tvm.relay.nn.leaky_relu(data)
    elif layer == 'softmax':
        data = relay.var("data", shape=[1, 100], dtype="float32")
        Trelay = tvm.relay.nn.softmax(data)
    elif layer == 'maxpool':
        data = relay.var("data", shape=[1, 10, 5, 5], dtype="float32") #NCHW
        Trelay = tvm.relay.nn.max_pool2d(data, pool_size=(3,3))
    elif layer == 'dwconv_3x3': # weight(I) == data(C)/groups
        (cw, oh, ow) = (32, 10, 12)
        data = relay.var("data", shape=[1, cw, oh+2, ow+2], dtype="float32") #NCHW
        weight = relay.var("weight", shape=[cw, 1, 3, 3], dtype="float32") #OIHW
        Trelay = tvm.relay.nn.conv2d(data, weight, strides=(1,1), padding=(0,0), groups=cw)
    elif layer == 'dense':
        data = relay.var("data", shape=[5, 10], dtype="float32") #(d_1, d_2, ..., units_in)
        weight = relay.var("weight", shape=[7,10], dtype="float32") #(units, units_in)
        Trelay = tvm.relay.nn.dense(data, weight)
    elif layer == 'corr':
        data1 = relay.var("data", shape=[1, 8, 5, 5], dtype="float32") #NCHW
        data2 = relay.var("data", shape=[1, 8, 5, 5], dtype="float32") #NCHW
        Trelay = tvm.relay.nn.correlation(data1, data2, kernel_size=3, max_displacement=5,
                                    stride1=1, stride2=1, padding=0, is_multiply=True, layout="NCHW")
    else:
        raise NotImplementedError("Currently NOT implemented: %s" %(layer))

    args = relay.analysis.free_vars(Trelay)   # tvm.relay.Var
    # mod = tvm.IRModule.from_expr(Trelay)
    func = relay.Function(args, Trelay)
    # mod = tvm.IRModule.from_expr(func)
    mod, params = init.create_workload(func)
    return mod
