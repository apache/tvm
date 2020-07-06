import tvm
from tvm import relay
from tvm import hago
import numpy as np

def create_hardware():
    hardware = hago.Hardware()
    hardware['nn.dense'].append(hago.OpDesc(idtypes=['int8', 'int8'], odtypes=['int32']))
    return hardware


def test_dense(ishape=(32, 16), wshape=(10, 16), batch_num=3):
    data = relay.var('data', shape=ishape)
    weight = relay.var('weight', shape=wshape)
    out = relay.nn.dense(data, weight)
    func = relay.Function([data, weight], out)

    weight_np = np.random.rand(*wshape).astype('float32')

    # generate dataset
    dataset = []
    for i in range(batch_num):
        data_np = np.random.rand(*ishape).astype('float32')
        ex = relay.create_executor("debug", ctx=ctx, target=target)
        out_np = ex.evaluate(func)(data_np, weight_np).asnumpy()
        pred_np = np.argmax(out_np, axis=1)
        dataset.append({'data': tvm.nd.array(data_np), 'label': tvm.nd.array(pred_np)})

    params = {'weight': tvm.nd.array(weight_np)}
    return func, params, dataset

def test_conv2d():
    pass

device = 'gpu'
if device == 'cpu':
    target = 'llvm'
    ctx = tvm.cpu()
elif device == 'gpu':
    target = 'cuda'
    ctx = tvm.gpu(1)
# prepared by user
hardware = create_hardware()
func, params, dataset = test_dense() 

qconfig = hago.qconfig(skip_conv_layers=[0],
                       log_file='temp.log',
                       threshold_estimate_method="power_of_two_range")
with qconfig:
    func = hago.prerequisite_optimize(func, params)
    space = hago.generate_search_space(func, hardware)
    tuner = hago.BatchedGreedySearchTuner(space, 'accuracy')
    strategy, result = hago.search_quantize_strategy(func, hardware, dataset, tuner, ctx, target)
    quantizer = hago.create_quantizer(func, hardware, strategy)
    simulated_graph = quantizer.simulate()
    quantized_graph = quantizer.quantize()
    print(strategy)
    print(result)
    print(simulated_graph)
    print(quantized_graph)
