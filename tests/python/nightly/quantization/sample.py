import tvm
from tvm import relay
from tvm import hago
import numpy as np

def create_hardware():
    hardware = hago.Hardware()
    hardware['nn.dense'].append(hago.OpDesc(idtypes=['int8', 'int8'], odtypes=['int32']))
    return hardware

target = 'llvm'
ctx = tvm.cpu()
hardware = create_hardware()

ishape = (1, 16)
wshape = (10, 16)

data = relay.var('data', shape=ishape)
weight = relay.var('weight', shape=wshape)
out = relay.nn.dense(data, weight)
func = relay.Function([data, weight], out)

data_np = np.random.rand(*ishape).astype('float32')
weight_np = np.random.rand(*wshape).astype('float32')
out_np = np.matmul(data_np, weight_np.transpose())
pred_np = np.argmax(out_np, axis=1)

mod = tvm.IRModule()
mod['main'] = func
mod = hago.prerequisite_optimize(mod, {'weight': tvm.nd.array(weight_np)})
ex = relay.create_executor("debug", mod=mod, ctx=ctx, target=target)
out_np = ex.evaluate()(data_np)
print('real out')
print(out_np)
dataset = [{'data': data_np, 'label': pred_np}]
strategy, acc = hago.batched_search_quantize_strategy(mod, hardware, dataset=dataset)

# graph = relay.create_executor('graph')
# print(pred_np)
# out = graph.evaluate(func)(data_np)
# print(out.asnumpy())

