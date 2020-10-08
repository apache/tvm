import numpy as np
import tvm 
from tvm import hago
from tvm import relay
from common_hago import *

def test_simulated_quantize():
    target, ctx = target_and_ctx('cpu')
    ishape = [3, 4, 5]
    axis = 1

    # init data
    threshold = [float(2**i) for i in range(ishape[axis])]
    data_np = np.zeros(ishape).astype('float32')

    data_np = np.moveaxis(data_np, axis, -1)
    sub_shape = list(data_np.shape)
    sub_shape[-1] = 1

    index = []
    for dim_size in sub_shape: 
        index.append(slice(0, dim_size))
    index = tuple(index)

    for i in range(ishape[axis]):
        data_np[index] = np.random.uniform(-threshold[i], threshold[i], sub_shape).astype('float32')
    data_np = np.moveaxis(data_np, -1, axis)

    in_scale = 1.0
    out_scale = np.array([thold / 128 for thold in threshold]).astype('float32')

    data = relay.var('data', shape=ishape, dtype="float32")
    out = hago.quantize.simulated_quantize(data,
                                           in_scale=in_scale,
                                           out_scale=out_scale,
                                           clip_min=-127,
                                           clip_max=127,
                                           in_dtype='float32',
                                           out_dtype='int8',
                                           axis=axis)

    func = relay.Function([data], out)
    print(func)
    data_np = np.random.rand(*ishape).astype('float32')
    ex = relay.create_executor("debug", ctx=ctx, target=target)
    out_np = ex.evaluate(func)(data_np).asnumpy()
    print(data_np)
    print(out_np)


if __name__ == '__main__':
    test_simulated_quantize()
    # device = 'cpu'
    # func, params, dataset = test_add(device=device)
    # print('original model:')
    # print(func)
    # check_results(func, params, dataset, device)
