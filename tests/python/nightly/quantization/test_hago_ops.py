import tvm
from tvm import relay
from tvm import hago
import numpy as np
from common_utils import target_and_ctx

def create_hardware():
    hardware = hago.Hardware()
    hardware.add_op_desc('concatenate', hago.OpDesc(in_dtypes='float32', out_dtypes='float32'))
    hardware.add_op_desc('concatenate', hago.OpDesc(in_dtypes='int8', out_dtypes='int8'))
    hardware.add_op_desc('concatenate', hago.OpDesc(in_dtypes='int32', out_dtypes='int32'))
    hardware.add_op_desc('add', hago.OpDesc(in_dtypes='int8', out_dtypes='int32'))
    hardware.add_op_desc('nn.dense', hago.OpDesc(in_dtypes='int8', out_dtypes='int32'))
    hardware.add_op_desc('nn.conv2d', hago.OpDesc(in_dtypes='int8', out_dtypes='int32'))
    return hardware


def test_dense(ishape=(8, 16), wshape=(10, 16), batch_num=5, device='cpu'):
    target, ctx = target_and_ctx(device)
    data = relay.var('data', shape=ishape)
    weight = relay.var('weight', shape=wshape)
    out = relay.nn.dense(data, weight)
    func = relay.Function([data, weight], out)

    # weight_np = np.random.rand(*wshape).astype('float32')
    weight_np = np.random.normal(size=wshape).astype('float32')
    
    # generate dataset
    batches = []
    for i in range(batch_num):
        data_np = np.random.rand(*ishape).astype('float32')
        ex = relay.create_executor("debug", ctx=ctx, target=target)
        out_np = ex.evaluate(func)(data_np, weight_np).asnumpy()
        pred_np = np.argmax(out_np, axis=1)
        batches.append({'data': tvm.nd.array(data_np), 'label': tvm.nd.array(pred_np)})
    dataset = hago.CalibrationDataset(batches)

    params = {'weight': tvm.nd.array(weight_np)}
    return func, params, dataset


def test_concatenate(ishape=(8, 16), wshape=(10, 16), batch_num=3, device='cpu'):
    target, ctx = target_and_ctx(device)
    w0shape = wshape 
    w1shape = (wshape[1], wshape[0])
    data_a = relay.var('data_a', shape=ishape)
    data_b = relay.var('data_b', shape=ishape)
    data_c = relay.var('data_c', shape=ishape)
    data_d = relay.var('data_d', shape=ishape)
    weight0 = relay.var('weight0', shape=w0shape)
    weight1 = relay.var('weight1', shape=w0shape)
    weight2 = relay.var('weight2', shape=w0shape)
    weight3 = relay.var('weight3', shape=w0shape)
    dense_a = relay.nn.dense(data_a, weight0)
    dense_b = relay.nn.dense(data_b, weight1)
    dense_c = relay.nn.dense(data_c, weight2)
    dense_d = relay.nn.dense(data_d, weight3)
    concat = relay.concatenate([dense_a, dense_b, dense_c, dense_d], axis=0)
    # 32, 10
    weight5 = relay.var('weight5', shape=w1shape)
    out = relay.nn.dense(concat, weight5)
    func = relay.Function([data_a, data_b, data_c, data_d, weight0, weight1, weight2, weight3, weight5], out)

    weight0_np = np.random.normal(size=w0shape).astype('float32')
    weight1_np = np.random.normal(size=w0shape).astype('float32')
    weight2_np = np.random.normal(size=w0shape).astype('float32')
    weight3_np = np.random.normal(size=w0shape).astype('float32')
    weight5_np = np.random.normal(size=w1shape).astype('float32')

    # generate dataset
    batches = []
    for i in range(batch_num):
        data_a_np = np.random.rand(*ishape).astype('float32')
        data_b_np = np.random.rand(*ishape).astype('float32')
        data_c_np = np.random.rand(*ishape).astype('float32')
        data_d_np = np.random.rand(*ishape).astype('float32')
        ex = relay.create_executor("debug", ctx=ctx, target=target)
        out_np = ex.evaluate(func)(data_a_np, data_b_np, data_c_np, data_d_np, weight0_np,
                weight1_np, weight2_np, weight3_np, weight5_np).asnumpy()
        pred_np = np.argmax(out_np, axis=1)
        batches.append({'data_a': tvm.nd.array(data_a_np),
                        'data_b': tvm.nd.array(data_b_np),
                        'data_c': tvm.nd.array(data_c_np),
                        'data_d': tvm.nd.array(data_d_np),
                        'label': tvm.nd.array(pred_np)})
    dataset = hago.CalibrationDataset(batches)

    params = {'weight0': tvm.nd.array(weight0_np),
              'weight1': tvm.nd.array(weight0_np),
              'weight2': tvm.nd.array(weight0_np),
              'weight3': tvm.nd.array(weight0_np),
              'weight5': tvm.nd.array(weight5_np)}
    return func, params, dataset

def test_conv2d(ishape=(1,3,224,224), wshape=(32,3,5,5), batch_num=5, device='cpu'):
    target, ctx = target_and_ctx(device)
    data = relay.var('data', shape=ishape)
    weight = relay.var('weight', shape=wshape)
    out = relay.nn.conv2d(data, weight, kernel_size=(5,5))
    func = relay.Function([data, weight], out)

    weight_np = np.random.normal(size=wshape).astype('float32')

    # generate dataset
    batches = []
    for i in range(batch_num):
        data_np = np.random.rand(*ishape).astype('float32')
        ex = relay.create_executor("debug", ctx=ctx, target=target)
        out_np = ex.evaluate(func)(data_np, weight_np).asnumpy()
        pred_np = np.argmax(out_np, axis=1)
        batches.append({'data': tvm.nd.array(data_np), 'label': tvm.nd.array(pred_np)})
    dataset = hago.CalibrationDataset(batches)

    params = {'weight': tvm.nd.array(weight_np)}
    return func, params, dataset

def test_add(ishape=(32,32), wshape=(32,32), batch_num=5, device='cpu'):
    target, ctx = target_and_ctx(device)
    data = relay.var('data', shape=ishape)
    weight = relay.var('weight', shape=wshape)
    out = data + weight
    func = relay.Function([data, weight], out)

    weight_np = np.random.normal(size=wshape).astype('float32')

    # generate dataset
    batches = []
    for i in range(batch_num):
        data_np = np.random.rand(*ishape).astype('float32')
        ex = relay.create_executor("debug", ctx=ctx, target=target)
        out_np = ex.evaluate(func)(data_np, weight_np).asnumpy()
        pred_np = np.argmax(out_np, axis=1)
        batches.append({'data': tvm.nd.array(data_np), 'label': tvm.nd.array(pred_np)})
    dataset = hago.CalibrationDataset(batches)

    params = {'weight': tvm.nd.array(weight_np)}
    return func, params, dataset

def check_results(func, params, dataset, device='cpu'):
    original_func = func
    target, ctx = target_and_ctx(device)
    # prepared by user
    hardware = create_hardware()

    func = hago.prerequisite_optimize(func, params)
    print('after optimize')
    print(func)
    space = hago.generate_search_space(func, hardware)
    # tuner = hago.BatchedGreedySearchTuner(space, 'accuracy')
    tuner = hago.DefaultSetting(space, 'accuracy')
    strategy, result = hago.search_quantize_strategy(func, hardware, dataset, tuner, ctx, target)
    quantizer = hago.create_quantizer(func, hardware, strategy)
    simulated_graph = quantizer.simulate()
    quantized_graph = quantizer.quantize()
    print(strategy)
    print(result)
    print(simulated_graph)
    print("Quantized graph", quantized_graph)

    input_data = [ {**data, **params} for data in dataset.batches ]
    expected_out = hago.base.evaluate(original_func, input_data)[0]
    simulated_out = hago.base.evaluate(simulated_graph, input_data)[0]

    for exp_out, sim_out in zip(expected_out, simulated_out):
        hago.analysis.compare(exp_out, sim_out)

if __name__ == '__main__':
    qconfig = hago.qconfig(log_file='temp.log',
                           use_channel_quantize=1,
                           threshold_estimate_method="avg_range")
    with qconfig:
        device = 'cpu'
        func, params, dataset = test_concatenate(device=device)
        print('original model:')
        print(func)
        check_results(func, params, dataset, device)
