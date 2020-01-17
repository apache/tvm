import tvm
from tvm import relay
from tvm import hago
import mxnet as mx
from mxnet import gluon
import os
import numpy as np


def get_val_data(model_name,
                 rec_val,
                 batch_size,
                 num_workers=4):
    rec_val = os.path.expanduser(rec_val)
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    img_size = 299 if model_name == 'inceptionv3' else 224
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        preprocess_threads  = num_workers,
        shuffle             = False,
        batch_size          = batch_size,
        resize              = 256,
        data_shape          = (3, img_size, img_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return val_data, batch_fn


def get_calibration_dataset(dataset, batch_fn, num_samples=100):
    dataset.reset()
    calib = []
    for i, batch in enumerate(dataset):
        if i * dataset.batch_size > num_samples:
            break
        data, label = batch_fn(batch, [mx.cpu(0)])
        calib.append({'data': data[0].asnumpy(), 'label': label[0].asnumpy()})
    return calib


def get_model(name):
    batch_size = 32
    gluon_model = gluon.model_zoo.vision.get_model(name, pretrained=True)
    img_size = 299 if name == 'inceptionv3' else 224
    data_shape = (batch_size, 3, img_size, img_size)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})
    return mod, params

def get_partial_net(params, break_point=1):
    def p(name):
        return relay.const(params[name])
    def to_break(node):
        if to_break.cnt == break_point:
            to_break.out = node
        to_break.cnt += 1
        return
    to_break.cnt = 0
    to_break.out = None

    data = relay.var("data", shape=[32, 3, 224, 224])
    to_break(data)

    conv = relay.nn.conv2d(data, p('resnetv10_conv0_weight'), strides=[2, 2], padding=[3, 3], channels=64, kernel_size=[7, 7])
    to_break(conv)

    bn = relay.nn.batch_norm(conv,
                             p('resnetv10_stage1_batchnorm1_gamma'),
                             p('resnetv10_stage1_batchnorm1_beta'),
                             p('resnetv10_stage1_batchnorm1_running_mean'),
                             p('resnetv10_stage1_batchnorm1_running_var'))
    to_break(bn[0])

    relu = relay.nn.relu(bn[0])
    to_break(relu)

    max_pool2d = relay.nn.max_pool2d(relu, pool_size=[3, 3], strides=[2, 2], padding=[1, 1])
    to_break(max_pool2d)

    conv = relay.nn.conv2d(max_pool2d, p('resnetv10_stage1_conv0_weight'), padding=[1, 1], channels=64, kernel_size=[3, 3])
    to_break(conv)

    func = relay.Function([data], to_break.out)
    return func

model_name = 'resnet18_v1'
rec_val = "/scratch/tqchen/imagenet/val.rec"

mod, params = get_model(model_name)
print('resent:')
print(mod['main'])

net = get_partial_net(params, break_point=5)
print(net)

val_data, batch_fn = get_val_data(model_name, rec_val=rec_val, batch_size=32)
calib = get_calibration_dataset(val_data, batch_fn)
sample = calib[0]


def evaluate(graph):
    with relay.build_config(opt_level=2):
        graph, lib, params = relay.build_module.build(graph, target='llvm')
    runtime = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu())
    runtime.set_input('data', sample['data'])
    runtime.set_input(**params)
    runtime.run()
    out = runtime.get_output(0).asnumpy()
    return out

real_out = evaluate(net)

with hago.qconfig(search_strategy='default_setting'):
    hardware = hago.create_accelerator_description()
    mod = relay.Module.from_expr(net)
    mod = hago.prerequisite_optimize(mod)
    strategy, acc = hago.search_quantize_strategy(mod, hardware, dataset=calib)
    quantizer = hago.create_quantizer(mod['main'], hardware, strategy)
    simulated_graph = quantizer.simulate()
    print('after simulate')
    print(simulated_graph.astext(show_meta_data=False))
    simulated_out = evaluate(simulated_graph)
    quantized_graph = quantizer.quantize()
    print('after quantize')
    print(quantized_graph.astext(show_meta_data=False))
    quantized_out = evaluate(quantized_graph)
    print('output shape: ', real_out.shape)
    hago.compare(real_out, simulated_out)
    hago.compare(real_out, quantized_out)
    # print('maximum absoluate error:')
    # err = np.abs(quant_out - real_out)
    # idx = np.unravel_index(np.argmax(err, axis=None), err.shape)
    # print("{:.2f}, while real output is {:.2f}, and quant output is {:.2f}".format(np.max(err), real_out[idx], quant_out[idx]))
    
