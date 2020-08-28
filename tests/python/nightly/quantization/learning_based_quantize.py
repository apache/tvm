# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from collections import namedtuple
import tvm
from tvm import relay
from tvm import hago
import mxnet as mx
import numpy as np
from mxnet import gluon
import logging
import os
import pickle

logging.basicConfig(level=logging.DEBUG)

Config = namedtuple('Config', ['model', 'expected_acc'])


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
        shuffle             = True,
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


def get_model(model_name, batch_size, qconfig, original=False, simulated=False, dataset=None):
    gluon_model = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
    img_size = 299 if model_name == 'inceptionv3' else 224
    data_shape = (batch_size, 3, img_size, img_size)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})

    graph = hago.prerequisite_optimize(mod['main'], params=params)
    logging.debug('original')
    logging.debug(graph.astext(show_meta_data=False))

    if original:
        return graph

    with qconfig:
        logging.debug('current quantize config')
        logging.debug(hago.current_qconfig())
        hardware = hago.create_accelerator_description()
        space = hago.generate_search_space(graph, hardware)
        # tuner = hago.BatchedGreedySearchTuner(space, 'accuracy')
        tuner = hago.DefaultSetting(space, 'accuracy')
        ctx = tvm.gpu()
        target = 'cuda'
        strategy, result = hago.search_quantize_strategy(graph, hardware, dataset, tuner, ctx, target)

        quantizer = hago.create_quantizer(graph, hardware, strategy)
        simulated_graph = quantizer.simulate()
        quantized_graph = quantizer.quantize()
        logging.debug('simulated graph')
        logging.debug(simulated_graph.astext(show_meta_data=False))
        logging.debug('quantize graph')
        logging.debug(quantized_graph.astext(show_meta_data=False))
        # hago.inspect_graph_statistic(graph, hardware, strategy, dataset, ctx, target)
        return quantized_graph


def eval_acc(mod, dataset, batch_fn, target='cuda', ctx=tvm.gpu(), log_interval=100):
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target)
    # create runtime module
    m = tvm.contrib.graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)

    # setup evaluaiton metric
    dataset.reset()
    batch_size = dataset.batch_size
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    # Execute
    for i, batch in enumerate(dataset):
        data, label = batch_fn(batch, [mx.cpu(0)])
        m.run(data=data[0].asnumpy())
        out_arr = m.get_output(0)
        acc_top1.update(label, [mx.nd.array(out_arr.asnumpy())])
        acc_top5.update(label, [mx.nd.array(out_arr.asnumpy())])

        if not (i + 1) % log_interval:
            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            nsamples = (i + 1) * batch_size
            logging.info('[%d samples] validation: acc-top1=%f acc-top5=%f', nsamples, top1, top5)
    logging.info('[final] validation: acc-top1=%f acc-top5=%f', top1, top5)
    return top1


def get_calibration_dataset(dataset, batch_fn, num_samples=100):
    dataset.reset()
    ret = []
    for i, batch in enumerate(dataset):
        if i * dataset.batch_size > num_samples:
            break
        data, label = batch_fn(batch, [mx.cpu(0)])
        ret.append({'data': tvm.nd.array(data[0].asnumpy()),
                    'label': tvm.nd.array(label[0].asnumpy())})
    return ret


def test_quantize_acc(cfg, rec_val):
    qconfig = hago.qconfig(skip_conv_layers=[0],
                           log_file='temp.log')

    batch_size = 32
    val_data, batch_fn = get_val_data(cfg.model, rec_val=rec_val, batch_size=batch_size)
    dataset = get_calibration_dataset(val_data, batch_fn)

    for orig in [True, False]:
        mod = get_model(cfg.model, batch_size, qconfig, dataset=dataset, original=orig)
        acc = eval_acc(mod, val_data, batch_fn, target='cuda', ctx=tvm.gpu())
        print("Final accuracy", "int8" if orig else "fp32", acc)
    return acc


if __name__ == "__main__":
    #TODO(for user): replace the line with the path to imagenet validation dataset
    rec_val = "~/tensorflow_datasets/downloads/manual/imagenet2012/val_rec.rec"

    results = []
    configs = [
        Config('resnet18_v1', expected_acc=0.67),
        # Config('resnet50_v1', expected_acc=0.67),
        # Config('inceptionv3', expected_acc=0.67),
    ]
    # rec = hago.pick_best(".quantize_strategy_search.log", 'quant_acc')

    for config in configs:
        acc = test_quantize_acc(config, rec_val)
        results.append((config, acc))
    for res in results:
        print(res)
