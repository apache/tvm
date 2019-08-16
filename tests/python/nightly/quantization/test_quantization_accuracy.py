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
from tvm import te
from tvm import relay
from tvm.relay import quantize as qtz
import mxnet as mx
import numpy as np
from mxnet import gluon
import logging
import os

logging.basicConfig(level=logging.DEBUG)

Config = namedtuple('Config', ['model', 'nbit_input',  'dtype_input', 'nbit_output', 'dtype_output', 'global_scale', 'expected_acc'])


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


def get_model(model_name, batch_size, qconfig, target=None, original=False, simulated=False, calib_set=None):
    gluon_model = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
    img_size = 299 if model_name == 'inceptionv3' else 224
    data_shape = (batch_size, 3, img_size, img_size)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})

    with tvm.transform.PassContext(opt_level=3):
        qmod = relay.quantize.prerequisite_optimize(mod, params=params)
    logging.debug('original')
    logging.debug(qmod['main'].astext(show_meta_data=False))

    def visit(e):
        if isinstance(e, tvm.relay.Call):
            print(e.op.name)
            for var in e.args:
                if isinstance(var, tvm.relay.Constant):
                    print(np.max(var.data.asnumpy()))
    relay.analysis.post_order_visit(qmod['main'], visit)

    if original:
        return qmod

    with qconfig:
        logging.debug('current quantize config')
        logging.debug(qtz.current_qconfig())
        qmod = qtz.quantize(qmod, dataset=calib_set)
        logging.debug('after quantize')
        logging.debug(qmod['main'].astext(show_meta_data=False))
    return qmod


def eval_acc(model, dataset, batch_fn, target=tvm.target.cuda(), ctx=tvm.gpu(), log_interval=100):
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(model, target)
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
    for i, batch in enumerate(dataset):
        if i * dataset.batch_size > num_samples:
            break
        data, label = batch_fn(batch, [mx.cpu(0)])
        yield {'data': data[0].asnumpy()}


def test_quantize_acc(cfg, rec_val):
    qconfig = qtz.qconfig(skip_conv_layers=[],
                          nbit_input=cfg.nbit_input,
                          nbit_weight=cfg.nbit_input,
                          dtype_input=cfg.dtype_input,
                          dtype_weight=cfg.dtype_input,
                          dtype_activation=cfg.dtype_output,
                          global_scale=cfg.global_scale,
                          do_simulation=False,
                          debug_enabled_ops=None)

    val_data, batch_fn = get_val_data(cfg.model, rec_val=rec_val, batch_size=32)
    calib_set = get_calibration_dataset(val_data, batch_fn)
    # calib_set = None

    mod = get_model(cfg.model, 32, qconfig, tvm.target.cuda(), calib_set=calib_set)
    acc = eval_acc(mod, val_data, batch_fn)
    assert acc > cfg.expected_acc
    return acc


if __name__ == "__main__":
    #TODO(for user): replace the line with the path to imagenet validation dataset
    rec_val = "/data/datasets/imagenet/rec/val.rec"

    results = []
    configs = [
        # resnet18_v1 best configuration
        # Config('resnet18_v1', nbit_input=8, dtype_input='int8', nbit_output=16, dtype_output='int16', global_scale=8.0, expected_acc=0.675),
        Config('resnet18_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=8.0, expected_acc=0.674),
        # Config('resnet34_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=8.0, expected_acc=0.714),
        # Config('resnet50_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=4.0, expected_acc=0.743),
        # Config('resnet101_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=2.0, expected_acc=0.751),

        # # resnet18_v2 best configuration
        # Config('resnet18_v2', nbit_input=8, dtype_input='int8', nbit_output=16, dtype_output='int16', global_scale=4.0, expected_acc=0.611),
        # Config('resnet18_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=4.0, expected_acc=0.612),
        # Config('resnet34_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=4.0, expected_acc=0.726),
        # Config('resnet50_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=2.0, expected_acc=0.752),
        # Config('resnet101_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=2.0, expected_acc=0.765),

        # resnet18_v1 history
        # Config('resnet18_v1', nbit_input=8, dtype_input='int8', nbit_output=16, dtype_output='int16', global_scale=2.0, expected_acc=0.000),
        # Config('resnet18_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=2.0, expected_acc=0.401),
        # Config('resnet34_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=2.0, expected_acc=0.259),
        # Config('resnet50_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=2.0, expected_acc=0.738),
        # Config('resnet101_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=2.0, expected_acc=0.751),

        # Config('resnet18_v1', nbit_input=8, dtype_input='int8', nbit_output=16, dtype_output='int16', global_scale=4.0, expected_acc=0.367),
        # Config('resnet18_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=4.0, expected_acc=0.672),
        # Config('resnet34_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=4.0, expected_acc=0.699),
        # Config('resnet50_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=4.0, expected_acc=0.743),
        # Config('resnet101_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=4.0, expected_acc=0.759),

        # Config('resnet18_v1', nbit_input=8, dtype_input='int8', nbit_output=16, dtype_output='int16', global_scale=8.0, expected_acc=0.675),
        # Config('resnet18_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=8.0, expected_acc=0.674),
        # Config('resnet34_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=8.0, expected_acc=0.714),
        # Config('resnet50_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=8.0, expected_acc=0.696),
        # Config('resnet101_v1', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=8.0, expected_acc=0.713),

        # resnet18_v2 history
        # Config('resnet18_v2', nbit_input=8, dtype_input='int8', nbit_output=16, dtype_output='int16', global_scale=2.0, expected_acc=0.250)
        # Config('resnet18_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=2.0, expected_acc=0.454),
        # Config('resnet34_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=2.0, expected_acc=0.459),
        # Config('resnet50_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=2.0, expected_acc=0.752),
        # Config('resnet101_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=2.0, expected_acc=0.765),

        # Config('resnet18_v2', nbit_input=8, dtype_input='int8', nbit_output=16, dtype_output='int16', global_scale=4.0, expected_acc=0.611),
        # Config('resnet18_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=4.0, expected_acc=0.612),
        # Config('resnet34_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=4.0, expected_acc=0.726),
        # Config('resnet50_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=4.0, expected_acc=0.750),
        # Config('resnet101_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=4.0, expected_acc=0.752),

        # Config('resnet18_v2', nbit_input=8, dtype_input='int8', nbit_output=16, dtype_output='int16', global_scale=8.0, expected_acc=0.500),
        # Config('resnet18_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=8.0, expected_acc=0.500),
        # Config('resnet34_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=8.0, expected_acc=0.705),
        # Config('resnet50_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=8.0, expected_acc=0.661),
        # Config('resnet101_v2', nbit_input=8, dtype_input='int8', nbit_output=32, dtype_output='int32', global_scale=8.0, expected_acc=0.526),
    ]

    # global scales
    for config in configs:
        acc = test_quantize_acc(config, rec_val)
        results.append((config, acc))
    for res in results:
        print(res)

