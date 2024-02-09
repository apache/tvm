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
import logging
import os

import mxnet as mx
from mxnet import gluon
import pytest

import tvm
import tvm.testing
from tvm import relay
from tvm.relay import quantize as qtz

logging.basicConfig(level=logging.INFO)

Config = namedtuple(
    "Config",
    [
        "model",
        "nbit_input",
        "dtype_input",
        "nbit_output",
        "dtype_output",
        "global_scale",
        "expected_acc",
    ],
)


def get_val_data(model_name, rec_val, batch_size, num_workers=4):
    rec_val = os.path.expanduser(rec_val)
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    img_size = 299 if model_name == "inceptionv3" else 224
    val_data = mx.io.ImageRecordIter(
        path_imgrec=rec_val,
        preprocess_threads=num_workers,
        shuffle=False,
        batch_size=batch_size,
        resize=256,
        data_shape=(3, img_size, img_size),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
    )
    return val_data, batch_fn


def get_model(model_name, batch_size, qconfig, original=False):
    try:
        gluon_model = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
    except RuntimeError:
        pytest.skip(reason="mxnet downloads no longer supported")
    img_size = 299 if model_name == "inceptionv3" else 224
    data_shape = (batch_size, 3, img_size, img_size)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})

    logging.debug("original")
    logging.debug(mod.astext(show_meta_data=False))
    if original:
        return mod, params

    with qconfig:
        logging.debug("current quantize config")
        logging.debug(qtz.current_qconfig())
        qfunc = qtz.quantize(mod, params)
        logging.debug("after quantize")
        logging.debug(qfunc.astext(show_meta_data=False))
    return qfunc, params


def eval_acc(
    model, params, dataset, batch_fn, target=tvm.target.cuda(), device=tvm.cuda(), log_interval=500
):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(model, target, params=params)
    # create runtime module
    m = tvm.contrib.graph_executor.GraphModule(lib["default"](device))

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
        m.set_input("data", tvm.nd.array(data[0].asnumpy()))
        m.run()
        out_arr = m.get_output(0)
        acc_top1.update(label, [mx.nd.array(out_arr.numpy())])
        acc_top5.update(label, [mx.nd.array(out_arr.numpy())])

        if not (i + 1) % log_interval:
            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            nsamples = (i + 1) * batch_size
            logging.info("[%d samples] validation: acc-top1=%f acc-top5=%f", nsamples, top1, top5)
    logging.info("[final] validation: acc-top1=%f acc-top5=%f", top1, top5)
    return top1


@tvm.testing.requires_gpu
def test_quantize_acc(cfg, rec_val):
    qconfig = qtz.qconfig(
        skip_conv_layers=[0],
        nbit_input=cfg.nbit_input,
        nbit_weight=cfg.nbit_input,
        global_scale=cfg.global_scale,
        dtype_input=cfg.dtype_input,
        dtype_weight=cfg.dtype_input,
        dtype_activation=cfg.dtype_output,
        debug_enabled_ops=None,
    )

    batch_size = 1
    model, params = get_model(cfg.model, batch_size, qconfig)
    val_data, batch_fn = get_val_data(cfg.model, rec_val=rec_val, batch_size=batch_size)

    acc = eval_acc(model, params, val_data, batch_fn)
    assert acc > cfg.expected_acc
    return acc


if __name__ == "__main__":
    # TODO(for user): replace the line with the path to imagenet validation dataset
    rec_val = "/scratch/tqchen/imagenet/val.rec"

    results = []
    configs = [
        # TODO: need to fix accuracy and add AutoTVM log
        Config(
            "mobilenetv2_1.0",
            nbit_input=8,
            dtype_input="int8",
            nbit_output=32,
            dtype_output="int32",
            global_scale=4.0,
            expected_acc=0.666,
        ),
        Config(
            "mobilenetv2_1.0",
            nbit_input=8,
            dtype_input="int8",
            nbit_output=16,
            dtype_output="int16",
            global_scale=4.0,
            expected_acc=0.666,
        ),
        Config(
            "resnet18_v1",
            nbit_input=8,
            dtype_input="int8",
            nbit_output=16,
            dtype_output="int16",
            global_scale=8.0,
            expected_acc=0.692,
        ),
        Config(
            "resnet18_v1",
            nbit_input=8,
            dtype_input="int8",
            nbit_output=32,
            dtype_output="int32",
            global_scale=8.0,
            expected_acc=0.692,
        ),
        Config(
            "resnet34_v1",
            nbit_input=8,
            dtype_input="int8",
            nbit_output=32,
            dtype_output="int32",
            global_scale=8.0,
            expected_acc=0.733,
        ),
        Config(
            "resnet50_v1",
            nbit_input=8,
            dtype_input="int8",
            nbit_output=32,
            dtype_output="int32",
            global_scale=8.0,
            expected_acc=0.747,
        ),
        Config(
            "resnet101_v1",
            nbit_input=8,
            dtype_input="int8",
            nbit_output=32,
            dtype_output="int32",
            global_scale=8.0,
            expected_acc=0.756,
        ),
    ]

    for config in configs:
        acc = test_quantize_acc(config, rec_val)
        results.append((config, acc))
    for res in results:
        print(res)
