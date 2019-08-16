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


def get_model(model_name, batch_size, qconfig, original=False, simulated=False, calib_set=None):
    gluon_model = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
    img_size = 299 if model_name == 'inceptionv3' else 224
    data_shape = (batch_size, 3, img_size, img_size)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})

    qmod = hago.prerequisite_optimize(mod, params=params)
    logging.debug('original')
    logging.debug(qmod['main'].astext(show_meta_data=False))

    if original:
        return qmod

    with qconfig:
        logging.debug('current quantize config')
        logging.debug(hago.current_qconfig())
        hardware = hago.create_accelerator_description()
        strategy, acc = hago.search_quantize_strategy(qmod, hardware, dataset=calib_set)
        print('simulated accuracy on calibration dataset: {}'.format(acc))
        quantizer = hago.create_quantizer(qmod['main'], hardware, strategy)
        simulated_graph = quantizer.simulate()
        quantized_graph = quantizer.quantize()
        logging.debug('after quantize')
        logging.debug(quantized_graph.astext(show_meta_data=False))
        out, acc = hago.eval_acc(quantized_graph, calib_set)
        print('quantized accuracy on calibration dataset: {}'.format(acc))
        # hago.inspect_graph_statistic(qmod['main'], hardware, strategy, dataset=calib_set)
        qmod = relay.Module.from_expr(quantized_graph)
        raise ValueError
    return qmod


def eval_acc(mod, dataset, batch_fn, target='llvm', ctx=tvm.cpu(), log_interval=100):
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target)
    # create runtime module
