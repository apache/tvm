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
import os
import sys
import logging

import pytest

pytest.importorskip("onnx")

import onnx

import tvm
from tvm import relay
from tvm.relay import quantize as qtz
import tvm.testing
from test_quantization_accuracy import Config, get_val_data, eval_acc

logging.basicConfig(level=logging.INFO)


def calibrate_dataset(model_name, rec_val, batch_size, calibration_samples):
    val_data, _ = get_val_data(model_name, rec_val=rec_val, batch_size=batch_size)
    val_data.reset()
    for i, batch in enumerate(val_data):
        if i * batch_size >= calibration_samples:
            break
        data = batch.data[0].asnumpy()
        yield {"data": data}


def download_file(url_base, file_name):
    if not os.path.exists(file_name) or not os.path.isfile(file_name):
        import urllib.request as urllib2

        url = "{}/{}".format(url_base, file_name)
        try:
            print("download from {}".format(url))
            if sys.version_info >= (3,):
                urllib2.urlretrieve(url, file_name)
            else:
                f = urllib2.urlopen(url)
                data = f.read()
                with open(file_name, "wb") as code:
                    code.write(data)
        except Exception as err:
            if os.path.exists(file_name):
                os.remove(file_name)
            raise Exception("download {} failed due to {}!".format(file_name, repr(err)))


def get_onnx_model(model_name, batch_size, qconfig, original=False, dataset=None):
    assert model_name == "vit32", "Only support vit32 model!"
    base = "https://github.com/TheGreatCold/tvm-vit/raw/d2aa1e60eef42e2fdedbd1e13aa85ac5faf0a7fc"
    logfile = "gtx1660_vit_B32_224.log"
    onnx_path = "vit_B32_224.onnx"

    download_file(base, logfile)
    download_file(base, onnx_path)

    onnx_graph = onnx.load(open(onnx_path, "rb"))
    data_shape = (batch_size, 3, 224, 224)
    mod, params = relay.frontend.from_onnx(onnx_graph, {"data": data_shape})

    with tvm.transform.PassContext(opt_level=3):
        qfunc = relay.quantize.prerequisite_optimize(mod, params=params)
    logging.debug("original")
    logging.debug(qfunc.astext(show_meta_data=False))
    if original:
        return qfunc, params, logfile

    with qconfig:
        logging.debug("current quantize config")
        logging.debug(qtz.current_qconfig())

        if dataset is not None:
            with tvm.target.cuda():
                with tvm.autotvm.apply_history_best(logfile):
                    qfunc = qtz.quantize(qfunc, params, dataset=dataset)
        else:
            qfunc = qtz.quantize(qfunc, params)

        logging.debug("after quantize")
        logging.debug(qfunc.astext(show_meta_data=False))
    return qfunc, params, logfile


@tvm.testing.requires_gpu
def test_onnx_quantize_acc(cfg, rec_val, batch_size=1, original=False):
    qconfig = qtz.qconfig(
        skip_conv_layers=[0],
        skip_dense_layer=False,
        nbit_input=cfg.nbit_input,
        nbit_weight=cfg.nbit_input,
        dtype_input=cfg.dtype_input,
        dtype_weight=cfg.dtype_input,
        dtype_activation=cfg.dtype_output,
        debug_enabled_ops=None,
        calibrate_mode="percentile",
        calibrate_chunk_by=8,
    )

    dataset = list(calibrate_dataset(cfg.model, rec_val, batch_size, 64))
    model, params, logfile = get_onnx_model(
        cfg.model, batch_size, qconfig, original=original, dataset=dataset
    )
    val_data, batch_fn = get_val_data(cfg.model, rec_val=rec_val, batch_size=batch_size)

    with tvm.autotvm.apply_history_best(logfile):
        acc = eval_acc(model, params, val_data, batch_fn, log_interval=1000)
    assert acc > cfg.expected_acc
    return acc


if __name__ == "__main__":
    # TODO(for user): replace the line with the path to imagenet validation dataset
    rec_val = "/scratch/tqchen/imagenet/val.rec"

    configs = [
        Config(
            "vit32",
            nbit_input=8,
            dtype_input="int8",
            nbit_output=32,
            dtype_output="int32",
            global_scale=8.0,
            expected_acc=0.727,
        ),
    ]

    for config in configs:

        # float32 model
        acc = test_onnx_quantize_acc(config, rec_val, batch_size=1, original=True)
        print("{}-float32: {}".format(config.model, acc))

        # int8 model
        acc = test_onnx_quantize_acc(config, rec_val, batch_size=1, original=False)
        print("{}-int8: {}".format(config.model, acc))
