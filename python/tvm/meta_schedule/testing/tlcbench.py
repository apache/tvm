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
# pylint: disable=invalid-name,import-outside-toplevel
# type: ignore
"""Model loader for TLCBench."""
import multiprocessing
import os
import logging
import tvm
from tvm import relay
from tvm.error import TVMError
from tvm.contrib.download import download_testdata


log = logging.getLogger(__name__)


def _convert(args):
    onnx_model, shape_dict, json_path, params_path = args
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    seq = tvm.transform.Sequential(
        [relay.transform.InferType(), relay.transform.FakeQuantizationToInteger(use_qat=True)]
    )
    mod = seq(mod)

    with open(json_path, "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    with open(params_path, "wb") as fo:
        fo.write(relay.save_param_dict(params))


def convert_to_qnn(onnx_path, json_path, params_path, input_info):
    """Run the ONNX frontend and the FQ2I pass. The output is serialized to disk."""
    import onnx

    onnx_model = onnx.load(onnx_path)

    shape_dict = dict(input_info)

    log.info("Converting te ONNX model to Relay and running the FQ2I pass, it may take a while...")

    with multiprocessing.Pool(processes=1) as pool:
        pool.map(_convert, [(onnx_model, shape_dict, json_path, params_path)])


def deserialize_relay(json_path, params_path):
    with open(json_path, "r") as fi:
        mod = tvm.ir.load_json(fi.read())

    with open(params_path, "rb") as fi:
        params = relay.load_param_dict(fi.read())

    return mod, params


def load_quantized_bert_base(batch_size=1, seq_len=384):
    """
    Load the quantized bert-base model from TLCBench, possibly downloading it from github
    and caching the converted int8 QNN module to disk.

    In addition to returing the relay module and its parameters, it also returns input name
    and shape information, which can be used at the deployment time as follows:

    ```
    mod, params, input_info = load_quantized_bert_base()

    ...

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    for name, shape in input_info:
        arr = np.random.uniform(1, 10, size=shape).astype("int64")
        runtime.set_input(name, arr)

    runtime.run()
    ```

    """
    url = "https://github.com/tlc-pack/TLCBench/raw/main/models/bert-base-qat.onnx"
    log.info("Downloading quantized bert-base model.")
    onnx_path = download_testdata(url, "bert-base-qat.onnx", module="tlcbench")
    data_dir = os.path.dirname(onnx_path)

    json_path = os.path.join(data_dir, "bert_base_int8_b%d_s%d.json" % (batch_size, seq_len))
    params_path = os.path.join(data_dir, "bert_base_int8_b%d_s%d.params" % (batch_size, seq_len))

    # Input names and order encoded in the ONNX model
    input_info = [
        ("input_ids", (batch_size, seq_len)),
        ("segment_ids", (batch_size, seq_len)),
        ("input_mask", (batch_size, seq_len)),
    ]

    if not os.path.exists(json_path) or not os.path.exists(params_path):
        convert_to_qnn(onnx_path, json_path, params_path, input_info)

    def deserialize():
        try:
            return deserialize_relay(json_path, params_path)
        except TVMError:
            # A serialized Relay json file may become invalid after TVM bump
            # Update the serialized model and try loading again
            convert_to_qnn(onnx_path, json_path, params_path, input_info)
            return deserialize_relay(json_path, params_path)

    mod, params = deserialize()

    return mod, params, input_info
