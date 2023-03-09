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
import pytest
import numpy as np

import tvm
import tvm.testing
import tvm.micro.testing
import tvm.relay as relay
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.download import download_testdata

from . import utils


@tvm.testing.requires_micro
@pytest.mark.skip_boards(["mps2_an521", "mps3_an547"])
def test_tflite(workspace_dir, board, microtvm_debug, serial_number):
    """Testing a TFLite model."""
    input_shape = (1, 49, 10, 1)
    output_shape = (1, 12)
    build_config = {"debug": microtvm_debug}

    model_url = "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite"
    model_path = download_testdata(model_url, "kws_ref_model.tflite", module="model")

    # Import TFLite model
    tflite_model_buf = open(model_path, "rb").read()
    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    # Load TFLite model and convert to Relay
    relay_mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict={"input_1": input_shape}, dtype_dict={"input_1 ": "int8"}
    )

    target = tvm.micro.testing.get_target("zephyr", board)
    executor = Executor(
        "aot", {"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 4}
    )
    runtime = Runtime("crt")
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(relay_mod, target, params=params, runtime=runtime, executor=executor)

    sample_url = "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/keyword_spotting_int8_6.pyc.npy"
    sample_path = download_testdata(sample_url, "keyword_spotting_int8_6.pyc.npy", module="data")
    sample = np.load(sample_path)

    project, _ = utils.generate_project(
        workspace_dir,
        board,
        lowered,
        build_config,
        sample,
        output_shape,
        "int8",
        False,
        serial_number,
    )

    result, _ = utils.run_model(project)
    assert result == 6


if __name__ == "__main__":
    tvm.testing.main()
