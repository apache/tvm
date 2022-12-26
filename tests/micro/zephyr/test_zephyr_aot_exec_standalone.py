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
import pathlib

import pytest
import numpy as np

import tvm
import tvm.testing
from tvm.micro.project_api import server
import tvm.relay as relay
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.download import download_testdata

from . import utils


@tvm.testing.requires_micro
@pytest.mark.skip_boards(["mps2_an521", "mps3_an547"])
def test_tflite(workspace_dir, board, microtvm_debug, serial_number):
    """Testing a TFLite model."""
    model = utils.ZEPHYR_BOARDS[board]
    input_shape = (1, 49, 10, 1)
    output_shape = (1, 12)
    build_config = {"debug": microtvm_debug}

    model_url = "https://github.com/tlc-pack/web-data/raw/25fe99fb00329a26bd37d3dca723da94316fd34c/testdata/microTVM/model/keyword_spotting_quant.tflite"
    model_path = download_testdata(model_url, "keyword_spotting_quant.tflite", module="model")

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

    target = tvm.target.target.micro(model)
    executor = Executor(
        "aot", {"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 4}
    )
    runtime = Runtime("crt")
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(relay_mod, target, params=params, runtime=runtime, executor=executor)

    sample_url = "https://github.com/tlc-pack/web-data/raw/967fc387dadb272c5a7f8c3461d34c060100dbf1/testdata/microTVM/data/keyword_spotting_int8_6.pyc.npy"
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


@tvm.testing.requires_micro
@pytest.mark.skip_boards(["mps2_an521", "mps3_an547"])
def test_qemu_make_fail(workspace_dir, board, microtvm_debug, serial_number):
    """Testing QEMU make fail."""
    if board not in ["qemu_x86", "mps2_an521", "mps3_an547"]:
        pytest.skip(msg="Only for QEMU targets.")

    model = utils.ZEPHYR_BOARDS[board]
    build_config = {"debug": microtvm_debug}
    shape = (10,)
    dtype = "float32"

    # Construct Relay program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(np.ones(shape=shape, dtype=dtype)))
    func = relay.Function([x], z)
    ir_mod = tvm.IRModule.from_expr(func)

    target = tvm.target.target.micro(model)
    executor = Executor("aot")
    runtime = Runtime("crt")
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(ir_mod, target, executor=executor, runtime=runtime)

    sample = np.zeros(shape=shape, dtype=dtype)
    project, project_dir = utils.generate_project(
        workspace_dir,
        board,
        lowered,
        build_config,
        sample,
        shape,
        dtype,
        False,
        serial_number,
    )

    file_path = pathlib.Path(project_dir) / "build" / "build.ninja"
    assert file_path.is_file(), f"[{file_path}] does not exist."

    # Remove a file to create make failure.
    os.remove(file_path)
    project.flash()
    with pytest.raises(server.JSONRPCError) as excinfo:
        project.transport().open()
    assert "QEMU setup failed" in str(excinfo.value)


if __name__ == "__main__":
    tvm.testing.main()
