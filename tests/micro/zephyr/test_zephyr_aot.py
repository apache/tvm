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
import io
import logging
import os
import sys
import logging
import pathlib
import tarfile
import tempfile

import pytest
import numpy as np

import tvm
import tvm.testing
from tvm.micro.project_api import server
import tvm.relay as relay

from tvm.contrib.download import download_testdata
from tvm.micro.model_library_format import generate_c_interface_header
from tvm.micro.testing import aot_transport_init_wait, aot_transport_find_message

import test_utils


@tvm.testing.requires_micro
def test_tflite(temp_dir, board, west_cmd, tvm_debug):
    """Testing a TFLite model."""
    model = test_utils.ZEPHYR_BOARDS[board]
    input_shape = (1, 49, 10, 1)
    output_shape = (1, 12)
    build_config = {"debug": tvm_debug}

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

    target = tvm.target.target.micro(
        model, options=["-link-params=1", "--executor=aot", "--unpacked-api=1", "--interface-api=c"]
    )
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(relay_mod, target, params=params)

    # Load sample and generate input/output header files
    sample_url = "https://github.com/tlc-pack/web-data/raw/967fc387dadb272c5a7f8c3461d34c060100dbf1/testdata/microTVM/data/keyword_spotting_int8_6.pyc.npy"
    sample_path = download_testdata(
        sample_url, "keyword_spotting_int8_6.pyc.npy", module="data"
    )
    sample = np.load(sample_path)

    with tempfile.NamedTemporaryFile() as tar_temp_file:
        with tarfile.open(tar_temp_file.name, "w:gz") as tf:
            with tempfile.TemporaryDirectory() as tar_temp_dir:
                model_files_path = os.path.join(tar_temp_dir, "include")
                os.mkdir(model_files_path)
                header_path = generate_c_interface_header(
                    lowered.libmod_name, ["input_1"], ["output"], model_files_path
                )
                tf.add(header_path, arcname=os.path.relpath(header_path, tar_temp_dir))

            test_utils.create_header_file("input_data", sample, "include", tf)
            test_utils.create_header_file(
                "output_data", np.zeros(shape=output_shape, dtype="int8"), "include", tf
            )

        project, _ = test_utils.build_project(
            temp_dir,
            board,
            west_cmd,
            lowered,
            build_config,
            extra_files_tar=tar_temp_file.name,
        )

    project.flash()
    with project.transport() as transport:
        aot_transport_init_wait(transport)
        transport.write(b"infer%", timeout_sec=5)
        result_line = aot_transport_find_message(transport, "result", timeout_sec=60)

    result_line = result_line.strip("\n")
    result_line = result_line.split(":")
    result = int(result_line[1])
    time = int(result_line[2])
    logging.info(f"Result: {result}\ttime: {time} ms")
    assert result == 6


@tvm.testing.requires_micro
def test_qemu_make_fail(temp_dir, board, west_cmd, tvm_debug):
    """Testing QEMU make fail."""
    if board not in ["qemu_x86", "mps2_an521"]:
        pytest.skip(msg="Only for QEMU targets.")

    model = test_utils.ZEPHYR_BOARDS[board]
    build_config = {"debug": tvm_debug}
    shape = (10,)
    dtype = "float32"

    # Construct Relay program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(np.ones(shape=shape, dtype=dtype)))
    func = relay.Function([x], z)
    ir_mod = tvm.IRModule.from_expr(func)

    target = tvm.target.target.micro(model, options=["-link-params=1", "--executor=aot"])
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(ir_mod, target)

    # Generate input/output header files
    with tempfile.NamedTemporaryFile() as tar_temp_file:
        with tarfile.open(tar_temp_file.name, "w:gz") as tf:
            with tempfile.TemporaryDirectory() as tar_temp_dir:
                model_files_path = os.path.join(tar_temp_dir, "include")
                os.mkdir(model_files_path)
                header_path = generate_c_interface_header(
                    lowered.libmod_name, ["input_1"], ["output"], model_files_path
                )
                tf.add(header_path, arcname=os.path.relpath(header_path, tar_temp_dir))
            test_utils.create_header_file(
                "input_data", np.zeros(shape=shape, dtype=dtype), "include", tf
            )
            test_utils.create_header_file(
                "output_data", np.zeros(shape=shape, dtype=dtype), "include", tf
            )

        project, project_dir = test_utils.build_project(
            temp_dir,
            board,
            west_cmd,
            lowered,
            build_config,
            extra_files_tar=tar_temp_file.name,
        )

    file_path = (
        pathlib.Path(project_dir) / "build" / "zephyr" / "CMakeFiles" / "run.dir" / "build.make"
    )
    assert file_path.is_file(), f"[{file_path}] does not exist."

    # Remove a file to create make failure.
    os.remove(file_path)
    project.flash()
    with pytest.raises(server.JSONRPCError) as excinfo:
        project.transport().open()
    assert "QEMU setup failed" in str(excinfo.value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
