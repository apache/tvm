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
from urllib.parse import urlparse

import pytest
import tensorflow as tf
import numpy as np
import tarfile
import tempfile
import pathlib

import tvm
import tvm.testing
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from tvm.micro.project_api import server
from tvm.contrib.download import download_testdata
from tvm.micro import export_model_library_format
from tvm.micro.model_library_format import generate_c_interface_header
from tvm.micro.testing import create_aot_session, predict_labels_aot
from tvm.micro.testing.utils import (
    create_header_file,
    mlf_extract_workspace_size_bytes,
)

MLPERF_TINY_MODELS = {
    "kws": {
        "name": "Keyword Spotting",
        "index": 1,
        "url": "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite",
        "sample": "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/keyword_spotting_int8_6.pyc.npy",
    },
    "vww": {
        "name": "Visual Wake Words",
        "index": 2,
        "url": "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite",
        "sample": "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/visual_wake_word_int8_1.npy",
    },
    # Note: The reason we use quantized model with float32 I/O is
    # that TVM does not handle the int8 I/O correctly and accuracy
    # would drop significantly.
    "ad": {
        "name": "Anomaly Detection",
        "index": 3,
        "url": "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant_fullint_micro.tflite",
        "sample": "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/anomaly_detection_normal_id_01.npy",
        # This model takes in a (1, 640) vector, so it must be called 40 times -
        # once for each time slice.
    },
    "ic": {
        "name": "Image Classification",
        "index": 4,
        "url": "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite",
        "sample": "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/image_classification_int8_0.npy",
    },
}


def mlperftiny_get_module(model_name: str):
    model_url = MLPERF_TINY_MODELS[model_name]["url"]
    url = urlparse(model_url)
    model_path = download_testdata(model_url, os.path.basename(url.path), module="model")

    tflite_model_buf = open(model_path, "rb").read()
    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    model_info = {
        "input_name": input_details[0]["name"],
        "input_shape": tuple(input_details[0]["shape"]),
        "input_dtype": np.dtype(input_details[0]["dtype"]).name,
        "output_name": output_details[0]["name"],
        "output_shape": tuple(output_details[0]["shape"]),
        "output_dtype": np.dtype(output_details[0]["dtype"]).name,
    }

    if model_name != "ad":
        model_info["quant_output_scale"] = output_details[0]["quantization_parameters"]["scales"][0]
        model_info["quant_output_zero_point"] = output_details[0]["quantization_parameters"][
            "zero_points"
        ][0]

    relay_mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={model_info["input_name"]: model_info["input_shape"]},
        dtype_dict={model_info["input_name"]: model_info["input_dtype"]},
    )
    return relay_mod, params, model_info


def get_test_data(model_name: str) -> list:
    sample_url = MLPERF_TINY_MODELS[model_name]["sample"]
    url = urlparse(sample_url)
    sample_path = download_testdata(sample_url, os.path.basename(url.path), module="data")
    return [np.load(sample_path)]


def predict_ad_labels_aot(session, aot_executor, input_data, runs_per_sample=1):
    """A special version of tvm/micro/testing/evaluation.py's predict_labels_aot.
    The runtime returned for each sample is the median of the runtimes for all slices
    in that sample."""

    assert runs_per_sample > 0
    assert aot_executor.get_num_inputs() == 1
    assert aot_executor.get_num_outputs() == 1

    sample_counter = 0
    for sample in input_data:
        output_fp32 = np.empty_like(sample)
        slice_runtimes = []

        for i, time_slice in enumerate(sample):
            aot_executor.get_input(0).copyfrom(time_slice.reshape((1, 640)))
            result = aot_executor.module.time_evaluator(
                "run", session.device, number=runs_per_sample
            )()
            slice_runtimes.append(result.mean)
            output_fp32[i, :] = aot_executor.get_output(0).numpy()

        sample_counter += 1
        errors = np.mean(np.square(sample - output_fp32), axis=1)
        yield np.mean(errors), np.median(slice_runtimes)


@pytest.mark.parametrize("model_name", ["kws", "vww", "ad", "ic"])
@pytest.mark.parametrize("project_type", ["host_driven", "mlperftiny"])
@tvm.testing.requires_micro
@pytest.mark.skip_boards(
    ["mps2_an521", "mps3_an547", "stm32f746g_disco", "nucleo_f746zg", "nrf5340dk_nrf5340_cpuapp"]
)
def test_mlperftiny_models(platform, board, workspace_dir, serial_number, model_name, project_type):
    """MLPerfTiny models test.
    Testing MLPerfTiny models using host_driven project. In this case one input sample is used
    to verify the end to end execution. Accuracy is not checked in this test.

    Also, this test builds each model in standalone mode that can be used with EEMBC runner.
    """
    if platform != "zephyr":
        pytest.skip(reason="Other platforms are not supported yet.")

    use_cmsis_nn = False
    relay_mod, params, model_info = mlperftiny_get_module(model_name)
    target = tvm.micro.testing.get_target(platform, board)
    project_options = {"config_main_stack_size": 4000, "serial_number": serial_number}

    if use_cmsis_nn:
        project_options["cmsis_path"] = os.getenv("CMSIS_PATH")

    if model_name == "ad":
        predictor = predict_ad_labels_aot
    else:
        predictor = predict_labels_aot

    if project_type == "host_driven":
        with create_aot_session(
            platform,
            board,
            target,
            relay_mod,
            params,
            build_dir=workspace_dir,
            # The longest models take ~5 seconds to infer, but running them
            # ten times (with NUM_TESTING_RUNS_PER_SAMPLE) makes that 50
            timeout_override=server.TransportTimeouts(
                session_start_retry_timeout_sec=300,
                session_start_timeout_sec=150,
                session_established_timeout_sec=150,
            ),
            project_options=project_options,
            use_cmsis_nn=use_cmsis_nn,
        ) as session:
            aot_executor = tvm.runtime.executor.aot_executor.AotModule(
                session.create_aot_executor()
            )
            args = {
                "session": session,
                "aot_executor": aot_executor,
                "input_data": get_test_data(model_name),
                "runs_per_sample": 10,
            }
            predicted_labels, runtimes = zip(*predictor(**args))

        avg_runtime = float(np.mean(runtimes) * 1000)
        print(f"Model {model_name} average runtime: {avg_runtime}")

    elif project_type == "mlperftiny":
        runtime = Runtime("crt")
        executor = Executor(
            "aot", {"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 8}
        )

        config = {"tir.disable_vectorize": True}
        if use_cmsis_nn:
            from tvm.relay.op.contrib import cmsisnn

            config["relay.ext.cmsisnn.options"] = {"mcpu": target.mcpu}
            relay_mod = cmsisnn.partition_for_cmsisnn(relay_mod, params, mcpu=target.mcpu)

        with tvm.transform.PassContext(opt_level=3, config=config):
            module = tvm.relay.build(
                relay_mod, target=target, params=params, runtime=runtime, executor=executor
            )

        temp_dir = tvm.contrib.utils.tempdir()
        model_tar_path = temp_dir / "model.tar"
        export_model_library_format(module, model_tar_path)
        workspace_size = mlf_extract_workspace_size_bytes(model_tar_path)

        extra_tar_dir = tvm.contrib.utils.tempdir()
        extra_tar_file = extra_tar_dir / "extra.tar"
        with tarfile.open(extra_tar_file, "w:gz") as tf:
            with tempfile.TemporaryDirectory() as tar_temp_dir:
                model_files_path = os.path.join(tar_temp_dir, "include")
                os.mkdir(model_files_path)
                header_path = generate_c_interface_header(
                    module.libmod_name,
                    [model_info["input_name"]],
                    [model_info["output_name"]],
                    [],
                    {},
                    [],
                    0,
                    model_files_path,
                    {},
                    {},
                )
                tf.add(header_path, arcname=os.path.relpath(header_path, tar_temp_dir))

            create_header_file(
                "output_data",
                np.zeros(
                    shape=model_info["output_shape"],
                    dtype=model_info["output_dtype"],
                ),
                "include/tvm",
                tf,
            )

        input_total_size = 1
        input_shape = model_info["input_shape"]
        for i in range(len(input_shape)):
            input_total_size *= input_shape[i]

        template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects(platform))
        project_options.update(
            {
                "extra_files_tar": str(extra_tar_file),
                "project_type": project_type,
                "board": board,
                "compile_definitions": [
                    f"-DWORKSPACE_SIZE={workspace_size + 512}",  # Memory workspace size, 512 is a temporary offset
                    # since the memory calculation is not accurate.
                    f'-DTARGET_MODEL={MLPERF_TINY_MODELS[model_name]["index"]}',  # Sets the model index for project compilation.
                    f"-DTH_MODEL_VERSION=EE_MODEL_VERSION_{model_name.upper()}01",  # Sets model version. This is required by MLPerfTiny API.
                    f"-DMAX_DB_INPUT_SIZE={input_total_size}",  # Max size of the input data array.
                ],
            }
        )

        if model_name != "ad":
            project_options["compile_definitions"].append(
                f'-DOUT_QUANT_SCALE={model_info["quant_output_scale"]}'
            )
            project_options["compile_definitions"].append(
                f'-DOUT_QUANT_ZERO={model_info["quant_output_zero_point"]}'
            )

        project = tvm.micro.project.generate_project_from_mlf(
            template_project_path, workspace_dir / "project", model_tar_path, project_options
        )
        project.build()


if __name__ == "__main__":
    tvm.testing.main()
