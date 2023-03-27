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
import re
import logging
from urllib.parse import urlparse
import struct

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
from tvm.micro.testing.utils import aot_transport_find_message

MLPERF_TINY_MODELS = {
    "kws": {
        "name": "Keyword Spotting",
        "index": 1,
        "url": "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite",
        "sample": "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/keyword_spotting_int8_6.pyc.npy",
        "sample_label": 6,
    },
    "vww": {
        "name": "Visual Wake Words",
        "index": 2,
        "url": "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite",
        "sample": "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/visual_wake_word_int8_1.npy",
        "sample_label": 1,
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
        "sample_label": 0,
    },
}

MLPERFTINY_READY_MSG = "m-ready"
MLPERFTINY_RESULT_MSG = "m-results"
MLPERFTINY_NAME_MSG = "m-name"


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


def get_test_data(model_name: str, project_type: str) -> list:
    sample_url = MLPERF_TINY_MODELS[model_name]["sample"]
    url = urlparse(sample_url)
    sample_path = download_testdata(sample_url, os.path.basename(url.path), module="data")
    sample = np.load(sample_path)
    if project_type == "mlperftiny" and model_name != "ad":
        sample = sample.astype(np.uint8)
    sample_label = None
    if "sample_label" in MLPERF_TINY_MODELS[model_name].keys():
        sample_label = MLPERF_TINY_MODELS[model_name]["sample_label"]
    return [sample], [sample_label]


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


def _mlperftiny_get_name(device_transport) -> str:
    """Get device name."""
    device_transport.write(b"name%", timeout_sec=5)
    name_message = aot_transport_find_message(device_transport, MLPERFTINY_NAME_MSG, timeout_sec=5)
    m = re.search(r"\[([A-Za-z0-9_]+)\]", name_message)
    return m.group(1)


def _mlperftiny_infer(transport, warmup: int, infer: int, timeout: int):
    """Send MLPerfTiny infer command."""
    cmd = f"infer {warmup} {infer}%".encode("UTF-8")
    transport.write(cmd, timeout_sec=timeout)


def _mlperftiny_write_sample(device_transport, data: list, timeout: int):
    """Write a sample with MLPerfTiny compatible format."""
    cmd = f"db load {len(data)}%".encode("UTF-8")
    logging.debug(f"transport write: {cmd}")
    device_transport.write(cmd, timeout)
    aot_transport_find_message(device_transport, MLPERFTINY_READY_MSG, timeout_sec=timeout)
    for item in data:
        if isinstance(item, float):
            ba = bytearray(struct.pack("<f", item))
            hex_array = ["%02x" % b for b in ba]
        else:
            hex_val = format(item, "x")
            # make sure hex value is in HH format
            if len(hex_val) < 2:
                hex_val = "0" + hex_val
            elif len(hex_val) > 2:
                raise ValueError(f"Hex value not in HH format: {hex_val}")
            hex_array = [hex_val]

        for hex_val in hex_array:
            cmd = f"db {hex_val}%".encode("UTF-8")
            logging.debug(f"transport write: {cmd}")
            device_transport.write(cmd, timeout)
            aot_transport_find_message(device_transport, MLPERFTINY_READY_MSG, timeout_sec=timeout)


def _mlperftiny_test_dataset(device_transport, dataset, timeout):
    """Run test dataset compatible with MLPerfTiny format."""
    num_correct = 0
    total = 0
    samples, labels = dataset
    i_counter = 0
    for sample in samples:
        label = labels[i_counter]
        logging.info(f"Writing Sample {i_counter}")
        _mlperftiny_write_sample(device_transport, sample.flatten().tolist(), timeout)
        _mlperftiny_infer(device_transport, 1, 0, timeout)
        results = aot_transport_find_message(
            device_transport, MLPERFTINY_RESULT_MSG, timeout_sec=timeout
        )

        m = re.search(r"m\-results\-\[([A-Za-z0-9_,.]+)\]", results)
        results = m.group(1).split(",")
        results_val = [float(x) for x in results]
        results_val = np.array(results_val)

        if np.argmax(results_val) == label:
            num_correct += 1
        total += 1
        i_counter += 1
    return float(num_correct / total)


def _mlperftiny_test_dataset_ad(device_transport, dataset, timeout):
    """Run test dataset compatible with MLPerfTiny format for AD model."""
    samples, _ = dataset
    result_output = np.zeros(samples[0].shape[0])

    for slice in range(0, 40):
        _mlperftiny_write_sample(device_transport, samples[0][slice, :].flatten().tolist(), timeout)
        _mlperftiny_infer(device_transport, 1, 0, timeout)
        results = aot_transport_find_message(
            device_transport, MLPERFTINY_RESULT_MSG, timeout_sec=timeout
        )
        m = re.search(r"m\-results\-\[([A-Za-z0-9_,.]+)\]", results)
        results = m.group(1).split(",")
        results_val = [float(x) for x in results]
        result_output[slice] = np.array(results_val)
    return np.average(result_output)


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

    Also, this test builds each model in standalone mode that can be used for MLPerfTiny submissions.
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

    samples, labels = get_test_data(model_name, project_type)
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
                "input_data": samples,
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

        # float input
        if model_name == "ad":
            input_total_size *= 4

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
        project.flash()
        with project.transport() as transport:
            aot_transport_find_message(transport, MLPERFTINY_READY_MSG, timeout_sec=200)
            print(f"Testing {model_name} on {_mlperftiny_get_name(transport)}.")
            assert _mlperftiny_get_name(transport) == "microTVM"
            if model_name != "ad":
                accuracy = _mlperftiny_test_dataset(transport, [samples, labels], 100)
                print(f"Model {model_name} accuracy: {accuracy}")
            else:
                mean_error = _mlperftiny_test_dataset_ad(transport, [samples, None], 100)
                print(
                    f"""Model {model_name} mean error: {mean_error}.
                      Note that this is not the final accuracy number.
                      To calculate that, you need to use sklearn.metrics.roc_auc_score function."""
                )


if __name__ == "__main__":
    tvm.testing.main()
