import datetime
import os
import pathlib
import shutil
import sys
import time

import pytest
import tflite
import tvm
from tvm import micro, relay

import conftest

"""
This unit test simulates a simple user workflow, where we:
1. Generate a base sketch using a simple audio model
2. Modify the .ino file, much like a user would
3. Compile the sketch for the target platform
-- If physical hardware is present --
4. Upload the sketch to a connected board
5. Open a serial connection to the board
6. Use serial connection to ensure model behaves correctly
"""


# Since these tests are sequential, we'll use the same project for all tests
@pytest.fixture(scope="module")
def workspace_dir(request, platform):
    return conftest.make_workspace_dir("arduino_workflow", platform)


@pytest.fixture(scope="module")
def project_dir(workspace_dir):
    return workspace_dir / "project"


def _generate_project(arduino_board, arduino_cli_cmd, workspace_dir, mod, build_config):
    return tvm.micro.generate_project(
        str(conftest.TEMPLATE_PROJECT_DIR),
        mod,
        workspace_dir / "project",
        {
            "arduino_board": arduino_board,
            "arduino_cli_cmd": arduino_cli_cmd,
            "project_type": "example_project",
            "verbose": bool(build_config.get("debug")),
        },
    )


# We MUST pass workspace_dir, not project_dir, or the workspace will be dereferenced too soon
@pytest.fixture(scope="module")
def project(platform, arduino_cli_cmd, tvm_debug, workspace_dir):
    current_dir = os.path.dirname(__file__)
    model, arduino_board = conftest.PLATFORMS[platform]
    build_config = {"debug": tvm_debug}

    with open(f"{current_dir}/testdata/yes_no.tflite", "rb") as f:
        tflite_model_buf = f.read()
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    mod, params = relay.frontend.from_tflite(tflite_model)
    target = tvm.target.target.micro(
        model, options=["--link-params=1", "--unpacked-api=1", "--executor=aot"]
    )

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = relay.build(mod, target, params=params)

    return _generate_project(arduino_board, arduino_cli_cmd, workspace_dir, mod, build_config)


def test_project_folder_structure(project_dir, project):
    assert set(["microtvm_api_server.py", "project.ino", "src"]).issubset(os.listdir(project_dir))

    source_dir = project_dir / "src"
    assert set(os.listdir(source_dir)) == set(["model", "standalone_crt", "model.c", "model.h"])


def test_project_model_integrity(project_dir, project):
    model_dir = project_dir / "src" / "model"
    assert set(os.listdir(model_dir)) == set(["default_lib0.c", "default_lib1.c", "model.tar"])


def test_model_header_templating(project_dir, project):
    # Ensure model.h was templated with correct WORKSPACE_SIZE
    with (project_dir / "src" / "model.h").open() as f:
        model_h = f.read()
        assert "#define WORKSPACE_SIZE 21312" in model_h


def test_import_rerouting(project_dir, project):
    # Check one file to ensure imports were rerouted
    runtime_path = project_dir / "src" / "standalone_crt" / "src" / "runtime"
    c_backend_api_path = runtime_path / "crt" / "common" / "crt_backend_api.c"
    assert c_backend_api_path.exists()

    with c_backend_api_path.open() as f:
        c_backend_api_c = f.read()
        assert '#include "inttypes.h"' in c_backend_api_c
        assert "include/tvm/runtime/crt/platform.h" in c_backend_api_c


# Build on top of the generated project by replacing the
# top-level .ino fileand adding data input files, much
# like a user would
@pytest.fixture(scope="module")
def modified_project(project_dir, project):
    testdata_dir = pathlib.Path(os.path.dirname(__file__)) / "testdata"

    shutil.copy2(testdata_dir / "project.ino", project_dir / "project.ino")

    project_data_dir = project_dir / "src" / "data"
    project_data_dir.mkdir()
    for sample in ["yes.c", "no.c", "silence.c", "unknown.c"]:
        shutil.copy2(testdata_dir / sample, project_data_dir / sample)

    return project


@pytest.fixture(scope="module")
def compiled_project(modified_project):
    modified_project.build()
    return modified_project


def test_compile_yes_no_project(project_dir, project, compiled_project):
    build_dir = project_dir / "build"
    assert build_dir.exists()
    first_build_file = next(build_dir.iterdir(), None)
    assert first_build_file is not None


"""------------------------------------------------------------
If we're not running on real hardware, no further tests are run
------------------------------------------------------------"""


@pytest.fixture(scope="module")
def uploaded_project(compiled_project, run_hardware_tests):
    if not run_hardware_tests:
        pytest.skip()

    compiled_project.flash()
    return compiled_project


""" Sample serial output:

category,runtime,yes,no,silence,unknown
yes,56762,115,-123,-125,-123,
no,56762,-128,4,-123,-9,
silence,56792,-128,-118,107,-117,
unknown,56792,-128,-125,-128,125,
"""
SERIAL_OUTPUT_HEADERS = "category,runtime,yes,no,silence,unknown"


@pytest.fixture(scope="module")
def serial_output(uploaded_project):
    # Give time for the board to open a serial connection
    time.sleep(1)

    transport = uploaded_project.transport()
    transport.open()
    out = transport.read(2048, -1)
    out_str = out.decode("utf-8")
    out_lines = out_str.split("\r\n")

    assert SERIAL_OUTPUT_HEADERS in out_lines
    headers_index = out_lines.index(SERIAL_OUTPUT_HEADERS)
    data_lines = out_lines[headers_index + 1 : headers_index + 5]
    split_lines = [line.split(",") for line in data_lines]

    return [[line[0]] + list(map(int, line[1:6])) for line in split_lines]


TENSORFLOW_EVALUATIONS = {
    "yes": [115, -123, -125, -123],
    "no": [-128, 4, -123, -9],
    "silence": [-128, -118, 107, -117],
    "unknown": [-128, -125, -128, 125],
}
MAX_PREDICTION_DIFFERENCE = 2


def test_project_inference_correctness(serial_output):
    predictions = {line[0]: line[2:] for line in serial_output}

    for sample, prediction in predictions.items():
        # Due to rounding issues, we don't get the *exact* same
        # values as Tensorflow gives, but they're pretty close

        reference_prediction = TENSORFLOW_EVALUATIONS[sample]
        deltas = [prediction[i] - reference_prediction[i] for i in range(4)]
        assert max(deltas) < MAX_PREDICTION_DIFFERENCE


MAX_INFERENCE_TIME_US = 200 * 1000
MAX_INFERENCE_TIME_RANGE_US = 1000


def test_project_inference_runtime(serial_output):
    runtimes_us = [line[1] for line in serial_output]

    # Inference time will vary based on architecture
    # and clock speed. However, anything more than 200 ms
    # is way too long. Each inference takes ~60 ms on the
    # Sony spresense, running at 156 MHz
    assert max(runtimes_us) < MAX_INFERENCE_TIME_US

    # Clock speeds should be consistent for each input. On
    # the Sony spresense, they vary by <100 us. Note that
    # running with other attached hardware (like the
    # Spresense extension board) may cause this check to fail
    range_runtimes_us = max(runtimes_us) - min(runtimes_us)
    assert range_runtimes_us < MAX_INFERENCE_TIME_RANGE_US


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
