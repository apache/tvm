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

PLATFORMS = conftest.PLATFORMS


def _generate_project(model, target, arduino_board, arduino_cmd, mod, build_config):
    parent_dir = os.path.dirname(__file__)
    filename = os.path.splitext(os.path.basename(__file__))[0]
    prev_build = (
        f"{os.path.join(parent_dir, 'archive')}_{filename}_{arduino_board}_last_build.micro"
    )
    workspace_root = os.path.join(
        f"{os.path.join(parent_dir, 'workspace')}_{filename}_{arduino_board}",
        datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
    )
    workspace_parent = os.path.dirname(workspace_root)
    if not os.path.exists(workspace_parent):
        os.makedirs(workspace_parent)
    workspace = tvm.micro.Workspace(debug=False, root=workspace_root)

    template_project_dir = (
        pathlib.Path(__file__).parent
        / ".."
        / ".."
        / ".."
        / "apps"
        / "microtvm"
        / "arduino"
        / "template_project"
    ).resolve()
    project = tvm.micro.generate_project(
        str(template_project_dir),
        mod,
        workspace.relpath("project"),
        {"arduino_board": arduino_board, "arduino_cmd": arduino_cmd, "verbose": 0},
    )
    return (workspace, project)


# This is bad, don't do this
TARGET = "c -keys=cpu -link-params=1 -mcpu=cortex-m33 -model=nrf5340dk -runtime=c -system-lib=1"


@pytest.fixture(scope="module")
def yes_no_project(platform, arduino_cmd):
    current_dir = os.path.dirname(__file__)
    model, arduino_board = PLATFORMS[platform]
    # target = tvm.target.target.micro(model, options=["-link-params=1"])
    build_config = {}

    with open(f"{current_dir}/testdata/yes_no.tflite", "rb") as f:
        tflite_model_buf = f.read()
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    mod, params = relay.frontend.from_tflite(tflite_model)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = relay.build(mod, TARGET, params=params)

    return _generate_project(model, TARGET, arduino_board, arduino_cmd, mod, build_config)
    # return tvm.micro.Session(project.transport())


@pytest.fixture(scope="module")
def project(yes_no_project):
    workspace, project = yes_no_project
    return project


@pytest.fixture(scope="module")
def project_dir(yes_no_project):
    workspace, project = yes_no_project
    return pathlib.Path(workspace.path) / "project"


def test_project_folder_structure(project_dir):
    assert set(["microtvm_api_server.py", "project.ino", "src"]).issubset(os.listdir(project_dir))

    source_dir = project_dir / "src"
    assert set(os.listdir(source_dir)) == set(
        ["model", "standalone_crt", "implementation.c", "model.cpp", "model.h", "parameters.h"]
    )


def test_project_model_integrity(project_dir):
    model_dir = project_dir / "src" / "model"
    assert set(os.listdir(model_dir)) == set(
        ["default_lib0.c", "default_lib1.c", "graph_json.c", "model.tar"]
    )
    with (model_dir / "graph_json.c").open() as f:
        graph_json_c = f.read()
        assert "static const char* graph_json" in graph_json_c


def test_parameter_header_templating(project_dir):
    # Ensure parameters.h was templated with correct information
    # for our yes/no model
    with (project_dir / "src" / "parameters.h").open() as f:
        parameters_h = f.read()
        assert "INPUT_DATA_SHAPE[] = {1, 1960};" in parameters_h


def test_import_rerouting(project_dir):
    # Check one file to ensure imports were rerouted
    runtime_c_path = project_dir / "src" / "standalone_crt" / "src" / "runtime"
    load_json_path = runtime_c_path / "crt" / "graph_executor" / "load_json.c"
    assert load_json_path.exists()

    with (load_json_path).open() as f:
        load_json_c = f.read()
        assert '#include "stdlib.h"' in load_json_c
        assert "include/tvm/runtime/crt/platform.h" in load_json_c


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
