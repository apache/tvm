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

import pathlib
import re
import shutil
import pytest

import tvm.testing

import test_utils

"""
This unit test simulates a simple user workflow, where we:
1. Generate a base sketch using a simple audio model
2. Modify the .ino file, much like a user would
3. Compile the sketch for the target board
-- If physical hardware is present --
4. Upload the sketch to a connected board
5. Open a serial connection to the board
6. Use serial connection to ensure model behaves correctly
"""

# Since these tests are sequential, we'll use the same project/workspace
# directory for all tests in this file. Note that --board can't be loaded
# from the fixture, since the fixture is function scoped (it has to be
# for the tests to be named correctly via parameterization).
@pytest.fixture(scope="module")
def workflow_workspace_dir(request):
    board = request.config.getoption("--board")
    return test_utils.make_workspace_dir("arduino_workflow", board)


@pytest.fixture(scope="module")
def project_dir(workflow_workspace_dir):
    return workflow_workspace_dir / "project"


# We MUST pass workspace_dir, not project_dir, or the workspace will be dereferenced
# too soon. We can't use the board fixture either for the reason mentioned above.
@pytest.fixture(scope="module")
def project(request, microtvm_debug, workflow_workspace_dir):
    board = request.config.getoption("--board")
    serial_number = request.config.getoption("--serial-number")
    return test_utils.make_kws_project(board, microtvm_debug, workflow_workspace_dir, serial_number)


def _get_directory_elements(directory):
    return set(f.name for f in directory.iterdir())


def test_project_folder_structure(project_dir, project):
    assert set(["microtvm_api_server.py", "project.ino", "src"]).issubset(
        _get_directory_elements(project_dir)
    )

    source_dir = project_dir / "src"
    assert _get_directory_elements(source_dir) == set(
        ["model", "standalone_crt", "platform.c", "platform.h"]
    )


def test_project_model_integrity(project_dir, project):
    model_dir = project_dir / "src" / "model"
    assert _get_directory_elements(model_dir) == set(
        ["default_lib0.c", "default_lib1.c", "default_lib2.c", "model.tar"]
    )


def test_model_platform_templating(project_dir, project):
    # Ensure platform.c was templated with correct TVM_WORKSPACE_SIZE_BYTES
    with (project_dir / "src" / "platform.c").open() as f:
        platform_c = f.read()
        workspace_size_defs = re.findall(r"\#define TVM_WORKSPACE_SIZE_BYTES ([0-9]*)", platform_c)
        assert workspace_size_defs
        assert len(workspace_size_defs) == 1

        # Make sure the TVM_WORKSPACE_SIZE_BYTES we define is a reasonable size. We don't want
        # to set an exact value, as this test shouldn't break if an improvement to
        # TVM causes the amount of memory needed to decrease.
        workspace_size = int(workspace_size_defs[0])
        assert workspace_size < 30000
        assert workspace_size > 9000


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
    this_dir = pathlib.Path(__file__).parent
    kws_testdata_dir = this_dir.parent / "testdata" / "kws"
    arduino_testdata_dir = this_dir / "testdata"

    shutil.copy2(arduino_testdata_dir / "project.ino", project_dir / "project.ino")

    project_data_dir = project_dir / "src" / "data"
    project_data_dir.mkdir()
    for sample in ["yes.c", "no.c", "silence.c", "unknown.c"]:
        shutil.copy2(kws_testdata_dir / sample, project_data_dir / sample)

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
def uploaded_project(compiled_project):
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
    transport = uploaded_project.transport()
    transport.open()
    out = transport.read(2048, 60)
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


@pytest.mark.requires_hardware
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


@pytest.mark.requires_hardware
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
    tvm.testing.main()
