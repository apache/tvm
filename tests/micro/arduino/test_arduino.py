import datetime
import os
import pathlib
import sys

import pytest
import tflite
import tvm
from tvm import micro, relay

import conftest

PLATFORMS = conftest.PLATFORMS

def _generate_project(model, target, arduino_board, arduino_cmd, mod, build_config):
    parent_dir = os.path.dirname(__file__)
    filename = os.path.splitext(os.path.basename(__file__))[0]
    prev_build = f"{os.path.join(parent_dir, 'archive')}_{filename}_{arduino_board}_last_build.micro"
    workspace_root = os.path.join(
        f"{os.path.join(parent_dir, 'workspace')}_{filename}_{arduino_board}",
        datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
    )
    workspace_parent = os.path.dirname(workspace_root)
    if not os.path.exists(workspace_parent):
        os.makedirs(workspace_parent)
    workspace = tvm.micro.Workspace(debug=True, root=workspace_root)

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
def _generate_yes_no_project(platform, arduino_cmd):
    current_dir = os.path.dirname(__file__)
    model, arduino_board = PLATFORMS[platform]
    #target = tvm.target.target.micro(model, options=["-link-params=1"])
    build_config = {}

    with open(f"{current_dir}/testdata/yes_no.tflite", "rb") as f:
        tflite_model_buf = f.read()
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    mod, params = relay.frontend.from_tflite(tflite_model)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = relay.build(mod, TARGET, params=params)

    return _generate_project(model, TARGET, arduino_board, arduino_cmd, mod, build_config)
    #project.build()
    #project.flash()
    #return tvm.micro.Session(project.transport())


def test_generate_yes_no_project(platform, arduino_cmd):
    workspace, project = _generate_yes_no_project(platform, arduino_cmd)

    # Ensure top-level directory structure looks good
    assert(os.listdir(workspace.path) == ['project'])

    project_dir = pathlib.Path(workspace.path) / "project"
    assert(set([
        'microtvm_api_server.py', 'project.ino', 'src']
    ).issubset(os.listdir(project_dir)))

    source_dir = project_dir / "src"
    assert(set(os.listdir(source_dir)) == set([
        'model', 'standalone_crt', 'implementation.c',
        'model.cpp', 'model.h', 'parameters.h'
    ]))


    # Ensure model was connected and graph_json compiled
    model_dir = source_dir / "model"
    assert(set(os.listdir(model_dir)) == set([
        'default_lib0.c', 'default_lib1.c', 'graph_json.c', 'model.tar'
    ]))
    with (model_dir / "graph_json.c").open() as f:
        graph_json_c = f.read()
        assert("static const char* graph_json" in graph_json_c)


    # Ensure parameters.h was templated with correct information
    # for our yes/no model
    with (source_dir / "parameters.h").open() as f:
        parameters_h = f.read()
        assert("INPUT_DATA_SHAPE[] = {1, 1960};" in parameters_h)


    # Check one file to ensure imports were rerouted
    runtime_c_path = source_dir / "standalone_crt" / "src" / "runtime"
    load_json_path = runtime_c_path / "crt" / "graph_executor" / "load_json.c"
    assert(load_json_path.exists())

    with (load_json_path).open() as f:
        load_json_c = f.read()
        assert('#include "stdlib.h"' in load_json_c)
        assert('include/tvm/runtime/crt/platform.h' in load_json_c)


def test_compile_yes_no_project(platform, arduino_cmd):
    workspace, project = _generate_yes_no_project(platform, arduino_cmd)
    project.build()

    # Make sure build_dir is not empty
    build_dir = pathlib.Path(workspace.path) / "project" / "build"
    assert(build_dir.exists())
    first_build_file = next(build_dir.iterdir(), None)
    assert(first_build_file is not None)


def test_upload_yes_no_project(platform, arduino_cmd, run_hardware_tests):
    if not run_hardware_tests:
        pytest.skip("skipping hardware tests")

    workspace, project = _generate_yes_no_project(platform, arduino_cmd)
    project.build()
    project.flash()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
