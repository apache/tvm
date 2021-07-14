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

def _make_session(model, target, arduino_board, arduino_cmd, mod, build_config):
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
    print("Outputing workspace root:")
    print(workspace_root)
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
    #project.build()
    #project.flash()
    #return tvm.micro.Session(project.transport())


# This is bad, don't do this
TARGET = "c -keys=cpu -link-params=1 -mcpu=cortex-m33 -model=nrf5340dk -runtime=c -system-lib=1"

def test_generate_yes_no_project(platform, arduino_cmd):
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

    session = _make_session(model, TARGET, arduino_board, arduino_cmd, mod, build_config)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
