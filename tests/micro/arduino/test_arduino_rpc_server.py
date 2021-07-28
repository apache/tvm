import datetime
import os
import pathlib
import shutil
import sys
import time

import numpy as np
import pytest
import tflite
import tvm
from tvm import micro, relay

import conftest

"""

"""

PLATFORMS = conftest.PLATFORMS


def _make_session(model, target, arduino_board, arduino_cli_cmd, mod, build_config):
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
        {"arduino_board": arduino_board, "arduino_cli_cmd": arduino_cli_cmd, "project_type": "host_driven", "verbose": 0},
    )
    project.build()
    project.flash()
    return tvm.micro.Session(project.transport())


# This is bad, don't do this
TARGET = "c -keys=cpu -executor=aot -link-params=1 -model=host -runtime=c -unpacked-api=1"

def test_relay(platform, arduino_cli_cmd):
    """Testing a simple relay graph"""
    model, arduino_board = PLATFORMS[platform]
    build_config = {}
    shape = (10,)
    dtype = "int8"

    # Construct Relay program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(np.ones(shape=shape, dtype=dtype)))
    func = relay.Function([x], z)

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(func, target=TARGET)

    with _make_session(model, TARGET, arduino_board, arduino_cli_cmd, mod, build_config) as session:
        graph_mod = tvm.micro.create_local_graph_executor(
            mod.get_graph_json(), session.get_system_lib(), session.device
        )
        graph_mod.set_input(**mod.get_params())
        x_in = np.random.randint(10, size=shape[0], dtype=dtype)
        graph_mod.run(x=x_in)
        result = graph_mod.get_output(0).numpy()
        tvm.testing.assert_allclose(graph_mod.get_input(0).numpy(), x_in)
        tvm.testing.assert_allclose(result, x_in * x_in + 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
