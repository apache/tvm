import datetime
import pathlib

import pytest

import tvm.target.target

# The models that should pass this configuration. Maps a short, identifying platform string to
# (model, zephyr_board).
PLATFORMS = {
    "due": ("sam3x8e", "due"),
    "feathers2": ("esp32", "feathers2"),
    "nano33ble": ("nrf52840", "nano33ble"),
    "pybadge": ("atsamd51", "pybadge"),
    "spresense": ("cxd5602gg", "spresense"),
    "teensy40": ("imxrt1060", "teensy40"),
    "teensy41": ("imxrt1060", "teensy41"),
}

TEMPLATE_PROJECT_DIR = (
    pathlib.Path(__file__).parent
    / ".."
    / ".."
    / ".."
    / "apps"
    / "microtvm"
    / "arduino"
    / "template_project"
).resolve()


def pytest_addoption(parser):
    parser.addoption(
        "--microtvm-platforms",
        default=["due"],
        nargs="*",
        choices=PLATFORMS.keys(),
        help="Target platforms for microTVM tests.",
    )
    parser.addoption(
        "--arduino-cli-cmd",
        default="arduino-cli",
        help="Path to `arduino-cli` command for flashing device.",
    )
    parser.addoption(
        "--run-hardware-tests",
        action="store_true",
        help="Run tests that require physical hardware.",
    )
    parser.addoption(
        "--tvm-debug",
        action="store_true",
        default=False,
        help="If set true, enable a debug session while the test is running. Before running the test, in a separate shell, you should run: <python -m tvm.exec.microtvm_debug_shell>",
    )


# We might do project generation differently for different boards in the future
# (to take advantage of multiple cores / external memory / etc.), so all tests
# are parameterized by board
def pytest_generate_tests(metafunc):
    platforms = metafunc.config.getoption("microtvm_platforms")
    metafunc.parametrize("platform", platforms, scope="session")


@pytest.fixture(scope="session")
def arduino_cli_cmd(request):
    return request.config.getoption("--arduino-cli-cmd")


@pytest.fixture(scope="session")
def tvm_debug(request):
    return request.config.getoption("--tvm-debug")


@pytest.fixture(scope="session")
def run_hardware_tests(request):
    return request.config.getoption("--run-hardware-tests")


def make_workspace_dir(test_name, platform):
    _, arduino_board = PLATFORMS[platform]
    filepath = pathlib.Path(__file__)
    board_workspace = (
        filepath.parent
        / f"workspace_{test_name}_{arduino_board}"
        / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    )

    number = 0
    while board_workspace.exists():
        number += 1
        board_workspace = pathlib.Path(str(board_workspace) + f"-{number}")
    board_workspace.parent.mkdir(exist_ok=True, parents=True)
    t = tvm.contrib.utils.tempdir(board_workspace)
    # time.sleep(200)
    return t
