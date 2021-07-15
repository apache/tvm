import pytest

import tvm.target.target

# The models that should pass this configuration. Maps a short, identifying platform string to
# (model, zephyr_board).
PLATFORMS = {
    "spresense_main": ("cxd5602gg", "spresense"),
}


def pytest_addoption(parser):
    parser.addoption(
        "--microtvm-platforms",
        default="spresense_main",
        choices=PLATFORMS.keys(),
        help=(
            "Specify a comma-separated list of test models (i.e. as passed to tvm.target.micro()) "
            "for microTVM tests."
        ),
    )
    parser.addoption(
        "--arduino-cmd", default="arduino-cli", help="Path to `arduino-cli` command for flashing device."
    )
    parser.addoption(
        "--skip-build",
        action="store_true",
        help="If set true, reuses build from the previous test run. Otherwise, build from the scratch.",
    )
    parser.addoption(
        "--tvm-debug",
        action="store_true",
        default=False,
        help="If set true, enable a debug session while the test is running. Before running the test, in a separate shell, you should run: <python -m tvm.exec.microtvm_debug_shell>",
    )


def pytest_generate_tests(metafunc):
    if "platform" in metafunc.fixturenames:
        metafunc.parametrize("platform", metafunc.config.getoption("microtvm_platforms").split(","))


@pytest.fixture
def arduino_cmd(request):
    return request.config.getoption("--arduino-cmd")
