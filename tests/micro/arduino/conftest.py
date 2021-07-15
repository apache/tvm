import pytest

import tvm.target.target

# The models that should pass this configuration. Maps a short, identifying platform string to
# (model, zephyr_board).
PLATFORMS = {
    "spresense": ("cxd5602gg", "spresense"),
}


def pytest_addoption(parser):
    parser.addoption(
        "--microtvm-platforms",
        default="spresense",
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
        "--run-hardware-tests", action="store_true", help="Run tests that require physical hardware."
    )

def pytest_generate_tests(metafunc):
    if "platform" in metafunc.fixturenames:
        metafunc.parametrize("platform", metafunc.config.getoption("microtvm_platforms").split(","))


@pytest.fixture
def arduino_cmd(request):
    return request.config.getoption("--arduino-cmd")

@pytest.fixture
def run_hardware_tests(request):
    return request.config.getoption("--run-hardware-tests")
