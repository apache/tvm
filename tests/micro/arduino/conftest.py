import pytest

import tvm.target.target

# The models that should pass this configuration. Maps a short, identifying platform string to
# (model, zephyr_board).
PLATFORMS = {
    "spresense": ("cxd5602gg", "spresense"),
    "nano33ble": ("nRF52840", "nano33ble"),
}


def pytest_addoption(parser):
    parser.addoption(
        "--platform",
        default="spresense",
        choices=PLATFORMS.keys(),
        help="Target platform for microTVM tests.",
    )
    parser.addoption(
        "--arduino-cmd",
        default="arduino-cli",
        help="Path to `arduino-cli` command for flashing device.",
    )
    parser.addoption(
        "--run-hardware-tests",
        action="store_true",
        help="Run tests that require physical hardware.",
    )


# TODO re-add parameterization
@pytest.fixture(scope="session")
def platform(request):
    return request.config.getoption("--platform")


@pytest.fixture(scope="session")
def arduino_cmd(request):
    return request.config.getoption("--arduino-cmd")


@pytest.fixture(scope="session")
def run_hardware_tests(request):
    return request.config.getoption("--run-hardware-tests")
