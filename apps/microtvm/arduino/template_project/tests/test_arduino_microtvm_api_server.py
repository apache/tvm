from unittest import mock
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import microtvm_api_server

sys.path.pop(0)


class TestGenerateProject:
    DEFAULT_OPTIONS = {"arduino_cli_cmd": "arduino-cli", "arduino_board": "nano33ble"}

    def test_print_c_array(self):
        handler = microtvm_api_server.Handler()
        c_arr = handler._print_c_array([1, 32, 32, 3])
        assert c_arr == "{1, 32, 32, 3}"

    def _set_pathlib_path_exists(self, value):
        with mock.patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = value

    @mock.patch("pathlib.Path")
    def test_find_modified_include_path(self, mock_pathlib_path):
        handler = microtvm_api_server.Handler()

        project_dir = mock_pathlib_path("/dummy/project")
        file_path = (
            project_dir
            / "src"
            / "standalone_crt"
            / "src"
            / "runtime"
            / "crt"
            / "graph_executor"
            / "load_json.c"
        )

        # Should return C standard libs unmodified
        clib_output = handler._find_modified_include_path(project_dir, file_path, "math.h")
        assert clib_output == "math.h"

        # If import already works, should return unmodified
        valid_arduino_import = "../../../../include/tvm/runtime/crt/platform.h"
        self._set_pathlib_path_exists(True)
        valid_output = handler._find_modified_include_path(
            project_dir, file_path, valid_arduino_import
        )
        assert valid_output == valid_arduino_import

    BOARD_CONNECTED_OUTPUT = bytes(
        "Port         Type              Board Name          FQBN                        Core             \n"
        "/dev/ttyACM0 Serial Port (USB) Arduino Nano 33 BLE arduino:mbed_nano:nano33ble arduino:mbed_nano\n"
        "/dev/ttyS4   Serial Port       Unknown                                                          \n"
        "\n",
        "utf-8",
    )
    BOARD_DISCONNECTED_OUTPUT = bytes(
        "Port       Type        Board Name FQBN Core\n"
        "/dev/ttyS4 Serial Port Unknown             \n"
        "\n",
        "utf-8",
    )

    @mock.patch("subprocess.check_output")
    def test_auto_detect_port(self, mock_subprocess_check_output):
        process_mock = mock.Mock()
        handler = microtvm_api_server.Handler()

        # Test it returns the correct port when a board is connected
        mock_subprocess_check_output.return_value = self.BOARD_CONNECTED_OUTPUT
        detected_port = handler._auto_detect_port(self.DEFAULT_OPTIONS)
        assert detected_port == "/dev/ttyACM0"

        # Test it raises an exception when no board is connected
        mock_subprocess_check_output.return_value = self.BOARD_DISCONNECTED_OUTPUT
        with pytest.raises(microtvm_api_server.BoardAutodetectFailed):
            handler._auto_detect_port(self.DEFAULT_OPTIONS)

    @mock.patch("subprocess.check_call")
    def test_flash(self, mock_subprocess_check_call):
        handler = microtvm_api_server.Handler()
        handler._port = "/dev/ttyACM0"

        # Test no exception thrown when code 0 returned
        mock_subprocess_check_call.return_value = 0
        handler.flash(self.DEFAULT_OPTIONS)
        mock_subprocess_check_call.assert_called_once()

        # Test InvalidPortException raised when port incorrect
        mock_subprocess_check_call.return_value = 2
        with pytest.raises(microtvm_api_server.InvalidPortException):
            handler.flash(self.DEFAULT_OPTIONS)

        # Test SketchUploadException raised for other issues
        mock_subprocess_check_call.return_value = 1
        with pytest.raises(microtvm_api_server.SketchUploadException):
            handler.flash(self.DEFAULT_OPTIONS)
