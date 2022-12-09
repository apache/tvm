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

import subprocess
import sys
from pathlib import Path
from unittest import mock

from packaging import version
import pytest

import tvm
from tvm.micro.project_api import server

sys.path.insert(0, tvm.micro.get_microtvm_template_projects("arduino"))
import microtvm_api_server

sys.path.pop(0)


class TestGenerateProject:
    DEFAULT_OPTIONS = {"arduino_cli_cmd": "arduino-cli", "board": "nano33ble"}

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

    # Format for arduino-cli v0.18.2
    BOARD_CONNECTED_V18 = (
        "Port         Type              Board Name          FQBN                        Core             \n"
        "/dev/ttyACM0 Serial Port (USB) Arduino Nano 33 BLE arduino:mbed_nano:nano33ble arduino:mbed_nano\n"
        "/dev/ttyACM1 Serial Port (USB) Arduino Nano 33     arduino:mbed_nano:nano33    arduino:mbed_nano\n"
        "/dev/ttyS4   Serial Port       Unknown                                                          \n"
        "\n"
    )
    # Format for arduino-cli v0.21.1 and above
    BOARD_CONNECTED_V21 = (
        "Port         Protocol Type Board Name FQBN                        Core             \n"
        "/dev/ttyACM0 serial                   arduino:mbed_nano:nano33ble arduino:mbed_nano\n"
        "\n"
    )
    BOARD_DISCONNECTED_V21 = (
        "Port       Protocol Type        Board Name FQBN Core\n"
        "/dev/ttyS4 serial   Serial Port Unknown\n"
        "\n"
    )

    def test_parse_connected_boards(self):
        h = microtvm_api_server.Handler()
        boards = h._parse_connected_boards(self.BOARD_CONNECTED_V21)
        assert list(boards) == [
            {
                "port": "/dev/ttyACM0",
                "protocol": "serial",
                "type": "",
                "board name": "",
                "fqbn": "arduino:mbed_nano:nano33ble",
                "core": "arduino:mbed_nano",
            }
        ]

    @mock.patch("subprocess.run")
    def test_auto_detect_port(self, mock_run):
        process_mock = mock.Mock()
        handler = microtvm_api_server.Handler()

        # Test it returns the correct port when a board is connected
        mock_run.return_value.stdout = bytes(self.BOARD_CONNECTED_V18, "utf-8")
        assert handler._auto_detect_port(self.DEFAULT_OPTIONS) == "/dev/ttyACM0"

        # Should work with old or new arduino-cli version
        mock_run.return_value.stdout = bytes(self.BOARD_CONNECTED_V21, "utf-8")
        assert handler._auto_detect_port(self.DEFAULT_OPTIONS) == "/dev/ttyACM0"

        # Test it raises an exception when no board is connected
        mock_run.return_value.stdout = bytes(self.BOARD_DISCONNECTED_V21, "utf-8")
        with pytest.raises(microtvm_api_server.BoardAutodetectFailed):
            handler._auto_detect_port(self.DEFAULT_OPTIONS)

        # Test that the FQBN needs to match EXACTLY
        handler._get_fqbn = mock.MagicMock(return_value="arduino:mbed_nano:nano33")
        mock_run.return_value.stdout = bytes(self.BOARD_CONNECTED_V18, "utf-8")
        assert (
            handler._auto_detect_port({**self.DEFAULT_OPTIONS, "board": "nano33"}) == "/dev/ttyACM1"
        )

    BAD_CLI_VERSION = "arduino-cli  Version: 0.7.1 Commit: 7668c465 Date: 2019-12-31T18:24:32Z\n"
    GOOD_CLI_VERSION = "arduino-cli  Version: 0.21.1 Commit: 9fcbb392 Date: 2022-02-24T15:41:45Z\n"

    @mock.patch("subprocess.run")
    def test_auto_detect_port(self, mock_run):
        handler = microtvm_api_server.Handler()
        mock_run.return_value.stdout = bytes(self.GOOD_CLI_VERSION, "utf-8")
        arduino_cli_cmd = self.DEFAULT_OPTIONS.get("arduino_cli_cmd")
        warning_as_error = self.DEFAULT_OPTIONS.get("warning_as_error")

        cli_command = handler._get_arduino_cli_cmd(arduino_cli_cmd)
        handler._check_platform_version(cli_command=cli_command, warning_as_error=warning_as_error)
        assert handler._version == version.parse("0.21.1")

        handler = microtvm_api_server.Handler()
        mock_run.return_value.stdout = bytes(self.BAD_CLI_VERSION, "utf-8")
        with pytest.raises(server.ServerError) as error:
            handler._check_platform_version(cli_command=cli_command, warning_as_error=True)
        mock_run.reset_mock()

    @mock.patch("subprocess.run")
    def test_flash_retry(self, mock_run):
        mock_run.return_value.stdout = bytes(self.GOOD_CLI_VERSION, "utf-8")

        def side_effect(cmd, *args, **kwargs):
            if cmd[1] == "flash":
                raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
            return mock.DEFAULT

        mock_run.side_effect = side_effect

        handler = microtvm_api_server.Handler()
        handler._port = "/dev/ttyACM0"

        # handler.flash will try flashing `handler.FLASH_MAX_RETRIES` times,
        # after which it will raise a TimeoutExpired exception of its own
        with pytest.raises(RuntimeError):
            handler.flash(self.DEFAULT_OPTIONS)

        # Test we checked version then called upload once per retry attempt,
        # plus once to verify arduino-cli version.
        assert mock_run.call_count == handler.FLASH_MAX_RETRIES + 1

    @mock.patch("subprocess.run")
    def test_flash(self, mock_run):
        mock_run.return_value.stdout = bytes(self.GOOD_CLI_VERSION, "utf-8")

        handler = microtvm_api_server.Handler()
        handler._port = "/dev/ttyACM0"

        # Test no exception thrown when command works
        handler.flash(self.DEFAULT_OPTIONS)

        # Test we checked version then called upload
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0][0] == (["arduino-cli", "version"],)
        assert mock_run.call_args_list[1][0][0][0:2] == ["make", "flash"]
        mock_run.reset_mock()

        # Test exception raised when `arduino-cli upload` returns error code
        mock_run.side_effect = subprocess.CalledProcessError(2, [])
        with pytest.raises(subprocess.CalledProcessError):
            handler.flash(self.DEFAULT_OPTIONS)

        # Version information should be cached and not checked again
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0][0:2] == ["make", "flash"]


if __name__ == "__main__":
    tvm.testing.main()
