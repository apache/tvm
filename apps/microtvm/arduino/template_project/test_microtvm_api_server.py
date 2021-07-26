from unittest.mock import patch
from pathlib import Path

import pytest

import microtvm_api_server


class TestGenerateProject:
    """A class with common parameters, `param1` and `param2`."""

    def test_print_c_array(self):
        handler = microtvm_api_server.Handler()
        c_arr = handler._print_c_array([1, 32, 32, 3])
        assert c_arr == "{1, 32, 32, 3}"

    def _set_pathlib_path_exists(self, value):
        with patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = value

    @patch("pathlib.Path")
    def test_find_modified_include_path(self, MockPath):
        handler = microtvm_api_server.Handler()

        project_dir = MockPath("/dummy/project")
        file_path = project_dir / "src/standalone_crt/src/runtime/crt/graph_executor/load_json.c"

        # Should return C standard libs unmodified
        clib_output = handler._find_modified_include_path(project_dir, file_path, "math.h")
        assert clib_output == "math.h"

        # If import already works, should return unmodified
        valid_ardino_import = "../../../../include/tvm/runtime/crt/platform.h"
        self._set_pathlib_path_exists(True)
        valid_output = handler._find_modified_include_path(
            project_dir, file_path, valid_ardino_import
        )
        assert valid_output == valid_ardino_import
