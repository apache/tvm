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

import pathlib
import re
import sys

import pytest

import conftest
from tvm.micro.project_api.server import ServerError


# A new project and workspace dir is created for EVERY test
@pytest.fixture
def workspace_dir(request, board):
    return conftest.make_workspace_dir("arduino_error_detection", board)


@pytest.fixture
def project(board, arduino_cli_cmd, tvm_debug, workspace_dir):
    return conftest.make_kws_project(board, arduino_cli_cmd, tvm_debug, workspace_dir)


def test_blank_project_compiles(workspace_dir, project):
    project.build()


# Add a bug (an extra curly brace) and make sure the project doesn't compile
def test_bugged_project_compile_fails(workspace_dir, project):
    with open(workspace_dir / "project" / "project.ino", "a") as main_file:
        main_file.write("}\n")
    with pytest.raises(ServerError):
        project.build()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
