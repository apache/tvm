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

import pytest

from tvm.micro.project_api.server import ServerError

import test_utils
import tvm.testing


@pytest.fixture
def project(board, microtvm_debug, workspace_dir, serial_number):
    return test_utils.make_kws_project(board, microtvm_debug, workspace_dir, serial_number)


def test_blank_project_compiles(workspace_dir, project):
    project.build()


# Add a bug (an extra curly brace) and make sure the project doesn't compile
def test_bugged_project_compile_fails(workspace_dir, project):
    with open(workspace_dir / "project" / "project.ino", "a") as main_file:
        main_file.write("}\n")
    with pytest.raises(ServerError):
        project.build()


if __name__ == "__main__":
    tvm.testing.main()
