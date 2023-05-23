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

import os

import tvm

from .utils import build_project_api


@tvm.testing.requires_micro
def test_option_cmsis_path():
    """Test project API without CMSIS_PATH environment variable."""
    cmsis_path = os.environ.get("CMSIS_PATH", None)
    del os.environ["CMSIS_PATH"]
    build_project_api("zephyr")
    os.environ["CMSIS_PATH"] = cmsis_path
