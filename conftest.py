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
import sys
from pathlib import Path

pytest_plugins = ["tvm.testing.plugin"]
IS_IN_CI = os.getenv("CI", "") == "true"
REPO_ROOT = Path(__file__).resolve().parent


def pytest_sessionstart():
    if IS_IN_CI:
        hook_script_dir = REPO_ROOT / "tests" / "scripts" / "request_hook"
        sys.path.append(str(hook_script_dir))
        import request_hook  # pylint: disable=import-outside-toplevel

        request_hook.init()
