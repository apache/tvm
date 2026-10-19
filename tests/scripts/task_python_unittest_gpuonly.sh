#!/usr/bin/env bash
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

set -euxo pipefail

# Every GPU test carries the `gpu` marker; the specific backend is gated by skipif.
export TVM_TEST_MARKER="gpu"

# Keep dedicated Vulkan coverage limited to its existing smoke-test files.
./tests/scripts/task_python_unittest.sh \
    tests/python tirx/tests/python --ignore=tirx/tests/python/backend/vulkan

./tests/scripts/task_python_unittest.sh \
    tirx/tests/python/backend/vulkan/codegen/test_basic.py \
    tirx/tests/python/backend/vulkan/test_module_export.py
