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

import sys
import tvm
import pytest

collect_ignore = []
if sys.platform.startswith("win"):
    collect_ignore.append("frontend/caffe")
    collect_ignore.append("frontend/caffe2")
    collect_ignore.append("frontend/coreml")
    collect_ignore.append("frontend/darknet")
    collect_ignore.append("frontend/keras")
    collect_ignore.append("frontend/mxnet")
    collect_ignore.append("frontend/pytorch")
    collect_ignore.append("frontend/tensorflow")
    collect_ignore.append("frontend/tflite")
    collect_ignore.append("frontend/onnx")
    collect_ignore.append("driver/tvmc/test_autoscheduler.py")
    collect_ignore.append("unittest/test_auto_scheduler_cost_model.py")  # stack overflow
    # collect_ignore.append("unittest/test_auto_scheduler_measure.py") # exception ignored
    collect_ignore.append("unittest/test_auto_scheduler_search_policy.py")  # stack overflow
    # collect_ignore.append("unittest/test_auto_scheduler_measure.py") # exception ignored

    collect_ignore.append("unittest/test_tir_intrin.py")

if tvm.support.libinfo().get("USE_MICRO", "OFF") != "ON":
    collect_ignore.append("unittest/test_micro_transport.py")


def pytest_addoption(parser):
    parser.addoption(
        "--corstone300", action="store_true", default=False, help="Run Corstone-300 FVP tests"
    )


def pytest_collection_modifyitems(config, items):
    skip_corstone300 = pytest.mark.skip(reason="Need --corstone300 option to run")
    skip_non_corstone300 = pytest.mark.skip(reason="Test shold be marked 'corstone300' to run")

    for item in items:
        if config.getoption("--corstone300"):
            if not "corstone300" in item.keywords:
                item.add_marker(skip_non_corstone300)
        else:
            if "corstone300" in item.keywords:
                item.add_marker(skip_corstone300)
