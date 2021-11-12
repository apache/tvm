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
import numpy as np
import pytest
import tvm


def test_make_se_scope_for_device():
    se_scope = tvm.target.make_se_scope(tvm.device("cuda"))
    assert se_scope.device_type == 2
    # ie kDLCUDA
    assert se_scope.virtual_device_id == 0
    assert se_scope.target is None
    assert se_scope.memory_scope == ""


def test_make_se_scope_for_device_and_target():
    target = tvm.target.Target("cuda")
    se_scope = tvm.target.make_se_scope(tvm.device("cuda"), target)
    assert se_scope.device_type == 2  # ie kDLCUDA
    assert se_scope.target == target
    assert se_scope.memory_scope == ""


def test_make_se_scope_for_device_target_and_memory_scope():
    target = tvm.target.Target("cuda")
    scope = "local"
    se_scope = tvm.target.make_se_scope(tvm.device("cuda"), target, scope)
    assert se_scope.device_type == 2  # ie kDLCUDA
    assert se_scope.target == target
    assert se_scope.memory_scope == scope


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__] + sys.argv[1:]))
