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
# pylint: disable=invalid-name, missing-function-docstring
"""Common functions for popen_pool test cases"""
import tvm
from . import _ffi_api

TEST_GLOBAL_STATE_1 = 0
TEST_GLOBAL_STATE_2 = 0
TEST_GLOBAL_STATE_3 = 0


def initializer(test_global_state_1, test_global_state_2, test_global_state_3):
    global TEST_GLOBAL_STATE_1, TEST_GLOBAL_STATE_2, TEST_GLOBAL_STATE_3
    TEST_GLOBAL_STATE_1 = test_global_state_1
    TEST_GLOBAL_STATE_2 = test_global_state_2
    TEST_GLOBAL_STATE_3 = test_global_state_3


def after_initializer():
    global TEST_GLOBAL_STATE_1, TEST_GLOBAL_STATE_2, TEST_GLOBAL_STATE_3
    return TEST_GLOBAL_STATE_1, TEST_GLOBAL_STATE_2, TEST_GLOBAL_STATE_3


@tvm._ffi.register_func("testing.identity_py")
def identity_py(arg):
    return arg


def register_ffi():
    @tvm._ffi.register_func("testing.nested_identity_py")
    def _identity_py(arg):  # pylint: disable=unused-variable
        return arg


def call_py_ffi(arg):
    _identity_py = tvm._ffi.get_global_func("testing.nested_identity_py")
    return _identity_py(arg)


def call_cpp_ffi(arg):
    return tvm.testing.echo(arg)


def call_cpp_py_ffi(arg):
    return tvm.testing.identity_cpp(arg)


def fast_summation(n):
    return n * (n + 1) // 2


def slow_summation(n):
    r = 0
    for i in range(0, n + 1):
        r += i
    return r


def timeout_job(n):
    _ffi_api.sleep_in_ffi(n * 1.5)
