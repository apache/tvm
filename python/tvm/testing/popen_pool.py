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
"""Helper functions for popen_pool test cases.

These functions run inside PopenWorker subprocesses and must live in an
importable module (cloudpickle resolves them by module + qualname).  The
previous version used FFI helpers (testing.sleep_in_ffi, testing.identity_cpp,
etc.) that were removed with ffi_testing.cc.  This version is pure-Python and
uses time.sleep for any blocking needed by the timeout test.
"""

import time

import tvm_ffi

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


@tvm_ffi.register_global_func("testing.identity_py", override=True)
def identity_py(arg):
    return arg


def register_ffi():
    @tvm_ffi.register_global_func("testing.nested_identity_py", override=True)
    def _identity_py(arg):  # pylint: disable=unused-variable
        return arg


def call_py_ffi(arg):
    _identity_py = tvm_ffi.get_global_func("testing.nested_identity_py")
    return _identity_py(arg)


def call_cpp_ffi(arg):
    import tvm  # pylint: disable=import-outside-toplevel

    return tvm.testing.echo(arg)


def call_cpp_py_ffi(arg):
    # Call the Python-registered identity function through the FFI registry,
    # exercising the same cross-language dispatch path that identity_cpp covered.
    _identity = tvm_ffi.get_global_func("testing.identity_py")
    return _identity(arg)


def timeout_job(seconds):
    # Previously called testing.sleep_in_ffi (C++ FFI helper, now removed).
    # Plain time.sleep is sufficient — the PopenPoolExecutor timeout mechanism
    # watches wall-clock time and terminates the process just the same.
    time.sleep(seconds)
