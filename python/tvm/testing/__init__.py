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

# pylint: disable=redefined-builtin, wildcard-import
"""Utility Python functions for TVM testing"""
from . import auto_scheduler, autotvm
from ._ffi_api import (
    ErrorTest,
    FrontendTestModule,
    device_test,
    echo,
    identity_cpp,
    nop,
    object_use_count,
    run_check_signal,
    test_check_eq_callback,
    test_raise_error_callback,
    test_wrap_callback,
)
from .popen_pool import (
    after_initializer,
    call_cpp_ffi,
    call_cpp_py_ffi,
    call_py_ffi,
    fast_summation,
    initializer,
    register_ffi,
    slow_summation,
    timeout_job,
)
from .runner import local_run, rpc_run
from .utils import *
