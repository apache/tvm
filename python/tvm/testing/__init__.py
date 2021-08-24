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
from .utils import assert_allclose, assert_prim_expr_equal, check_bool_expr_is_true
from .utils import check_int_constraints_trans_consistency, check_numerical_grads
from .utils import device_enabled, enabled_targets, exclude_targets
from .utils import fixture, parameter, parameters, parametrize_targets, uses_gpu
from .utils import known_failing_targets, requires_cuda, requires_cudagraph
from .utils import requires_gpu, requires_llvm, requires_rocm, requires_rpc
from .utils import requires_tensorcore, requires_metal, requires_micro, requires_opencl
from .utils import _auto_parametrize_target, _count_num_fixture_uses
from .utils import _remove_global_fixture_definitions, _parametrize_correlated_parameters
from .utils import _pytest_target_params, identity_after, terminate_self

from ._ffi_api import nop, echo, device_test, run_check_signal, object_use_count
from ._ffi_api import test_wrap_callback, test_raise_error_callback, test_check_eq_callback
from ._ffi_api import ErrorTest, FrontendTestModule, identity_cpp

from .popen_pool import initializer, after_initializer, register_ffi, call_cpp_ffi
from .popen_pool import call_py_ffi, call_cpp_py_ffi

from . import auto_scheduler
