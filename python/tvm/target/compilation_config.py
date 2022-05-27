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
"""Python bindings for creating CompilationConfigs."""
import tvm
from . import _ffi_api


def make_compilation_config(ctxt, target, target_host=None):
    """Returns a CompilationConfig appropriate for target and target_host, using the same
    representation conventions as for the standard build interfaces. Intended only for unit
    testing."""
    raw_targets = tvm.target.Target.canon_multi_target_and_host(target, target_host)
    return _ffi_api.MakeCompilationConfig(ctxt, raw_targets)
