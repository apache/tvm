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
from . import _ffi_api


def make_compilation_config(ctxt, targets, host_target=None):
    """Returns a CompilationConfig appropriate for targets and an optional host_target.
    Currently intended just for unit tests and will be replaced by a Python CompilationConfig
    class in the future. Note that targets must be a dictionary from IntImm objects to Targets
    and we do not support any of the lighter-weight conventions used by the various build(...)
    APIs."""
    return _ffi_api.MakeCompilationConfig(ctxt, targets, host_target)
