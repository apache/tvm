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
"""Hardware requirements for TIRx codegen tests."""

from pathlib import Path

from tvm_tirx.testing._pytest import (
    _release_cuda_cache_between_tests as _release_cuda_cache_between_tests,
)
from tvm_tirx.testing._pytest import (
    _source_codegen_without_cuda as _source_codegen_without_cuda,
)
from tvm_tirx.testing._pytest import (
    codegen_collection_modifyitems,
)


def pytest_collection_modifyitems(config, items):
    codegen_collection_modifyitems(config, items, Path(__file__).resolve().parent)
