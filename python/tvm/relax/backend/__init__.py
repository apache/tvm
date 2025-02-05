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
"""Relax backends"""

from tvm.target import Target

from . import contrib
from .dispatch_sampling import DispatchSampling
from .dispatch_sort_scan import DispatchSortScan
from .pattern_registry import get_pattern, get_patterns_with_prefix


def get_default_pipeline(target: Target):
    """Get the default Relax compilation pipeline for the given target."""
    if target.kind.name == "cuda":
        from . import cuda  # pylint: disable=import-outside-toplevel

        return cuda.get_default_pipeline(target)
    if target.kind.name == "llvm":
        from . import cpu_generic  # pylint: disable=import-outside-toplevel

        return cpu_generic.get_default_pipeline(target)
    # Todo(tvm-team): support gpu-generic
    raise ValueError(
        f"Target {target} is not yet supported by default pipeline. "
        "Please lower and build the IRModule manually."
    )
