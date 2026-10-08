# isort: skip_file
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
"""Python-side target tag registry.

Importing this package registers all Python-defined target tags.
"""

from . import registry
from . import arm_cpu
from . import riscv_cpu
from . import aws_cpu

from .. import _ffi_api
from ..target import Target

# Target descriptions are available independently of backend Python services.
_kinds = Target.list_kinds() if hasattr(_ffi_api, "ListTargetKinds") else []
if "cuda" in _kinds:
    from . import cuda
if "metal" in _kinds:
    from . import metal
if "opencl" in _kinds or "vulkan" in _kinds:
    from . import adreno
if "hexagon" in _kinds:
    from . import hexagon
if "trn" in _kinds:
    from . import trn

# Validate all tags at import time
registry.list_tags()
