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

"""Vendored RK3588 NPU core: hardware definitions, register command generation,
and data layout utilities.

Originally from rknpu-compiler (MIT license). Imports adjusted for in-tree use.
"""

from .regcmd_gen import RegCmdGenerator, compute_m_tile  # noqa: F401
from .abstract import (  # noqa: F401
    AbstractMatmulTask, AbstractConv2DTask,
    AbstractElementwiseTask, AbstractMaxPoolTask,
)
from .handles import TensorHandle  # noqa: F401
from .alignment import align_up, pad_m  # noqa: F401
from .hardware import REGCMD_COUNT, Task  # noqa: F401
from .layout import weight_index_fp16, weight_index_int8  # noqa: F401
