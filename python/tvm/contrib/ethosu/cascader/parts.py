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
"""Parts used by the NPU cascader."""
from typing import List
import tvm._ffi

from .propagator import Propagator
from .graph import Part, TESubgraph
from . import _ffi_api


@tvm._ffi.register_object("contrib.ethosu.cascader.InlinePart")
class InlinePart(Part):
    """InlinePart class"""

    def __init__(
        self,
        te_subgraph: TESubgraph,
        propagators: List[Propagator],
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.InlinePart,
            te_subgraph.input_tensors,
            te_subgraph.output_tensor,
            propagators,
        )
