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
# pylint: disable=invalid-name
"""Block config to hold an output block shape and a corresponding input block shape"""
from typing import List
import tvm._ffi

from tvm.runtime import Object

from . import _ffi_api


@tvm._ffi.register_object("contrib.ethosu.cascader.BlockConfig")
class BlockConfig(Object):
    """BlockConfig class"""

    def __init__(
        self,
        input_shape: List[int],
        output_shape: List[int],
        compute_cycles: int,
        output_cycles: int,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.BlockConfig, input_shape, output_shape, compute_cycles, output_cycles
        )

    @property
    def input_shape(self) -> List[int]:
        return list(self._input_shape)

    @property
    def output_shape(self) -> List[int]:
        return list(self._output_shape)

    @property
    def compute_cycles(self) -> int:
        return int(self._compute_cycles)

    @property
    def output_cycles(self) -> int:
        return int(self._output_cycles)

    def __ge__(self, other: "BlockConfig"):
        if len(self.output_shape) != len(other.output_shape):
            return False

        return all(a >= b for a, b in zip(self.output_shape, other.output_shape))

    def __lt__(self, other: "BlockConfig"):
        if len(self.output_shape) != len(other.output_shape):
            return False

        return other >= self

    def __repr__(self) -> str:
        return f"BlockConfig(output_shape={self.output_shape})"
