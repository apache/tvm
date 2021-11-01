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
"""Tensor config class to hold tensor scheduling information."""
from typing import List, Union
from enum import IntEnum
import tvm._ffi
from tvm.contrib.ethosu.cascader.stripe_config import StripeConfig

from tvm.runtime import Object

from . import _ffi_api
from .stripe_config import StripeConfig
from .graph import Tensor, BufferMode


class TensorConfigState(IntEnum):
    BOUNDARY = 0
    INTERIOR = 1


@tvm._ffi.register_object("contrib.ethosu.cascader.MemoryRegion")
class MemoryRegion(Object):
    """MemoryRegion class"""

    def __init__(self, name: str, size: int, read_bandwidth: int, write_bandwidth: int):
        self.__init_handle_by_constructor__(
            _ffi_api.MemoryRegion, name, size, read_bandwidth, write_bandwidth
        )


@tvm._ffi.register_object("contrib.ethosu.cascader.TensorConfig")
class TensorConfig(Object):
    """TensorConfig class"""

    def __init__(
        self,
        tensor: Tensor,
        home_region: MemoryRegion,
        state: TensorConfigState,
        buffer_mode: BufferMode,
        stripe_configs: List[StripeConfig],
        copy_tensor: bool = False,
        copy_region: Union[MemoryRegion, None] = None,
    ):
        if copy_region is None:
            copy_region = home_region
        self.__init_handle_by_constructor__(
            _ffi_api.TensorConfig,
            tensor,
            home_region,
            state,
            buffer_mode,
            stripe_configs,
            copy_tensor,
            copy_region,
        )

    def get_buffer_size(self):
        return _ffi_api.TensorConfigGetBufferSize(self)

    @property
    def tensor(self):
        return self._tensor

    @property
    def home_region(self):
        return self._home_region

    @property
    def state(self):
        return TensorConfigState(self._state)

    @property
    def buffer_mode(self):
        return BufferMode(self._buffer_mode)

    @property
    def stripe_configs(self):
        return list(self._stripe_configs)

    @property
    def copy_tensor(self):
        return bool(self._copy_tensor)

    @property
    def copy_region(self):
        return self._copy_region

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return _ffi_api.TensorConfigEqual(self, other)

    def __repr__(self):
        return (
            f"TensorConfig(tensor={self.tensor}, "
            f"home_region={self.home_region.name}, "
            f"state={self.state.name}, "
            f"buffer_mode={self.buffer_mode.name}, "
            f"stripe_configs={self.stripe_configs}, "
            f"copy_tensor={self.copy_tensor}, "
            f"copy_region={self.copy_region.name}"
        )
