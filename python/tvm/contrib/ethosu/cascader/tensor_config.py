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
    """
    The 'state' of a TensorConfig as used in the Plan generation algorithm.

    BOUNDARY - Should describe a Plan input/output Tensor.
    INTERIOR - Should describe an intermediate Tensor in a 'closed' Plan.

    """

    BOUNDARY = 0
    INTERIOR = 1


@tvm._ffi.register_object("contrib.ethosu.cascader.MemoryRegion")
class MemoryRegion(Object):
    """
    MemoryRegion class to store information about device memories.

    Attributes
    ----------
    name : str
        The name of the region.
    size : int
        The size of the region.
    read_bandwidth : int
        The read bandwidth of the region in bytes per cycle.
    write_bandwidth : int
        The write bandwidth of the region in bytes per cycle.

    """

    def __init__(
        self,
        name: str,
        size: int,
        read_bandwidth: int,
        write_bandwidth: int,
        read_latency: int = 0,
        write_latency: int = 0,
        burst_length: int = 1,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.MemoryRegion,
            name,
            size,
            read_bandwidth,
            write_bandwidth,
            read_latency,
            write_latency,
            burst_length,
        )


@tvm._ffi.register_object("contrib.ethosu.cascader.TensorConfig")
class TensorConfig(Object):
    """
    A class which describes how to realize a Tensor.

    The TensorConfig describes both how a Tensor is scheduled (the order in which it's
    produced/consumed) and how its allocated in memory (which region it should reside in
    and whether it should be copied).

    Attributes
    ----------
    tensor : Tensor
        The Tensor the config applies to.
    home_region : MemoryRegion
        The region where the tensor is allocated.
    state : TensorConfigState
        The state of the TensorConfig.

        The TensorConfigState is only used as part of the Plan generation algorithm. For a Plan
        to be 'closed' (and therefore not subject to any further merging), all the TensorConfigs
        that describe Plan input or output Tensors must be in the 'BOUNDARY' state with the rest
        being 'INTERIOR'. If any of the input or output tensors are described by an 'INTERIOR'
        TensorConfig, then the Plan is 'open' and should be merged with other 'open' Plans until
        the result becomes 'closed'.
    buffer_mode : BufferMode
        The mode in which the buffer should be realized.

        There are multiple buffering strategies by which a tensor may be realized (computed).
        These affect the amount of recomputation necessary as well as the size of buffer required
        to store the tensor. See 'BufferMode' for a description of the allowable buffering modes.
    stripe_configs : List[StringConfig]
       The StripeConfigs with which to compute the tensor.

       The StripeConfigs determine the order in which the elements of the tensor should be
       computed, including potentially computing them multiple times (recompute). Multiple
       StripeConfigs are used over just a single StripeConfig for the case where the tensor is
       consumed by two different Parts executing themselves with different StripeConfigs. In this
       case, there is a StripeConfig per consumer of the tensor.
    copy_tensor : bool, optional
        Whether to copy the tensor.

        While a tensor will originally reside in its home region, the TensorConfig may optionally
        specify that the tensor should be copied (according to the StripeConfigs) into another
        MemoryRegion. As an example for where this may be used, if a weights tensor initially
        resides in slow Flash memory then necessarily the home region will be Flash. However, if
        the weights values are used multiple times by a Part, it may be more performant to choose
        to copy the weights into a faster memory like SRAM.
    copy_region : Union[MemoryRegion, None], optional
        The region to copy the tensor to.

    """

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
        """
        The size of the buffer needed for the TensorConfig.

        The size of buffer necessary to store a tensor being produced using the TensorConfig is
        not necessarily just the size of the tensor. In Plans, a tensor may be being produced and
        consumed in 'stripes' which are smaller than the full tensor. Therefore, the buffer
        necessary to store the tensor may only need to be as large as the stripe. The precise size
        of the buffer will depend both on the BufferMode and StripeConfigs (as well as, of course,
        the Tensor).

        """
        return _ffi_api.TensorConfigGetBufferSize(self)

    @property
    def tensor(self):
        """The Tensor the config applies to."""
        return self._tensor

    @property
    def home_region(self):
        """The region where the tensor is allocated."""
        return self._home_region

    @property
    def state(self):
        """The state of the TensorConfig."""
        return TensorConfigState(self._state)

    @property
    def buffer_mode(self):
        """The mode in which the buffer should be realized."""
        return BufferMode(self._buffer_mode)

    @property
    def stripe_configs(self):
        """The StripeConfigs with which to compute the tensor."""
        return list(self._stripe_configs)

    @property
    def copy_tensor(self):
        """Whether to copy the tensor."""
        return bool(self._copy_tensor)

    @property
    def copy_region(self):
        """The region to copy the tensor to."""
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
