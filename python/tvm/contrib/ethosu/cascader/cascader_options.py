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
"""Object to hold options for the NPU cascader"""
import tvm._ffi

from tvm.runtime import Object

from . import _ffi_api
from .tensor_config import MemoryRegion


@tvm._ffi.register_object("contrib.ethosu.cascader.CascaderOptions")
class CascaderOptions(Object):
    """
    A class to hold configuration options for the cascader.

    Attributes
    ----------
    cascade_region : MemoryRegion
        The MemoryRegion to place cascading buffers into.
    max_proposals : int
        The maximum number of Proposals to generate.
    stripe_factors : int
        How many striping factors to try per axis.
    max_plan_size : int
        The maximum number of Parts in a Plan.
    always_copy_size : int
        The maximum size of a Tensor that will always be copied into the cascade region.

    """

    def __init__(
        self,
        cascade_region: MemoryRegion,
        max_proposals: int,
        stripe_factors: int,
        max_plan_size: int,
        always_copy_size: int,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.CascaderOptions,
            cascade_region,
            max_proposals,
            stripe_factors,
            max_plan_size,
            always_copy_size,
        )
