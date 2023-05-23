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
from tvm.contrib.ethosu.cascader import (
    StripeConfig,
    Tensor,
    MemoryRegion,
    TensorConfig,
    TensorConfigState,
    BufferMode,
)

import pytest


def test_tensor_config(DRAM, SRAM):
    stripe_config = StripeConfig(
        shape=[1, 2, 3],
        extent=[2, 3, 4],
        strides=[3, 4, 5],
        order=[4, 5, 6],
        stripes=[5, 6, 7],
        offset=[6, 7, 8],
    )
    tensor = Tensor(
        shape=[10, 10, 10],
        dtype="int8",
    )
    home_region = DRAM
    state = TensorConfigState.BOUNDARY
    buffer_mode = BufferMode.ROLLING
    copy_tensor = True
    copy_region = SRAM
    tensor_config = TensorConfig(
        tensor=tensor,
        home_region=home_region,
        state=state,
        buffer_mode=buffer_mode,
        stripe_configs=[stripe_config],
        copy_tensor=copy_tensor,
        copy_region=copy_region,
    )

    assert tensor_config.tensor == tensor
    assert tensor_config.home_region == home_region
    assert tensor_config.state == state
    assert tensor_config.buffer_mode == buffer_mode
    assert tensor_config.stripe_configs == [stripe_config]
    assert tensor_config.copy_tensor == copy_tensor
    assert tensor_config.copy_region == copy_region
    assert hash(tensor_config) != 0


def test_get_rolling_buffer(DRAM):
    stripe_config = StripeConfig(
        shape=[9, 4, 7],
        extent=[9, 16, 21],
        strides=[3, 5, 7],
        order=[1, 3, 2],
        stripes=[1, 3, 3],
        offset=[0, 0, 0],
    )
    tensor = Tensor(shape=[9, 16, 21], dtype="int32", compression_ratio=0.5)
    tensor_config = TensorConfig(
        tensor=tensor,
        home_region=DRAM,
        state=TensorConfigState.BOUNDARY,
        buffer_mode=BufferMode.ROLLING,
        stripe_configs=[stripe_config],
    )

    assert tensor_config.get_buffer_size() == 2016


def test_get_recompute_buffer(DRAM):
    stripe_config = StripeConfig(
        shape=[4, 5, 7],
        extent=[6, 7, 14],
        strides=[2, 3, 7],
        order=[1, 3, 2],
        stripes=[2, 2, 2],
        offset=[0, 0, 0],
    )
    tensor = Tensor(shape=[6, 7, 14], dtype="int32", compression_ratio=0.5)
    tensor_config = TensorConfig(
        tensor=tensor,
        home_region=DRAM,
        state=TensorConfigState.BOUNDARY,
        buffer_mode=BufferMode.RECOMPUTE,
        stripe_configs=[stripe_config],
    )

    assert tensor_config.get_buffer_size() == 280


if __name__ == "__main__":
    tvm.testing.main()
