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

import re
import tvm
import numpy as np
from tvm import relay
from tvm.relay import testing
from tvm.contrib import utils
from utils.adreno_utils import gpu_preprocess, build_run_compare, build_run_compare_vm
import pytest

executor_type = tvm.testing.parameter("ge")
dtype = tvm.testing.parameter("float32")


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_transpose_adreno(remote, target, executor_type, dtype):
    input_shape = (1, 256, 100, 100)
    filter_shape = (256, 64, 4, 4)
    channels = 64
    kernel_size = (4, 4)
    strides = (2, 2)
    padding = (1, 1, 1, 1)
    x = relay.var("data", shape=input_shape, dtype=dtype)
    w = relay.var("weight", shape=filter_shape, dtype=dtype)

    y = relay.nn.conv2d_transpose(
        x,
        w,
        channels=channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_layout="IOHW",
        data_layout="NCHW",
    )

    mod = relay.Function([x, w], y)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    initializer("weight", filter_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
    }

    if executor_type == "ge":
        build_run_compare(
            remote, mod, params1, {"data": input_shape}, {"data": dtype}, target, [], gpu_preprocess
        )
    else:
        build_run_compare_vm(
            remote, mod, params1, {"data": input_shape}, {"data": dtype}, target, [], gpu_preprocess
        )


if __name__ == "__main__":
    tvm.testing.main()
