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


executor_type = tvm.testing.parameter("ge", "vm")
dtype = tvm.testing.parameter("float32")


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_mod(remote, target, executor_type, dtype):
    # NCHW
    input_shape = (1, 25, 38, 64)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    scale = relay.const(2.0, dtype=dtype)
    op = relay.mod(A, scale)
    mod = relay.Function([A], op)

    if executor_type == "ge":
        build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_scatter_nd_add(remote, target, executor_type, dtype):
    # NCHW

    A = relay.var("data", shape=(6, 30, 30, 256), dtype=dtype)
    indices = relay.const(tvm.nd.array(np.random.randint(0, 1, (2, 6, 30, 30))), dtype="int64")
    update = relay.const(
        tvm.nd.array(np.random.uniform(-1, 1, size=(50, 50, 256)).astype(dtype)), dtype=dtype
    )
    op = relay.scatter_nd(update, indices, A, mode="add")
    mod = relay.Function([A], op)
    shape_dict = {
        "data": (6, 30, 30, 256),
    }
    dtype_dict = {
        "data": dtype,
    }

    if executor_type == "ge":
        build_run_compare(remote, mod, {}, shape_dict, dtype_dict, target)
    else:
        build_run_compare_vm(remote, mod, {}, shape_dict, dtype_dict, target)


if __name__ == "__main__":
    tvm.testing.main()
