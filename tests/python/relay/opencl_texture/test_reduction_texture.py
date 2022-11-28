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
from utils.adreno_utils import gpu_preprocess, build_run_compare


dtype = tvm.testing.parameter("float32")


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_mean(remote, target, dtype):
    # NCHW
    input_shape = (1, 3, 720, 1280)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    mean = relay.mean(A, axis=1, keepdims=True)
    mod = relay.Function([A], mean)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_argmax(remote, target, dtype):
    # NCHW
    input_shape = (1, 3, 720, 1280)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    argmax = relay.op.argmax(A, axis=[1])
    mod = relay.Function([A], argmax)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


if __name__ == "__main__":
    tvm.testing.main()
