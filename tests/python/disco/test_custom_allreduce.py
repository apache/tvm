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

import enum
from functools import reduce
from itertools import product

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.runtime import DataType, ShapeTuple, disco
from tvm.runtime.disco import Session


class AllReduceStrategyType(enum.IntEnum):
    ONESHOT = 1
    TWOSHOT = 2


_shapes = [(2, 3), (3, 4), (128, 128)]

_strategies = [
    AllReduceStrategyType.ONESHOT,
    AllReduceStrategyType.TWOSHOT,
]

_ccl = [ccl for ccl in tvm.get_global_func("runtime.disco.compiled_ccl")() if ccl == "nccl"]


@pytest.mark.parametrize("shape", _shapes)
@pytest.mark.parametrize("ccl", _ccl)
@pytest.mark.parametrize("strategy", _strategies)
def test_allreduce(shape, ccl, strategy):
    devices = [0, 1]
    sess: Session = disco.ProcessSession(num_workers=len(devices))
    sess.init_ccl(ccl, *devices)

    num_elements = reduce(lambda x, y: x * y, shape)
    dtype = "float32"
    falloc_ipc_storage = sess.get_global_func("runtime.disco.cuda_ipc.alloc_storage")
    falloc_tensor = sess.get_global_func("vm.builtin.alloc_tensor")
    fallreduce = sess.get_global_func("runtime.disco.cuda_ipc.custom_allreduce")
    d_storage = sess.call_packed(falloc_ipc_storage, ShapeTuple(shape), DataType(dtype))
    d_input = sess.call_packed(falloc_tensor, d_storage, 0, ShapeTuple(shape), DataType(dtype))

    array_1 = np.arange(num_elements, dtype="float32").reshape(*shape)
    array_2 = np.arange(start=1, stop=-(num_elements - 1), step=-1, dtype="float32").reshape(*shape)
    d_input.debug_copy_from(0, array_1)
    d_input.debug_copy_from(1, array_2)
    d_output = sess.empty(shape, "float32")

    sess.call_packed(fallreduce, d_input, strategy, d_output)
    result_1 = d_output.debug_get_from_remote(0).numpy()
    result_2 = d_output.debug_get_from_remote(1).numpy()
    expected = np.add(array_1, array_2)
    np.testing.assert_equal(result_1, expected)
    np.testing.assert_equal(result_2, expected)


if __name__ == "__main__":
    for shape, strategy in product(_shapes, _strategies):
        test_allreduce(shape, "nccl", strategy)
