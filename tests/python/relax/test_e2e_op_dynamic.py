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
import numpy as np
import pytest
import tvm
from tvm import relax
import tvm.topi.testing
from tvm.relax.transform import LegalizeOps
from tvm.script import relax as R, tir as T
import tvm.testing

# TODO(tvm-team): `tir.transform.DefaultGPUSchedule` does not work.
target, dev = "llvm", tvm.cpu()


def build(mod):
    exe = relax.build(mod, target=target)
    return relax.VirtualMachine(exe, dev)


@pytest.mark.parametrize(
    "begin, end, strides",
    [
        ([0, 2, 4, 4], [5, 5, 7, 8], [1, 1, 2, 3]),
        ([0, 2, 4, 4], [5, 5, 11, 10], [1, 1, 1, 1]),
        ([0, 2, 10, 14], [0, 5, 1, 1], [1, 1, -1, -2]),
    ],
)
def test_dynamic_strided_slice(begin, end, strides):
    # fmt: off
    @tvm.script.ir_module
    class DynamicStridedSlice:
        @R.function
        def main(x: R.Tensor((8, 9, 10, 10), "float32"), begin: R.Tensor((4,),"int64"), end: R.Tensor((4,),"int64"), strides: R.Tensor((4,),"int64")) -> R.Tensor("float32", ndim=4):
            gv: R.Tensor("float32", ndim=4) = R.dynamic_strided_slice(x, begin, end, strides)
            return gv
    # fmt: on
    vm = build(DynamicStridedSlice)

    x_np = np.random.rand(8, 9, 10, 10).astype(np.float32)
    data_nd = tvm.nd.array(x_np, dev)
    begin_nd = tvm.nd.array(np.array(begin).astype("int64"), dev)
    end_nd = tvm.nd.array(np.array(end).astype("int64"), dev)
    strides_nd = tvm.nd.array(np.array(strides).astype("int64"), dev)

    # Reference implementation
    out_npy = tvm.topi.testing.strided_slice_python(x_np, begin, end, strides)
    out_nd = vm["main"](data_nd, begin_nd, end_nd, strides_nd)
    tvm.testing.assert_allclose(out_nd.numpy(), out_npy)


@pytest.mark.parametrize(
    "begin, end, strides",
    [
        ([0, 2, 4, 4], [5, 5, 7, 8], [1, 1, 2, 3]),
        ([0, 2, 4, 4], [5, 5, 11, 10], [1, 1, 1, 1]),
        ([0, 2, 10, 14], [0, 5, 1, 1], [1, 1, -1, -2]),
    ],
)
def test_dynamic_strided_slice_symbolic(begin, end, strides):
    # fmt: off
    @tvm.script.ir_module
    class DynamicStridedSlice:
        @R.function
        def main(x: R.Tensor(("m", "n", 10, 10), "float32"), begin: R.Tensor((4,),"int64"), end: R.Tensor((4,),"int64"), strides: R.Tensor((4,),"int64")) -> R.Tensor("float32", ndim=4):
            m = T.int64()
            n = T.int64()
            gv: R.Tensor("float32", ndim=4) = R.dynamic_strided_slice(x, begin, end, strides)
            return gv
    # fmt: on
    vm = build(DynamicStridedSlice)

    x_np = np.random.rand(8, 9, 10, 10).astype(np.float32)
    data_nd = tvm.nd.array(x_np, dev)
    begin_nd = tvm.nd.array(np.array(begin).astype("int64"), dev)
    end_nd = tvm.nd.array(np.array(end).astype("int64"), dev)
    strides_nd = tvm.nd.array(np.array(strides).astype("int64"), dev)

    # Reference implementation
    out_npy = tvm.topi.testing.strided_slice_python(x_np, begin, end, strides)
    out_nd = vm["main"](data_nd, begin_nd, end_nd, strides_nd)
    tvm.testing.assert_allclose(out_nd.numpy(), out_npy)


if __name__ == "__main__":
    tvm.testing.main()
