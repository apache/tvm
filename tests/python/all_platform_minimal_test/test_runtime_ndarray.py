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
"""Basic runtime enablement test."""

import numpy as np
import tvm
import tvm.testing
from tvm import te


@tvm.testing.uses_gpu
def test_nd_create():
    """Test creating an array in TVM."""
    for _, dev in tvm.testing.enabled_targets():
        for dtype in ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32"]:
            main_data = np.random.randint(0, 10, size=(3, 4))
            np_array = np.array(main_data, dtype=dtype)
            tvm_array = tvm.nd.array(np_array, device=dev)
            tvm_array_device = tvm_array.copyto(dev)
            assert tvm_array.dtype == np_array.dtype
            assert tvm_array.shape == np_array.shape
            assert isinstance(tvm_array, tvm.nd.NDArray)
            np.testing.assert_equal(np_array, tvm_array.numpy())
            np.testing.assert_equal(np_array, tvm_array_device.numpy())
        # no need here, just to test usablity
        dev.sync()


def test_fp16_conversion():
    """Test converting to and from FP16."""
    arr_size = 100

    def get_compute_func(placeholder_a, dst):
        return lambda i: placeholder_a[i].astype(dst)

    for (src, dst) in [("float32", "float16"), ("float16", "float32")]:
        placeholder_a = te.placeholder((arr_size,), dtype=src)
        result_b = te.compute((arr_size,), get_compute_func(placeholder_a, dst))

        schedule = te.create_schedule([result_b.op])
        func = tvm.build(schedule, [placeholder_a, result_b], "llvm")

        x_tvm = tvm.nd.array(100 * np.random.randn(arr_size).astype(src) - 50)
        y_tvm = tvm.nd.array(100 * np.random.randn(arr_size).astype(dst) - 50)

        func(x_tvm, y_tvm)

        expected = x_tvm.numpy().astype(dst)
        real = y_tvm.numpy()

        tvm.testing.assert_allclose(expected, real)


def test_dtype():
    dtype = tvm.DataType("handle")
    assert dtype.type_code == tvm.DataTypeCode.HANDLE


if __name__ == "__main__":
    test_nd_create()
    test_fp16_conversion()
    test_dtype()
