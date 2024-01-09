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
import tvm
from tvm import te
from tvm import topi
import tvm.testing
import tvm.topi.testing
import numpy as np


def test_dilate():
    target = "llvm"
    dev = tvm.cpu(0)

    def _test_dilate(input_size, strides, dilation_value=None):
        Input = te.placeholder((input_size))
        if dilation_value is None:
            Output = topi.nn.dilate(Input, strides)
        else:
            Output = topi.nn.dilate(Input, strides, dilation_value)
        schedule = te.create_schedule(Output.op)
        input_np = np.random.uniform(size=input_size).astype(Input.dtype)
        if dilation_value is None:
            output_np = tvm.topi.testing.dilate_python(input_np, strides)
        else:
            output_np = tvm.topi.testing.dilate_python(input_np, strides, dilation_value)
        input_tvm = tvm.nd.array(input_np, device=dev)
        output_size = topi.utils.get_const_tuple(Output.shape)
        output_tvm = tvm.nd.array(np.zeros(shape=output_size).astype(Output.dtype), device=dev)
        f = tvm.build(schedule, [Input, Output], target)
        f(input_tvm, output_tvm)
        tvm.testing.assert_allclose(output_tvm.numpy(), output_np, rtol=1e-5)

    _test_dilate((32,), (2,))
    _test_dilate((32, 32), (2, 2))
    _test_dilate((1, 3, 32, 32), (1, 1, 1, 1))
    _test_dilate((1, 3, 32, 32), (2, 2, 2, 2))
    _test_dilate((1, 32, 32, 3, 3), (1, 1, 1, 1, 1))
    _test_dilate((1, 32, 32, 3, 3), (2, 2, 2, 2, 2))
    _test_dilate((1, 32, 32, 32, 3, 3), (1, 1, 1, 2, 2, 2))
    _test_dilate((1, 32, 32, 32, 3, 3), (2, 2, 2, 1, 1, 1))
    _test_dilate((1, 32, 32, 32, 3, 3), (2, 2, 2, 1, 1, 1), 1.0)


if __name__ == "__main__":
    test_dilate()
