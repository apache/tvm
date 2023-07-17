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
"""Test code for rms_norm."""
import numpy as np
import pytest
import tvm
from tvm import te
from tvm import topi
from tvm.topi.utils import get_const_tuple
import tvm.topi.testing

import tvm.testing


_rms_norm_schedule = {
    "generic": topi.generic.schedule_injective,
}


# only test on llvm because schedule is missing
@tvm.testing.parametrize_targets("llvm")
@pytest.mark.parametrize("shape,axis", [([4, 16], (1,)), ([4, 16, 16], (1, 2))])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_rms_norm(target, dev, shape, axis, dtype, episilon=1e-5, rtol=5e-3, atol=1e-4):
    data = te.placeholder(shape, dtype=dtype, name="data")
    scale_shape = [shape[dim] for dim in axis]
    weight = te.placeholder(scale_shape, dtype=dtype, name="weight")
    bias = te.placeholder(scale_shape, dtype=dtype, name="weight")
    B = topi.nn.rms_norm(data, weight, bias, axis, episilon)

    data_np = np.random.uniform(size=shape).astype(dtype)
    weight_np = np.random.uniform(size=scale_shape).astype(dtype)
    bias_np = np.random.uniform(size=scale_shape).astype(dtype)
    b_np = tvm.topi.testing.rms_norm_python(data_np, weight_np, bias_np, axis, episilon)

    with tvm.target.Target(target):
        s_func = tvm.topi.testing.dispatch(target, _rms_norm_schedule)
        s = s_func([B])
    data_tvm = tvm.nd.array(data_np, dev)
    weight_tvm = tvm.nd.array(weight_np, dev)
    bias_tvm = tvm.nd.array(bias_np, dev)
    b_tvm = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), dev)
    f = tvm.build(s, [data, weight, bias, B], target)
    f(data_tvm, weight_tvm, bias_tvm, b_tvm)
    tvm.testing.assert_allclose(b_tvm.numpy(), b_np, rtol=rtol, atol=atol)


if __name__ == "__main__":
    tvm.testing.main()
