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
@pytest.mark.parametrize(
    "shape,axis",
    [([4, 16], (1,)), ([4, 16, 16], (1, 2)), ([("a", 4), ("b", 16)], (1,)), ([2, 8192], (1,))],
)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_rms_norm(target, dev, shape, axis, dtype, episilon=1e-5, rtol=5e-3, atol=1e-4):
    shape_te = [te.var(v[0]) if isinstance(v, tuple) else v for v in shape]
    scale_shape_te = [shape_te[dim] for dim in axis]
    data = te.placeholder(shape_te, dtype=dtype, name="data")
    weight = te.placeholder(scale_shape_te, dtype=dtype, name="weight")
    B = topi.nn.rms_norm(data, weight, axis, episilon)

    shape_np = [v[1] if isinstance(v, tuple) else v for v in shape]
    scale_shape_np = [shape_np[dim] for dim in axis]
    data_np = np.random.uniform(size=shape_np).astype(dtype)
    weight_np = np.random.uniform(size=scale_shape_np).astype(dtype)
    b_np = tvm.topi.testing.rms_norm_python(data_np, weight_np, axis, episilon)

    with tvm.target.Target(target):
        s_func = tvm.topi.testing.dispatch(target, _rms_norm_schedule)
        s = s_func([B])
    data_tvm = tvm.nd.array(data_np, dev)
    weight_tvm = tvm.nd.array(weight_np, dev)
    b_tvm = tvm.nd.array(np.zeros(shape_np, dtype=dtype), dev)
    f = tvm.build(s, [data, weight, B], target)
    f(data_tvm, weight_tvm, b_tvm)
    tvm.testing.assert_allclose(b_tvm.numpy(), b_np, rtol=rtol, atol=atol)


if __name__ == "__main__":
    tvm.testing.main()
