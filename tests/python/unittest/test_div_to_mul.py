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
from tvm import relay
import pytest
import numpy as np


@pytest.mark.parametrize("dtype, rtol", [("float16", 1e-3), ("float32", 1e-7), ("float64", 1e-12)])
def test_div_to_mul(dtype, rtol):
    x = relay.var("x", relay.TensorType((), dtype))
    y = relay.Constant(tvm.nd.array(np.array([1.5]).astype(dtype)))
    z = x / y
    mod = tvm.IRModule.from_expr(z)
    transformed = relay.transform.DivToMul()(mod)
    assert transformed["main"].body.op.name == "multiply"
    np.testing.assert_allclose(transformed["main"].body.args[1].data.numpy()[0], 1 / 1.5, rtol=rtol)
