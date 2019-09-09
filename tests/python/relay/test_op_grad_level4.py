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
import pytest
from tvm import relay
from tvm.relay.testing import check_grad


def verify_sum_grad(d_shape, axis=None, keepdims=False, exclude=False):
    data = relay.var("data", relay.TensorType(d_shape, "float32"))
    fwd_func = relay.Function([data], relay.sum(data, axis=axis, keepdims=keepdims, exclude=exclude))
    check_grad(fwd_func)


def test_sum_grad():
    verify_sum_grad((4, 2))
    verify_sum_grad((4, 2), axis=-1, keepdims=True)
    verify_sum_grad((4, 2, 1), axis=(1, 2), exclude=True)


def test_max_grad():
    s = (10, 10)
    t = relay.TensorType(s)
    x = relay.var("x", t)
    axis = 0
    z = relay.max(x, axis)

    fwd_func = relay.Function([x], z)
    check_grad(fwd_func, scale=1e-3)


if __name__ == "__main__":
    pytest.main()
