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
import numpy as np
from tvm import relay
from tvm.relay.testing import check_grad, _np_randn_from_type


def verify_reduction_grad(red_fn, d_shape, axis=None, keepdims=False, exclude=False):
    data = relay.var("data", relay.TensorType(d_shape, "float32"))
    fwd_func = relay.Function([data], red_fn(data, axis=axis, keepdims=keepdims, exclude=exclude))
    check_grad(fwd_func)


def test_reduction_grad():
    def _unbiased_variance(x, axis=None, keepdims=False, exclude=False):
        return relay.variance(x, axis=axis, keepdims=keepdims, exclude=exclude, unbiased=True)

    for op in (relay.sum, relay.variance, _unbiased_variance, relay.mean):
        verify_reduction_grad(op, (4, 2))
        verify_reduction_grad(op, (4, 2), axis=-1, keepdims=True)
        verify_reduction_grad(op, (4, 2, 1), axis=(1, 2), exclude=True)
        verify_reduction_grad(op, (4, 2, 1), axis=1)


def verify_max_grad(d_shape, axis=None, keepdims=False, exclude=False):
    data = relay.var("data", relay.TensorType(d_shape, "float32"))
    fwd_func = relay.Function(
        [data], relay.max(data, axis=axis, keepdims=keepdims, exclude=exclude)
    )
    check_grad(fwd_func, scale=1e-3)


def test_max_grad():
    verify_max_grad((10, 10), axis=None)
    verify_max_grad((10, 10), axis=-1)
    verify_max_grad((6, 3, 2), axis=(1, 2), keepdims=True)
    verify_max_grad((5, 4, 3), axis=(0, 2), exclude=True)


def test_where_grad():
    cond_type = relay.TensorType((2, 3, 4), "int32")
    lhs_type = relay.TensorType((1, 3, 4), "float32")
    rhs_type = relay.TensorType((2, 1, 4), "float32")
    inputs = [
        np.random.randint(2, size=cond_type.concrete_shape, dtype=cond_type.dtype),
        _np_randn_from_type(lhs_type, scale=1e-5),
        _np_randn_from_type(rhs_type, scale=1e-5),
    ]

    cond = relay.var("cond", type_annotation=cond_type)
    lhs = relay.var("lhs", type_annotation=lhs_type)
    rhs = relay.var("rhs", type_annotation=rhs_type)
    fwd_func = relay.Function([cond, lhs, rhs], relay.where(cond, lhs, rhs))
    check_grad(fwd_func, inputs=inputs, test_inputs=inputs[1:])


def test_less_equal_grad():
    x_type = relay.TensorType((2, 3, 4), "float32")
    y_type = relay.TensorType((3, 1), "float32")
    # We need to generate inputs far apart to get correct numerical gradients
    # (otherwise adding epsilon may change comparison result). The gradient
    # should always be zero for both inputs.
    inputs = [
        np.random.choice([-1, 1], size=x_type.concrete_shape).astype(x_type.dtype),
        np.random.choice([-2, 2], size=y_type.concrete_shape).astype(y_type.dtype),
    ]

    x = relay.var("x", type_annotation=x_type)
    y = relay.var("y", type_annotation=y_type)
    fwd_func = relay.Function([x, y], relay.less_equal(x, y))
    check_grad(fwd_func, inputs=inputs, test_inputs=inputs, eps=1e-6)


if __name__ == "__main__":
    pytest.main()
