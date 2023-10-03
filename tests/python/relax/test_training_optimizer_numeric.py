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
"""Numeric tests for relax optimizer APIs."""
from typing import Callable, List

import numpy as np
import tvm
import tvm.testing
from tvm import relax
from tvm import IRModule
from tvm.relax.training.optimizer import Adam, SGD, MomentumSGD
from tvm.script.parser import relax as R
from tvm.runtime.relax_vm import VirtualMachine
from tvm.testing import assert_allclose


def _legalize_and_build(mod: IRModule, target, dev):
    ex = relax.build(mod, target)
    vm = VirtualMachine(ex, dev)
    return vm


def _numpy_to_tvm(data):
    if isinstance(data, (list, tuple)):
        return [_numpy_to_tvm(_data) for _data in data]
    return tvm.nd.array(data)


def _tvm_to_numpy(data):
    if isinstance(data, (list, tuple, tvm.ir.Array)):
        return [_tvm_to_numpy(_data) for _data in data]
    return data.numpy()


def _assert_allclose_nested(data1, data2):
    if isinstance(data1, (list, tuple)):
        assert isinstance(data2, (list, tuple))
        assert len(data1) == len(data2)
        for x, y in zip(data1, data2):
            _assert_allclose_nested(x, y)
    else:
        assert_allclose(data1, data2)


def _assert_run_result_same(tvm_func: Callable, np_func: Callable, np_inputs: List):
    result = _tvm_to_numpy(tvm_func(*[_numpy_to_tvm(i) for i in np_inputs]))
    expected = np_func(*np_inputs)
    _assert_allclose_nested(result, expected)


@tvm.testing.parametrize_targets("llvm")
def _test_optimizer(target, dev, np_func, opt_type, *args, **kwargs):
    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    opt = opt_type(*args, **kwargs).init([x, y])
    mod = IRModule.from_expr(opt.get_function().with_attr("global_symbol", "main"))
    tvm_func = _legalize_and_build(mod, target, dev)["main"]

    param_arr = [np.random.rand(3, 3).astype(np.float32), np.random.rand(3).astype(np.float32)]
    grad_arr = [np.random.rand(3, 3).astype(np.float32), np.random.rand(3).astype(np.float32)]
    state_arr = _tvm_to_numpy(opt.state)

    _assert_run_result_same(tvm_func, np_func, [param_arr, grad_arr, state_arr])


lr, weight_decay = tvm.testing.parameters(
    (0.01, 0),
    (0.01, 0.02),
)


@tvm.testing.parametrize_targets("llvm")
def test_sgd(target, dev, lr, weight_decay):
    def np_func(param_tuple, grad_tuple, state_tuple):
        num_steps = state_tuple[0]
        param_tuple_new, state_tuple_new = [], []
        state_tuple_new.append(num_steps + 1)
        for i in range(len(param_tuple)):
            param = param_tuple[i]
            grad = grad_tuple[i]
            param_tuple_new.append(param - lr * (grad + weight_decay * param))
        return param_tuple_new, state_tuple_new

    _test_optimizer(target, dev, np_func, SGD, lr, weight_decay)


lr, momentum, dampening, weight_decay, nesterov = tvm.testing.parameters(
    (0.01, 0.9, 0, 0, False),
    (0.01, 0.9, 0.85, 0.02, False),
    (0.01, 0.9, 0.85, 0.02, True),
)


@tvm.testing.parametrize_targets("llvm")
def test_momentum_sgd(target, dev, lr, momentum, dampening, weight_decay, nesterov):
    def np_func(param_tuple, grad_tuple, state_tuple):
        num_steps = state_tuple[0]
        param_tuple_new, state_tuple_new = [], []
        state_tuple_new.append(num_steps + 1)

        for i in range(len(param_tuple)):
            param = param_tuple[i]
            grad = grad_tuple[i]
            velocity = state_tuple[i + 1]
            grad = param * weight_decay + grad
            velocity = momentum * velocity + grad * (1 - dampening)
            if nesterov:
                param = param - (grad + momentum * velocity) * lr
            else:
                param = param - velocity * lr
            param_tuple_new.append(param)
            state_tuple_new.append(velocity)

        return param_tuple_new, state_tuple_new

    _test_optimizer(
        target, dev, np_func, MomentumSGD, lr, momentum, dampening, weight_decay, nesterov
    )


lr, betas, eps, weight_decay = tvm.testing.parameters(
    (0.01, (0.9, 0.999), 1e-08, 0),
    (0.01, (0.8, 0.85), 1e-07, 0.1),
)


@tvm.testing.parametrize_targets("llvm")
def test_adam(target, dev, lr, betas, eps, weight_decay):
    def np_func(param_tuple, grad_tuple, state_tuple):
        num_steps = state_tuple[0]
        num_steps_new = num_steps + 1

        param_tuple_new = []
        state_tuple_new = [None] * len(state_tuple)  # type: ignore
        state_tuple_new[0] = num_steps_new
        state_tuple_new[1] = state_tuple[1] * betas[0]
        state_tuple_new[2] = state_tuple[2] * betas[1]

        for i in range(len(param_tuple)):
            param = param_tuple[i]
            grad = grad_tuple[i]
            m = state_tuple[i + 3]
            v = state_tuple[i + 3 + len(param_tuple)]
            grad = grad + weight_decay * param
            m = betas[0] * m + (1 - betas[0]) * grad
            v = betas[1] * v + (1 - betas[1]) * grad * grad
            m_hat = m / (1 - betas[0] ** num_steps_new)
            v_hat = v / (1 - betas[1] ** num_steps_new)
            param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
            param_tuple_new.append(param)
            state_tuple_new[i + 3] = m
            state_tuple_new[i + 3 + len(param_tuple)] = v

        return param_tuple_new, state_tuple_new

    _test_optimizer(target, dev, np_func, Adam, lr, betas, eps, weight_decay)


if __name__ == "__main__":
    tvm.testing.main()
