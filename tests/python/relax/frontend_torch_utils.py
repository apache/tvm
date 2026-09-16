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
"""Small shared helpers for the two PyTorch importer test suites."""

import math

import numpy as np
import pytest
import torch

import tvm
from tvm import relax
from tvm.testing import env


class UnaryModule(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        return self.op(x)


def make_expected(input_info, compute, *, exported=False):
    """Build an explicit expected expression without repeating inferred tensor types."""
    inputs = [
        relax.Var(f"inp_{i}", relax.TensorType(shape, dtype))
        for i, (shape, dtype) in enumerate(input_info)
    ]
    builder = relax.BlockBuilder()
    with builder.function("main", inputs):
        with builder.dataflow():
            result = compute(*inputs)
            if exported:
                result = relax.Tuple(list(result) if isinstance(result, tuple | list) else [result])
            elif isinstance(result, tuple | list):
                result = relax.Tuple([builder.emit(value) for value in result])
            elif not isinstance(result, relax.Var):
                result = builder.emit(result)
            output = builder.emit_output(result)
        builder.emit_func_output(output)
    return builder.get()


def verify_numerically(mod, model, example_args, *, input_sets=None, rtol=1e-7, atol=1e-7):
    """Compile once, checking all output values, shapes and dtypes on every run."""
    if not env.has_llvm():
        pytest.skip("need llvm")
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm.cpu())
    for args in (example_args,) if input_sets is None else input_sets:
        actual = vm["main"](*[tvm.runtime.tensor(arg.detach().numpy()) for arg in args])
        actual = [actual] if isinstance(actual, tvm.runtime.Tensor) else list(actual)
        with torch.no_grad():
            expected = torch.utils._pytree.tree_leaves(model(*(arg.clone() for arg in args)))
        assert len(actual) == len(expected)
        for result, reference in zip(actual, expected):
            result, reference = result.numpy(), reference.numpy()
            assert result.shape == reference.shape
            assert result.dtype == reference.dtype
            np.testing.assert_allclose(result, reference, rtol=rtol, atol=atol)
    return vm


def constants(mod):
    result = []
    relax.analysis.post_order_visit(
        mod["main"].body,
        lambda expr: result.append(expr.data.numpy()) if isinstance(expr, relax.Constant) else None,
    )
    return result


def activation_cases(*, exported=False):
    """One explicit expression per activation; FX also exercises module dispatch."""
    op, nn, functional = relax.op, torch.nn, torch.nn.functional
    cases = []

    def add(name, torch_op, expected, module=None):
        cases.append(pytest.param(torch_op, expected, id=name))
        if module is not None and not exported:
            cases.append(pytest.param(module, expected, id=name + "-module"))

    def c(value):
        return relax.const(value, "float32")

    def hard_sigmoid(x):
        return op.divide(op.clip(op.add(x, c(3)), 0, 6), c(6))

    add(
        "celu",
        lambda x: functional.celu(x, alpha=2),
        lambda x: op.add(
            op.multiply(c(2), op.minimum(c(0), op.subtract(op.exp(op.divide(x, c(2))), c(1)))),
            op.nn.relu(x),
        ),
        nn.CELU(alpha=2),
    )
    add(
        "elu",
        lambda x: functional.elu(x, alpha=2),
        lambda x: op.add(
            op.multiply(c(-2), op.nn.relu(op.subtract(c(1), op.exp(x)))), op.nn.relu(x)
        ),
        nn.ELU(alpha=2),
    )
    add("gelu", functional.gelu, op.nn.gelu, nn.GELU())
    add("hardsigmoid", functional.hardsigmoid, hard_sigmoid, nn.Hardsigmoid())
    add(
        "hardswish", functional.hardswish, lambda x: op.multiply(x, hard_sigmoid(x)), nn.Hardswish()
    )
    add("hardtanh", functional.hardtanh, lambda x: op.clip(x, -1.0, 1.0), nn.Hardtanh())
    add("dropout", lambda x: torch.dropout(x, 0.5, train=False), lambda x: x, nn.Dropout().eval())
    add("log2", torch.log2, lambda x: op.divide(op.log(x), c(math.log(2))))
    add("log10", torch.log10, lambda x: op.divide(op.log(x), c(math.log(10))))
    add("log1p", torch.log1p, lambda x: op.log(op.add(x, c(1))))
    add("reciprocal", torch.reciprocal, lambda x: op.divide(c(1), x))
    add(
        "relu6",
        functional.relu6,
        (lambda x: op.clip(x, 0, 6)) if exported else op.nn.relu6,
        nn.ReLU6(),
    )
    add("selu", functional.selu, op.nn.selu, nn.SELU())
    add("silu", functional.silu, op.nn.silu, nn.SiLU())
    if exported:
        add("isfinite", torch.isfinite, op.isfinite)
        add("max", torch.max, op.max)
        add("min", torch.min, op.min)
    else:
        add("logical_not", torch.logical_not, lambda x: op.logical_not(op.astype(x, "bool")))
        add(
            "log_softmax",
            lambda x: functional.log_softmax(x, dim=1),
            lambda x: op.nn.log_softmax(x, axis=1),
            nn.LogSoftmax(dim=1),
        )
        add("relu", functional.relu, op.nn.relu, nn.ReLU())
        add("sigmoid", torch.sigmoid, op.sigmoid, nn.Sigmoid())
        add(
            "softmax",
            lambda x: functional.softmax(x, dim=1),
            lambda x: op.nn.softmax(x, axis=1),
            nn.Softmax(dim=1),
        )
        add("tanh", torch.tanh, op.tanh, nn.Tanh())
        add("tril", lambda x: torch.tril(x, 1), lambda x: op.tril(x, 1))
        add("tril-inplace", lambda x: x.tril_(1), lambda x: op.tril(x, 1))
        add("triu", lambda x: torch.triu(x, 1), lambda x: op.triu(x, 1))
        add("triu-inplace", lambda x: x.triu_(1), lambda x: op.triu(x, 1))
        add("trunc", torch.trunc, op.trunc)
    return cases


class PoolDivisorModel(torch.nn.Module):
    """Cover padded partial windows, signed divisors, and an unbatched input."""

    def __init__(self, ndim):
        super().__init__()
        self.pool = getattr(torch.nn, f"AvgPool{ndim}d")(
            3, stride=2, padding=1, ceil_mode=True, divisor_override=-7
        )
        self.functional = getattr(torch.nn.functional, f"avg_pool{ndim}d")

    def forward(self, x):
        return (
            self.pool(x),
            self.functional(x, 3, 2, 1, True, False, 5),
            self.functional(x[0], 2, divisor_override=3),
        )


class AntialiasedResizeModel(torch.nn.Module):
    def forward(self, x):
        resize = torch.nn.functional.interpolate
        return (
            resize(x, (3, 4), mode="bilinear", antialias=True),
            resize(x, (3, 4), mode="bilinear", align_corners=True, antialias=True),
            resize(x, (1, 1), mode="bilinear", align_corners=True, antialias=True),
            resize(x, (3, 4), mode="bicubic", antialias=True),
            resize(x, (10, 11), mode="bicubic", align_corners=True, antialias=True),
            resize(x, scale_factor=(0.6, 1.4), mode="bilinear", antialias=True),
            resize(
                x,
                scale_factor=(0.6, 1.4),
                recompute_scale_factor=True,
                mode="bilinear",
                antialias=True,
            ),
        )


class ExponentialModel(torch.nn.Module):
    def forward(self, x):
        first = x.exponential_().clone()
        second = x.exponential_(2.0)
        return first, second, x


def verify_exponential(mod, dtype):
    """Check distribution and effect ordering without demanding identical RNG streams."""
    if not env.has_llvm():
        pytest.skip("need llvm")
    if tvm.get_global_func("tvm.contrib.random.exponential", allow_missing=True) is None:
        pytest.skip("need USE_RANDOM")
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm.cpu())
    data = tvm.runtime.tensor(np.zeros(32768, dtype=dtype))
    previous = None
    for _ in range(2):
        first, second, alias = [value.numpy() for value in vm["main"](data)]
        np.testing.assert_array_equal(second, alias)
        assert not np.array_equal(first, second)
        if previous is not None:
            assert not np.array_equal(first, previous)
        previous = first.copy()
        for result, rate in ((first, 1.0), (second, 2.0)):
            assert result.dtype == np.dtype(dtype)
            assert result.shape == (32768,)
            assert np.isfinite(result).all() and (result >= 0).all()
            reference = torch.empty(32768, dtype=getattr(torch, dtype)).exponential_(rate).numpy()
            # Generous statistical bounds avoid dependence on either implementation's seed.
            assert abs(result.mean() * rate - 1) < 0.05
            assert abs(result.var() * rate**2 - 1) < 0.12
            assert abs(np.mean(result <= 1 / rate) - (1 - math.exp(-1))) < 0.02
            np.testing.assert_allclose(
                np.quantile(result, [0.25, 0.5, 0.75]),
                np.quantile(reference, [0.25, 0.5, 0.75]),
                rtol=0.12,
            )
