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
# pylint: disable=invalid-name

"""Compute and schedule for quantized add, multiply, subtract op

Please note the following assumptions made by the implementation:

1) The inputs will be multiple of crouton layout except for the axis that needs broadcasting."""

from tvm import te
from tvm import tir
from ..utils import get_layout_transform_fn


def qsqrt(x, in_scale, in_zero, out_scale, out_zero, dtype):
    """Computes square root of value

    x: tensor
    -----
    return: tensor
    """

    if dtype in ("uint8"):
        func_str = f"lambda x: math.sqrt({in_scale}*x-{in_zero})/{out_scale}+{out_zero}"
        return te.compute(
            (x.shape),
            lambda *i: (tir.sqrt(in_scale * (x[i] - in_zero)) / out_scale + out_zero).astype(dtype),
            name="qsqrt",
            tag=func_str,
        )
    else:
        raise ValueError(f"dtype {dtype} not supported for qsqrt!")


def qrsqrt(x, in_scale, in_zero, out_scale, out_zero, dtype):
    """Computes square root of value

    x: tensor
    -----
    return: tensor
    """

    if dtype in ("uint8"):
        func_str = f"lambda x: (1/math.sqrt({in_scale}*x-{in_zero}))/{out_scale}+{out_zero}"
        return te.compute(
            (x.shape),
            lambda *i: ((1 / tir.sqrt(in_scale * (x[i] - in_zero))) / out_scale + out_zero).astype(
                dtype
            ),
            name="qrsqrt",
            tag=func_str,
        )
    else:
        raise ValueError(f"dtype {dtype} not supported for qsrqrt!")


def qexp(x, in_scale, in_zero, out_scale, out_zero, dtype):
    """Computes square root of value

    x: tensor
    -----
    return: tensor
    """

    if dtype in ("uint8"):
        func_str = f"lambda x: math.exp({in_scale}*x-{in_zero})/{out_scale}+{out_zero}"
        return te.compute(
            (x.shape),
            lambda *i: (tir.exp(in_scale * (x[i] - in_zero)) / out_scale + out_zero).astype(dtype),
            name="qexp",
            tag=func_str,
        )
    else:
        raise ValueError(f"dtype {dtype} not supported for qexp!")


def qerf(x, in_scale, in_zero, out_scale, out_zero, dtype):
    """Computes square root of value

    x: tensor
    -----
    return: tensor
    """

    if dtype in ("uint8"):
        func_str = f"lambda x: math.erf({in_scale}*x-{in_zero})/{out_scale}+{out_zero}"
        return te.compute(
            (x.shape),
            lambda *i: (tir.erf(in_scale * (x[i] - in_zero)) / out_scale + out_zero).astype(dtype),
            name="qerf",
            tag=func_str,
        )
    else:
        raise ValueError(f"dtype {dtype} not supported for qerf!")


def qsigmoid(x, in_scale, in_zero, out_scale, out_zero, dtype):
    """Computes square root of value

    x: tensor
    -----
    return: tensor
    """

    if dtype in ("uint8"):
        sigmoid_str = lambda x: f"(1 / (1 + math.exp(-{x})))"
        deq_str = f"({in_scale}*x-{in_zero})"
        func_str = f"lambda x: {sigmoid_str(deq_str)}/{out_scale}+{out_zero}"
        return te.compute(
            (x.shape),
            lambda *i: (tir.sigmoid(in_scale * (x[i] - in_zero)) / out_scale + out_zero).astype(
                dtype
            ),
            name="qsigmoid",
            tag=func_str,
        )
    else:
        raise ValueError(f"dtype {dtype} not supported for qsigmoid!")


def qtanh(x, in_scale, in_zero, out_scale, out_zero, dtype):
    """Computes square root of value

    x: tensor
    -----
    return: tensor
    """

    if dtype in ("uint8"):
        func_str = f"lambda x: np.tanh({in_scale}*x-{in_zero})/{out_scale}+{out_zero}"
        return te.compute(
            (x.shape),
            lambda *i: (tir.tanh(in_scale * (x[i] - in_zero)) / out_scale + out_zero).astype(dtype),
            name="qtanh",
            tag=func_str,
        )
    else:
        raise ValueError(f"dtype {dtype} not supported for qtanh!")


def qlog(x, in_scale, in_zero, out_scale, out_zero, dtype):
    """Computes square root of value

    x: tensor
    -----
    return: tensor
    """

    if dtype in ("uint8"):
        func_str = f"lambda x: np.log({in_scale}*x-{in_zero})/{out_scale}+{out_zero}"
        return te.compute(
            (x.shape),
            lambda *i: (tir.log(in_scale * (x[i] - in_zero)) / out_scale + out_zero).astype(dtype),
            name="qlog",
            tag=func_str,
        )
    else:
        raise ValueError(f"dtype {dtype} not supported for qtanh!")


def qabs(x, in_scale, in_zero, out_scale, out_zero, dtype):
    """Computes square root of value

    x: tensor
    -----
    return: tensor
    """

    if dtype in ("uint8"):
        func_str = f"lambda x: np.abs({in_scale}*x-{in_zero})/{out_scale}+{out_zero}"
        return te.compute(
            (x.shape),
            lambda *i: (tir.abs(in_scale * (x[i] - in_zero)) / out_scale + out_zero).astype(dtype),
            name="qabs",
            tag=func_str,
        )
    else:
        raise ValueError(f"dtype {dtype} not supported for qabs!")
