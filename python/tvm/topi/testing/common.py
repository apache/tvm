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
"""Common utility for topi test"""

import numpy as np
import scipy.signal

import tvm
from tvm import topi
from tvm.testing import assert_allclose

_injective_schedule = {
    "generic": topi.generic.schedule_injective,
    "cpu": topi.x86.schedule_injective,
    "arm_cpu": topi.arm_cpu.schedule_injective,
    "gpu": topi.cuda.schedule_injective,
    "hls": topi.hls.schedule_injective,
}

_reduce_schedule = {
    "generic": topi.generic.schedule_reduce,
    "cpu": topi.x86.schedule_reduce,
    "gpu": topi.cuda.schedule_reduce,
    "hls": topi.cuda.schedule_reduce,
}


def dispatch(target, dispatch_map):
    if isinstance(target, str):
        target = tvm.target.Target(target)
    assert isinstance(target, tvm.target.Target)
    for key in target.keys:
        if key in dispatch_map:
            return dispatch_map[key]
    return dispatch_map["generic"]


def get_injective_schedule(target):
    return dispatch(target, _injective_schedule)


def get_reduce_schedule(target):
    return dispatch(target, _reduce_schedule)


get_broadcast_schedule = get_injective_schedule
get_elemwise_schedule = get_injective_schedule

_conv2d_nchw_implement = {
    "generic": (topi.nn.conv2d_nchw, topi.generic.schedule_conv2d_nchw),
    "cpu": (topi.x86.conv2d_nchw, topi.x86.schedule_conv2d_nchw),
    "arm_cpu": (
        topi.arm_cpu.conv2d_nchw_spatial_pack,
        topi.arm_cpu.schedule_conv2d_nchw_spatial_pack,
    ),
    "gpu": (topi.cuda.conv2d_nchw, topi.cuda.schedule_conv2d_nchw),
    "mali": (topi.mali.conv2d_nchw_spatial_pack, topi.mali.schedule_conv2d_nchw_spatial_pack),
    "bifrost": (
        topi.bifrost.conv2d_nchw_spatial_pack,
        topi.bifrost.schedule_conv2d_nchw_spatial_pack,
    ),
    "intel_graphics": (topi.intel_graphics.conv2d_nchw, topi.intel_graphics.schedule_conv2d_nchw),
    "hls": (topi.nn.conv2d_nchw, topi.hls.schedule_conv2d_nchw),
}


def get_conv2d_nchw_implement(target):
    return dispatch(target, _conv2d_nchw_implement)


def compare_numpy_tvm(inputs, output, target, device, compute, schedule):
    """Compare a numpy inputs and output of a function to the results of the TVM version.

    Parameters
    ----------
    inputs : Sequence[numpy.nd.array]
        List of input numpy arrays to pass to the function.
    output : numpy.nd.array
        Verified correct function output.
    target : tvm.target.Target
        Target to run on.
    device : tvm.runtime.Device
        Context to run on.
    compute : callable
        Topi compute function to test against.
    schedule : callable
        Topi scheduling function to test against.
    """
    te_inputs = [tvm.te.placeholder(shape=i.shape, dtype=str(i.dtype)) for i in inputs]
    te_out = tvm.nd.array(np.zeros(output.shape).astype(output.dtype), device=device)
    with tvm.target.Target(target):
        out = compute(*te_inputs)
        s = schedule([out])
        func = tvm.build(s, te_inputs + [out])
        arys = [tvm.nd.array(x, device=device) for x in inputs]
        func(*(arys + [te_out]))
        assert_allclose(te_out.numpy(), output, atol=1e-4, rtol=1e-4)


def _convolve2d(data, weights):
    """2d convolution operator in HW layout.

    This is intended to be used as a replacement for
    scipy.signals.convolve2d, with wider support for different dtypes.
    scipy.signal.convolve2d does not support all TVM-supported
    dtypes (e.g. float16).  Where possible, this function uses
    scipy.signal.convolve2d to take advantage of compiled scipy
    routines, falling back to an explicit loop only where needed.

    Parameters
    ----------
    data : numpy.ndarray
        2-D with shape [in_height, in_width]

    weights : numpy.ndarray
        2-D with shape [filter_height, filter_width].

    Returns
    -------
    b_np : np.ndarray
        2-D with shape [out_height, out_width]

        Return value and layout conventions are matched to
        ``scipy.signal.convolve2d(data, weights, mode="valid")``
    """

    try:
        return scipy.signal.convolve2d(data, weights, mode="valid")
    except ValueError:
        pass

    weights = np.rot90(weights, k=2)

    assert len(data.shape) == len(weights.shape) == 2

    dtype = data.dtype
    kernel_h, kernel_w = weights.shape

    output_shape = [a_dim - w_dim + 1 for a_dim, w_dim in zip(data.shape, weights.shape)]
    output = np.zeros(output_shape, dtype=dtype)

    for y in range(output_shape[0]):
        for x in range(output_shape[1]):
            output[y][x] = np.sum(data[y : y + kernel_h, x : x + kernel_w] * weights)

    return output
