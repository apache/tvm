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
"""SHL integration conv2d tests."""

import pytest
import numpy as np

import tvm
from tvm import relay
import socket
import time
import subprocess

from infrastructure import skip_runtime_test, verify
from infrastructure import Device, build_and_run


def net_is_used(port, ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, port))
        s.shutdown(2)
        return True
    except:
        return False


PORT = 9090


@pytest.fixture(scope="module")
def start_qemu(
    cpu="c906fdv",
    sysroot="/opt/riscv/riscv64-unknown-linux-gnu/sysroot/",
    cmd="./build-c906/tvm_rpc server",
    ip="127.0.0.1",
    port=PORT,
):
    if net_is_used(port, ip):
        raise Exception(f"Port {port} is busy, QEMU startup faild")

    start_cmd = f"exec qemu-riscv64 -cpu {cpu} -L {sysroot} {cmd} --host={ip} --port={port}"

    process = subprocess.Popen(start_cmd, shell=True)
    # Wait for QEMU to start
    count = 0
    while not net_is_used(port, ip):
        time.sleep(1)
        count += 1
        if count > 3:
            raise Exception(f"Startup timeout, QEMU startup faild")

    yield
    process.kill()


np.random.seed(0)

dtype = tvm.testing.parameter("float32")
trials = tvm.testing.parameter(
    # Normal convolution
    (2, 2, (1, 1), (1, 1), (1, 1), 4, (14, 10, 10), False, False),
    (2, 1, (2, 2), (1, 1), (1, 1), 7, (16, 12, 15), False, False),
    (2, 2, (1, 1), (1, 1), (1, 1), 4, (14, 10, 10), True, False),
    (3, 3, (1, 1), (1, 1), (1, 1), 16, (16, 12, 15), False, False),
    (5, 5, (1, 1), (2, 2), (1, 1), 4, (14, 10, 10), False, False),
    (1, 3, (1, 1), (1, 1), (1, 1), 7, (20, 20, 20), False, False),
    (2, 2, (2, 2), (1, 1), (1, 1), 4, (20, 20, 20), True, False),
    (5, 5, (1, 1), (2, 2), (1, 1), 4, (14, 10, 10), False, False),
    (3, 3, (2, 1), (1, 1), (1, 1), 7, (20, 20, 20), False, False),
    (3, 3, (1, 1), (2, 2), (1, 1), 16, (14, 10, 10), True, False),
    # Depth-wise convolution
    (3, 3, (1, 1), (1, 1), (1, 1), 20, (20, 20, 20), False, True),
    (5, 5, (2, 2), (1, 1), (1, 1), 20, (20, 20, 20), True, True),
    (3, 3, (2, 2), (2, 2), (1, 1), 14, (14, 10, 10), False, True),
    (5, 5, (0, 0), (1, 1), (1, 1), 20, (20, 20, 20), False, True),
    (3, 3, (1, 1), (2, 2), (1, 1), 14, (14, 10, 10), True, True),
)


def _get_model(
    shape,
    kernel_h,
    kernel_w,
    padding,
    strides,
    dilation,
    groups,
    dtype,
    channels,
    var_names,
    has_bias=False,
):
    """Return a model and any parameters it may have"""
    a = relay.var(next(var_names), shape=shape, dtype=dtype)
    if len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    shape = (shape[0], shape[1], shape[2] + padding[0] * 2, shape[3] + padding[1] * 2)

    weight_shape = (channels, shape[1] // groups, kernel_h, kernel_w)
    w = tvm.nd.array(np.random.uniform(-128, 127, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.nn.conv2d(
        a,
        weights,
        kernel_size=(kernel_h, kernel_w),
        data_layout="NCHW",
        dilation=dilation,
        strides=strides,
        padding=padding,
        groups=groups,
        channels=channels,
        out_dtype=dtype,
    )
    params = {"w": w}
    if has_bias:
        bias_shape = weight_shape[0]
        b = tvm.nd.array(np.random.uniform(-128, 127, bias_shape).astype(dtype))
        biasc = relay.const(b, dtype)
        out = relay.nn.bias_add(out, biasc, axis=1)
        params["b"] = b
    return out, params


@tvm.testing.requires_csinn2
def test_conv2d(start_qemu, dtype, trials):
    Device.load("test_config.json")
    Device.port = PORT

    if skip_runtime_test():
        return

    device = Device()

    kernel_h, kernel_w, pad, stride, dilation, out_channels, shape, has_bias, is_depthwise = trials

    shape = (1, *shape)
    if is_depthwise:
        groups = shape[1]
    else:
        groups = 1
    outputs = []
    inputs = {
        "a": tvm.nd.array(np.random.uniform(-128, 127, shape).astype(dtype)),
    }

    func, params = _get_model(
        shape,
        kernel_h,
        kernel_w,
        pad,
        stride,
        dilation,
        groups,
        dtype,
        out_channels,
        iter(inputs),
        has_bias,
    )
    for csinn in [False, True]:
        outputs.append(build_and_run(func, inputs, 1, params, device, enable_csinn=csinn)[0])

    config = {
        "shape": shape,
        "groups": groups,
        "kernel size": (kernel_h, kernel_w),
        "padding": pad,
        "stride": stride,
        "dilation": dilation,
        "out channels": out_channels,
        "has bias": has_bias,
    }
    verify(outputs, atol=0.002, rtol=0.01, config=config)


if __name__ == "__main__":
    tvm.testing.main()
