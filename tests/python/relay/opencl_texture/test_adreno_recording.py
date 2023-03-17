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
import json
import os
import re
import sys
import time

import pytest

import tvm
import tvm.testing
from tvm import te
import numpy as np
from tvm import relay

from tvm.contrib import utils, ndk
from tvm.contrib.adreno_recording import adreno_recording_executor as graph_runtime

from utils.adreno_utils import get_cpu_reference, gpu_preprocess

dtype = tvm.testing.parameter("float32")


@tvm.testing.requires_adrenorecording
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_recording_simple(remote, target, dtype):
    input_shape, filter_shape = (1, 16, 24, 24), (32, 16, 3, 3)
    bias_shape = (1, 1, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[1, 1, 1, 1],
        strides=[1, 1],
        out_dtype=dtype,
    )

    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(conv)
    mod = relay.Function([A, B], D)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    if remote is None:
        target_host = "llvm"
    else:
        target_host = "llvm -mtriple=arm64-linux-android"

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target_host=target_host, target=target, params=params)

    if remote is None:
        ctx = tvm.opencl()
        m = graph_runtime.create(graph, lib, ctx)
    else:
        temp = utils.tempdir()
        dso_binary = "dev_lib_cl.so"
        dso_binary_path = temp.relpath(dso_binary)
        ctx = remote.cl(0)
        lib.export_library(dso_binary_path, ndk.create_shared)
        remote.upload(dso_binary_path)
        rlib = remote.load_module(dso_binary)
        m = graph_runtime.create(graph, rlib, ctx)
    m.set_input(**params)
    input = np.random.normal(size=input_shape).astype(dtype)
    m.set_input("data", input)

    m.capture_graph()
    m.run_recording()

    ref_outputs = get_cpu_reference(mod, params, input_shape, [input])
    for i, ref_output in enumerate(ref_outputs):
        tvm_output = m.get_output(i)
        output = tvm_output.asnumpy()

        np.testing.assert_allclose(output, ref_output, rtol=1e-1, atol=1e-1)
    return graph


if __name__ == "__main__":
    test_recording_simple()
