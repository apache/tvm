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
"""Utils for adreno compute/schedules"""

import os
import tvm
import numpy as np
from tvm import relay
from tvm import autotvm
from tvm.relay import testing
from tvm.relay.transform import recast
from tvm.contrib import graph_runtime
import json


def get_cpu_reference(mod, params1, input_shape, inputs):
    mod_fp32 = recast(mod, "float32", "float32", ops=["nn.conv2d", "add", "nn.relu"])
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod_fp32, "llvm", params=params1)
    ctx = tvm.cpu()
    m = graph_runtime.create(graph, lib, ctx)
    if isinstance(input_shape, dict):
        for key in input_shape:
            m.set_input(key, inputs[-1])
    else:
        m.set_input("data", inputs[-1])
    m.set_input(**params)
    m.run()
    return [
        m.get_output(0).asnumpy(),
    ]


# build module run with opencl and cpu, compare results
def build_run_compare(
    tvm_mod,
    params1,
    input_shape,
    dtype="float32",
    target="llvm",
    static_mem_scopes=[],
    gpu_preprocess=None,
    stat_file=None,
):

    if "TVM_TRACKER_HOST" in os.environ and "TVM_TRACKER_PORT" in os.environ:
        rpc_tracker_host = os.environ["TVM_TRACKER_HOST"]
        rpc_tracker_port = os.environ["TVM_TRACKER_PORT"]
        run_on_host = 0
        target_host = "llvm -mtriple=arm64-linux-android"
        rpc_tracker_port = int(rpc_tracker_port)
    else:
        run_on_host = 1
        target_host = "llvm"

    if gpu_preprocess:
        tvm_mod_nchwc = gpu_preprocess(tvm_mod)
    else:
        tvm_mod_nchwc = tvm_mod

    if stat_file is not None:
        with autotvm.apply_history_best(stat_file):
            with tvm.transform.PassContext(opt_level=3):
                graph, lib, params = relay.build(
                    tvm_mod_nchwc, target_host=target_host, target=target, params=params1
                )
    else:
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build(
                tvm_mod_nchwc, target_host=target_host, target=target, params=params1
            )

    # verification that storage_scope has expected textures scopes
    graph_json = json.loads(graph)
    if "storage_scope" in graph_json["attrs"]:
        assert (
            len(static_mem_scopes) == len(graph_json["attrs"]["storage_scope"][1])
            or len(static_mem_scopes) == 0
        )
    else:
        assert len(static_mem_scopes) == 0

    for i in range(0, len(static_mem_scopes)):
        assert static_mem_scopes[i] == graph_json["attrs"]["storage_scope"][1][i]

    if run_on_host:
        ctx = tvm.opencl()
        m = graph_runtime.create(graph, lib, ctx)
    else:
        from tvm import rpc
        from tvm.contrib import utils, ndk

        rpc_key = "android"
        tracker = rpc.connect_tracker(rpc_tracker_host, rpc_tracker_port)
        remote = tracker.request(rpc_key, priority=0, session_timeout=600)
        temp = utils.tempdir()
        dso_binary = "dev_lib_cl.so"
        dso_binary_path = temp.relpath(dso_binary)
        ctx = remote.cl(0)
        lib.export_library(dso_binary_path, ndk.create_shared)
        remote.upload(dso_binary_path)
        rlib = remote.load_module(dso_binary)
        m = graph_runtime.create(graph, rlib, ctx)
    m.set_input(**params)
    inputs = []
    if isinstance(input_shape, dict):
        for key in input_shape:
            inputs.append(np.random.normal(size=input_shape[key]).astype(dtype))
            m.set_input(key, inputs[-1])
    else:
        inputs.append(np.random.normal(size=input_shape).astype(dtype))
        m.set_input("data", inputs[-1])
    m.run()

    ref_outputs = get_cpu_reference(tvm_mod, params1, input_shape, inputs)
    for i, ref_output in enumerate(ref_outputs):
        tvm_output = m.get_output(i)
        output = tvm_output.asnumpy()
        # for index, x in np.ndenumerate(ref_output):
        #     if abs(output[index] - x) > 0.01:
        #         print(index, output[index], x)

        np.testing.assert_allclose(output, ref_output, rtol=1e-1, atol=1e-1)
    return graph


def gpu_preprocess(tvm_mod):
    layout_config = relay.transform.LayoutConfig()
    desired_layouts = {"nn.conv2d": ["NCHW4c", "OIHW4o"]}
    with layout_config:
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = tvm.IRModule.from_expr(tvm_mod)
            tvm_mod_nchwc = seq(mod)
            return tvm_mod_nchwc
