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
import numpy as np
from tvm.contrib import graph_executor as graph_runtime
from tvm.relay.op.contrib import vsi_npu
from tvm import rpc
from tvm.contrib import utils as util
import os

RPC_HOST = os.environ["RPC_HOST"]
RPC_PORT = int(os.environ["RPC_PORT"])
CROSS_CC = os.environ["CROSS_CC"]
ROOTFS = os.environ["ROOTFS"]

def make_module(func, params):
    func = relay.Function(relay.analysis.free_vars(func), func)
    if params:
        relay.build_module.bind_params_by_name(func, params)
    return tvm.IRModule.from_expr(func)

def get_vsi_model(mod, params):
    remote = rpc.connect(RPC_HOST, RPC_PORT)
    tmp_path = util.tempdir()
    lib_name = "model.so"
    lib_path = tmp_path.relpath(lib_name)

    kwargs = {}
    kwargs["cc"] = CROSS_CC
    target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
    kwargs["options"] = ["-L"+ROOTFS+"/lib " , "-L" +ROOTFS+"/usr/lib " , "-L" + ROOTFS+"/usr/lib/aarch64-poky-linux/9.2.0 ", "--sysroot=" +ROOTFS]
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = vsi_npu.partition_for_vsi_npu(mod, params)
        lib  = relay.build(mod, target, params=params)
        lib.export_library(lib_path, fcompile=False, **kwargs)

    remote.upload(lib_path)
    lib = remote.load_module(lib_name)
    ctx = remote.cpu()

    rt_mod = graph_runtime.GraphModule(lib["default"](ctx))
    return rt_mod, ctx

def get_vsi_result(data, mod, params, out_shape, dtype):
    rt_mod, ctx = get_vsi_model(mod, params)
    rt_mod.set_input(**data)
    rt_mod.run()
    rt_out = tvm.nd.array(np.zeros(out_shape, dtype=dtype), ctx)
    rt_mod.get_output(0, rt_out)
    print(data)
    return rt_out

def benchmark_vsi(mod, params, repeat=50):
    rt_mod, ctx = get_vsi_model(mod, params)

    print("Evaluate graph runtime inference cost on VSI NPU")
    ftimer = rt_mod.module.time_evaluator("run", ctx, number=1, repeat=repeat)
    # Measure in millisecond.
    prof_res = np.array(ftimer().results) * 1000
    print("VSI NPU runtime inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res)))

    return np.mean(prof_res)

def get_ref_result(data, mod, params, out_shape, dtype):
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        lib  = relay.build(mod, target, params=params)
    print(mod.astext())

    cpu_mod = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
    cpu_mod.set_input(**data)
    cpu_mod.run()
    cpu_out = cpu_mod.get_output(0, tvm.nd.empty(out_shape, dtype))
    return cpu_out

def verify_vsi_result(inputs, model, params, data_shape, out_shape, dtype="float32"):
    wait=input("1. press any key and continue...")
    mod = make_module(model, params)

    ref_out = get_ref_result(inputs, mod, params, out_shape, dtype)

    try:
        vsi_out = get_vsi_result(inputs, mod, params, out_shape, dtype)
        print("ref_out",ref_out)
        print("vsi_out",vsi_out)
        if dtype == "uint8":
            atol = 1
            rtol = 1.0 / 255
        else:
            atol = 1e-3
            rtol = 1e-3
        tvm.testing.assert_allclose(ref_out.asnumpy(), vsi_out.asnumpy(), rtol=rtol, atol=atol)

    except Exception as err:
         print("\nExpected output: ")
         print(ref_out.asnumpy())
         print("Actual output: ")
         print(err)
         print("FAIL")
    else:
         print("PASS")
