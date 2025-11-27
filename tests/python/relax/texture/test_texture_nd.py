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

import os
import tvm
import tvm.testing
import pytest
import tempfile
import numpy as np

from tvm import (
    relax,
    IRModule,
)
from tvm.relax.transform.legalize_ops import adreno as legalize_adreno
from tvm.script import ir as I, tir as T
from tvm.target import Target
from tvm.contrib import ndk
from tvm import tir, DataType


class RemoteConnection:
    def __init__(self):
        self.RPC_TRACKER_HOST = os.getenv("TVM_TRACKER_HOST", "localhost")
        self.RPC_TRACKER_PORT = int(os.getenv("TVM_TRACKER_PORT", 7979))
        self.RPC_KEY = os.getenv("RPC_DEVICE_KEY", "android")
        self.tracker = tvm.rpc.connect_tracker(self.RPC_TRACKER_HOST, self.RPC_TRACKER_PORT)

    def __enter__(self):
        self.remote = self.tracker.request(self.RPC_KEY, priority=0, session_timeout=600)
        return self.remote

    def __exit__(self, exc_type, exc_value, traceback):
        self.remote.get_function("CloseRPCConnection")()


def preprocess_pipeline(mod: IRModule) -> IRModule:
    desired_layouts = {"relax.nn.conv2d": ["NCHW16c", "OIHW16o", "NCHW16c"]}
    seq = tvm.transform.Sequential(
        [
            tvm.tir.transform.BindTarget(Target.current(allow_none=False)),
            tvm.relax.transform.FoldConstant(),
            tvm.relax.transform.DecomposeOpsForInference(),
            tvm.relax.transform.FoldConstant(),
            tvm.tir.transform.BindTarget(tvm.target.Target.current(allow_none=False)),
            tvm.relax.transform.ConvertLayout(desired_layouts),
            tvm.relax.transform.Normalize(),
            tvm.relax.transform.FoldConstant(),
            tvm.relax.transform.LegalizeOps(),
            tvm.relax.transform.LegalizeOps(
                {"relax.nn.conv2d": legalize_adreno.conv2d_NCHWc_OIHWo}
            ),
            tvm.relax.transform.FoldConstant(),
            tvm.relax.transform.AnnotateTIROpPattern(),
            tvm.relax.transform.FuseOps(),
            tvm.relax.transform.FuseTIR(),
            tvm.relax.transform.DeadCodeElimination(),
            tvm.relax.transform.Normalize(),
        ]
    )
    mod = seq(mod)
    return mod


def postprocess_pipeline(mod: IRModule) -> IRModule:
    seq = tvm.transform.Sequential(
        [
            tvm.relax.transform.ToNonDataflow(),
            tvm.relax.transform.RemovePurityChecking(),
            tvm.relax.transform.CallTIRRewrite(),
            tvm.relax.transform.Normalize(),
            tvm.relax.transform.StaticPlanBlockMemory(),
            tvm.relax.transform.LowerAllocTensor(),
            tvm.relax.transform.KillAfterLastUse(),
            tvm.relax.transform.LowerRuntimeBuiltin(),
            tvm.relax.transform.VMShapeLower(),
            tvm.relax.transform.AttachGlobalSymbol(),
        ]
    )
    mod = seq(mod)
    return mod


@tvm.testing.requires_rpc
@tvm.testing.requires_opencl
@pytest.mark.parametrize(
    "target", [Target("opencl -device=adreno", "llvm -mtriple=aarch64-linux-android")]
)
@pytest.mark.parametrize("dtype", ["int8", "float16", "int16", "float32", "int32"])
@pytest.mark.parametrize("channel_size", [64, 128])
@pytest.mark.parametrize("read_width", [1, 2, 4, 8, 16])
def test_texture_copy(target, dtype, channel_size, read_width):
    M, N, K = (256, 1024, 128)
    lanes = channel_size // DataType(dtype).bits
    if read_width > lanes:
        return

    @I.ir_module
    class TextureCopy:
        @T.prim_func
        def main(A: T.Buffer((M, N), dtype), B: T.Buffer((M, N), dtype)):
            T.func_attr({"global_symbol": "main"})
            for li, lj in T.grid(M, N):
                with T.block("Copy"):
                    i, j = T.axis.remap("SS", [li, lj])
                    B[i, j] = A[i, j]

    def schedule_texture_read(sch: tir.Schedule):
        B_blk = sch.get_block("Copy")
        Ai_block = sch.cache_read(B_blk, 0, "global.texture")
        sch.transform_layout(Ai_block, ("write", 0), lambda i, j: (i, j // lanes, j % lanes))

        def schedule_default(blk, lanes):
            i, j = sch.get_loops(blk)
            jo, jv = sch.split(j, [None, lanes])

            b = sch.fuse(i, jo)
            bx, tx = sch.split(b, [None, 256])
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")

            sch.vectorize(jv)

        schedule_default(Ai_block, lanes)
        schedule_default(B_blk, read_width)

    mod = TextureCopy
    with target:
        mod = preprocess_pipeline(mod)
        sch = tir.Schedule(mod)
        schedule_texture_read(sch)
        mod = postprocess_pipeline(sch.mod)

    ex = relax.build(mod, target)
    load_path = "vm_library.so"
    inputs = [np.random.randint(0, 128, (M, N)).astype(dtype), np.zeros((M, N), dtype)]
    with RemoteConnection() as remote:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = temp_dir + "/" + load_path
            ex.export_library(path, fcompile=ndk.create_shared, options=["-shared", "-fPIC", "-lm"])

            remote.upload(path)
            rexec = remote.load_module(load_path)
            dev = remote.cl()

            vm = relax.VirtualMachine(rexec, [dev, dev, dev])
            inps = [tvm.runtime.tensor(inp, dev) for inp in inputs]
            vm["main"](*inps)

            np.testing.assert_equal(inps[-1].numpy(), inps[0].numpy())


if __name__ == "__main__":
    tvm.testing.main()
