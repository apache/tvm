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
import numpy as np
import collections
import json

import torch
from torchvision.models.quantization import resnet as qresnet

import tvm.testing
import tvm.utils
import tvm.tir.tensor_intrin
from tvm._ffi import get_global_func
from tvm import te, tir, relay
from tvm.script import tir as T
from tvm import meta_schedule as ms


ChromeTraceEvent = collections.namedtuple("ChromeTraceEvent", ["ts", "tid", "pid", "name", "ph"])


def convert_to_chrome_trace(log: str, trace_file="rt_trace.json"):
    """Dump the trace to the Chrome trace.json format."""
    naming = [
        "POOL_SUBMISSION",
        "POOL_EXECUTION",
        "POOL_FINISH_WAIT",
        "ALLOC",
        "FREE",
        "KERNEL",
        "DMA_WAIT",
    ]
    load_wid = True
    events = []
    wid_name = {}
    for line in log.splitlines():
        if line == "WID_NAMES":
            continue
        if line == "EVENTS":
            load_wid = False
            continue

        if load_wid:
            wid, name = line.split(" ")
            wid = int(wid)
            wid_name[wid] = name
        else:
            kind, tid, wid, ts_ns = (int(l) for l in line.split(" "))
            kind_name = naming[kind // 2] if kind // 2 < len(naming) else str(kind // 2)
            if wid in wid_name:
                name = f"{wid_name[wid]}_{kind_name}"
            else:
                name = f"{wid}_{kind_name}"

            ph = "B" if kind % 2 == 0 else "E"
            ts_us = ts_ns / 10**3
            events.append(ChromeTraceEvent(
                ts=ts_us,
                tid=tid,
                pid=1,
                ph=ph,
                name=name,
            ))

    result = dict(displayTimeUnit="ns", traceEvents=[e._asdict() for e in events])

    with open(trace_file, "w") as trace_f:
        json.dump(result, trace_f)


def some_compute(m, n, k):
    X = te.placeholder((m, k), name="X", dtype="uint8")
    packed_width = te.placeholder((n // 32, k // 4, 32, 4), name="W", dtype="uint8")  # OI32o4i

    axis_k = te.reduce_axis((0, k), name="k")
    out = te.compute(
        (m, n),
        lambda i, j: te.sum(
            X[i, axis_k].astype("int32")
            * packed_width[
                tvm.tir.indexdiv(j, 32), tvm.tir.indexdiv(axis_k, 4), j % 32, axis_k % 4
            ].astype("int32"),
            axis=axis_k,
            ),
    )
    return te.create_prim_func([X, packed_width, out])


def some_schedule_apply(sch, dev_type="hex") -> None:
    b0 = sch.get_block(name="compute", func_name="main")
    m, n, k = sch.get_loops(block=b0)
    # VRMPY intrinsic
    n_, n_b = sch.split(loop=n, factors=[None, 32], preserve_unit_iters=True)
    k_, k_b = sch.split(loop=k, factors=[None, 4], preserve_unit_iters=True)
    sch.reorder(k_, n_b, k_b)  # m, n_o, k_o, n_b, k_b
    b_compute_o = sch.blockize(loop=n_b)

    # Macro block to unroll
    mb_m, mb_n, mb_k = 4, 4, 4
    m_o, m_i = sch.split(loop=m, factors=[None, mb_m], preserve_unit_iters=True)
    n_o, n_i = sch.split(loop=n_, factors=[None, mb_n], preserve_unit_iters=True)
    k_o, k_i = sch.split(loop=k_, factors=[None, mb_k], preserve_unit_iters=True)
    sch.reorder(m_o, n_o, k_o, m_i, n_i, k_i)

    # Apply parallel, and unroll
    p_mn_o = sch.fuse(m_o, n_o)
    sch.parallel(loop=p_mn_o)
    sch.annotate(block_or_loop=p_mn_o, ann_key="pragma_auto_unroll_max_step", ann_val=64)

    # Vectorization of init loop
    b_compute_o_init = sch.decompose_reduction(block=b_compute_o, loop=k_o)  # decompose inside parallel loop
    b56, = sch.get_child_blocks(b_compute_o_init)
    l57, = sch.get_loops(block=b56)
    sch.vectorize(loop=l57)  # vectorize compute_o_init.

    # Vectorization of compute loop
    if dev_type == "hex":
        b58 = sch.get_block(name="compute_o_update", func_name="main")
        sch.tensorize(block_or_loop=b58, tensor_intrin="dot_32x4_u8u8i32_vrmpy")


@tvm.testing.requires_llvm
def test_grab_trace_host():
    llvm_target = tvm.target.Target("llvm")
    target = tvm.target.Target(llvm_target, host=llvm_target)

    rt_trace_get_info = get_global_func("runtime.rt_trace_logger.get_log")
    rt_trace_enable = get_global_func("runtime.rt_trace_logger.enable")

    m, n, k = 256, 256, 512

    mod = some_compute(m, n, k)
    sch = tir.Schedule(mod)
    some_schedule_apply(sch, dev_type="host")
    lib = tvm.build(sch.mod, target=target, name="dense")
    dev = tvm.cpu(0)

    rt_mod = lib
    def make_arg(info):
        shape = [int(d) for d in info.shape]
        np_arr = np.random.default_rng().integers(0, 16, shape, dtype=info.dtype)
        return tvm.runtime.ndarray.array(np_arr, device=dev, mem_scope="global")

    args_info = ms.arg_info.ArgInfo.from_prim_func(mod)
    args = [make_arg(info) for info in args_info]

    # Collect traces
    rt_mod(*args)
    rt_trace_enable(1)
    rt_mod(*args)
    rt_mod(*args)
    rt_trace_enable(0)
    rt_mod(*args)

    prof_info = rt_trace_get_info()
    convert_to_chrome_trace(log=prof_info, trace_file="rt_trace_host.json")


@tvm.testing.requires_hexagon
def test_grab_trace_hex(hexagon_session):
    dev_target = tvm.target.hexagon("v68")
    target = tvm.target.Target(dev_target, host=dev_target)

    rt_trace_get_info = hexagon_session.get_function("runtime.rt_trace_logger.get_log")
    rt_trace_enable = hexagon_session.get_function("runtime.rt_trace_logger.enable")

    m, n, k = 256, 256, 512

    mod = some_compute(m, n, k)
    sch = tir.Schedule(mod)
    some_schedule_apply(sch)
    lib = tvm.build(sch.mod, target=target, name="dense")

    dev = hexagon_session.device
    rt_mod = hexagon_session.load_module(lib)

    def make_arg(info):
        shape = [int(d) for d in info.shape]
        np_arr = np.random.default_rng().integers(0, 16, shape, dtype=info.dtype)
        return tvm.runtime.ndarray.array(np_arr, device=dev, mem_scope="global")

    args_info = ms.arg_info.ArgInfo.from_prim_func(mod)
    args = [make_arg(info) for info in args_info]

    # Collect traces
    rt_mod(*args)
    rt_trace_enable(1)
    rt_mod(*args)
    rt_mod(*args)
    rt_trace_enable(0)
    rt_mod(*args)

    prof_info = rt_trace_get_info()
    convert_to_chrome_trace(log=prof_info, trace_file="rt_trace_hex.json")


@tvm.script.ir_module
class AnnotatedModule:
    @T.prim_func
    def main(X: T.Buffer((256, 512), "uint8"),
             W: T.Buffer((8, 128, 32, 4), "uint8"),
             compute: T.Buffer((256, 256), "int32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i_0_j_0_0_fused in T.parallel(128, annotations={"pragma_auto_unroll_max_step": 64}):
            T.evaluate(T.call_extern("TVMRtTracePutRecord", 0, dtype=""))
            for i_1_init, j_0_1_init in T.grid(4, 4):
                with T.block("compute_o_init"):
                    v_i = T.axis.spatial(256, i_0_j_0_0_fused // 2 * 4 + i_1_init)
                    v_j_o = T.axis.spatial(8, i_0_j_0_0_fused % 2 * 4 + j_0_1_init)
                    T.reads()
                    T.writes(compute[v_i, v_j_o * 32:v_j_o * 32 + 32])
                    for j_1 in T.vectorized(32):
                        with T.block("compute_init"):
                            v_j_i_init = T.axis.spatial(32, j_1)
                            T.reads()
                            T.writes(compute[v_i, v_j_o * 32 + v_j_i_init])
                            compute[v_i, v_j_o * 32 + v_j_i_init] = 0
            for k_0_0, i_1, j_0_1, k_0_1 in T.grid(32, 4, 4, 4):
                with T.block("compute_o_update"):
                    v_i = T.axis.spatial(256, i_0_j_0_0_fused // 2 * 4 + i_1)
                    v_j_o = T.axis.spatial(8, i_0_j_0_0_fused % 2 * 4 + j_0_1)
                    v_k_o = T.axis.reduce(128, k_0_0 * 4 + k_0_1)
                    T.reads(compute[v_i, v_j_o * 32:v_j_o * 32 + 32], X[v_i, v_k_o * 4:v_k_o * 4 + 4], W[v_j_o, v_k_o, 0:32, 0:4])
                    T.writes(compute[v_i, v_j_o * 32:v_j_o * 32 + 32])
                    A = T.match_buffer(X[v_i, v_k_o * 4:v_k_o * 4 + 4], (4,), "uint8", offset_factor=1)
                    B = T.match_buffer(W[v_j_o, v_k_o, 0:32, 0:4], (32, 4), "uint8", offset_factor=1)
                    C = T.match_buffer(compute[v_i, v_j_o * 32:v_j_o * 32 + 32], (32,), "int32", offset_factor=1)
                    A_u8x4: T.uint8x4 = A[0:4]
                    A_i32: T.int32 = T.reinterpret("int32", A_u8x4)
                    B_i8x128 = B[0, 0:128]
                    B_i32x32: T.int32x32 = T.reinterpret("int32x32", B_i8x128)
                    C[0:32] = T.call_llvm_pure_intrin("int32x32", T.uint32(4374), T.uint32(3), C[0:32], B_i32x32, A_i32)
            T.evaluate(T.call_extern("TVMRtTracePutRecord", 0 + 1, dtype=""))


@tvm.testing.requires_hexagon
def test_annotation_hex(hexagon_session):
    dev_target = tvm.target.hexagon("v68")
    target = tvm.target.Target(dev_target, host=dev_target)

    rt_trace_get_info = hexagon_session.get_function("runtime.rt_trace_logger.get_log")
    rt_trace_enable = hexagon_session.get_function("runtime.rt_trace_logger.enable")

    mod = AnnotatedModule
    lib = tvm.build(mod, target=target, name="dense")

    dev = hexagon_session.device
    rt_mod = hexagon_session.load_module(lib)

    def make_arg(info):
        shape = [int(d) for d in info.shape]
        np_arr = np.random.default_rng().integers(0, 16, shape, dtype=info.dtype)
        return tvm.runtime.ndarray.array(np_arr, device=dev, mem_scope="global")

    args_info = ms.arg_info.ArgInfo.from_prim_func(mod["main"])
    args = [make_arg(info) for info in args_info]

    # Collect traces
    rt_mod(*args)
    rt_trace_enable(1)
    rt_mod(*args)
    rt_mod(*args)
    rt_trace_enable(0)
    rt_mod(*args)

    prof_info = rt_trace_get_info()
    convert_to_chrome_trace(log=prof_info, trace_file="rt_trace_annotated_hex.json")


@tvm.testing.requires_hexagon
def test_resnet_trace(hexagon_session):
    dev_target = tvm.target.hexagon("v68")
    target = tvm.target.Target(dev_target, host=dev_target)

    rt_trace_get_info = hexagon_session.get_function("runtime.rt_trace_logger.get_log")
    rt_trace_enable = hexagon_session.get_function("runtime.rt_trace_logger.enable")

    def quantize_model(model, inp):
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(model, inplace=True)
        model(inp)
        torch.quantization.convert(model, inplace=True)

    pt_model = qresnet.resnet50(pretrained=True).eval()

    pt_inp = torch.randn(1, 3, 224, 224)
    tvm_inp = np.random.randn(1, 3, 224, 224).astype("float32")
    quantize_model(pt_model, pt_inp)
    script_module = torch.jit.trace(pt_model, pt_inp).eval()

    # import
    input_name = "image"
    input_shapes = [(input_name, pt_inp.shape)]
    mod, params = relay.frontend.from_pytorch(
        script_module, input_shapes, keep_quantized_weight=True
    )

    # build
    with tvm.transform.PassContext(opt_level=3):
        hexagon_lowered = relay.build(
            mod,
            target=target,
            params=params,
            executor=relay.backend.Executor("graph", {"link-params": True}),
        )

    # run
    graph_mod = hexagon_session.get_executor_from_factory(hexagon_lowered)
    graph_mod.set_input(input_name, tvm_inp.copy())
    rt_trace_enable(1)
    graph_mod.run()
    rt_trace_enable(0)

    prof_info = rt_trace_get_info()
    convert_to_chrome_trace(log=prof_info, trace_file="rt_trace_resnet_hex.json")


if __name__ == "__main__":
    tvm.testing.main()
