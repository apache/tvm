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
"""LSTM Example, still work in progress.."""
import os

import numpy as np
import tvm
from tvm import te
from tvm.contrib import nvcc

# Quick knobs
TASK = "lstm"
USE_MANUAL_CODE = False
PERSIST_KERNEL = True
DETECT_GLOBAL_BARRIER = PERSIST_KERNEL
SKIP_CHECK = False
UNROLL_WLOAD = True


@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code, target):
    """Use nvcc compiler for better perf."""
    ptx = nvcc.compile_cuda(code, target_format="ptx")
    return ptx


def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)


@tvm.register_func
def tvm_callback_cuda_postproc(code, target):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code


def lstm():
    if not PERSIST_KERNEL:
        raise ValueError("Non persist LSTM not yet supported")
    num_thread_y = 8
    num_thread_x = 16 * 3 // 2
    num_sm = 24
    n_num_step = 128
    num_step = te.var("num_step")
    num_hidden = 1152 // 2
    batch_size = 1
    # Global transition matrix
    # Input hidden channel can be pre-caculated by a gemm
    Xi2h = te.placeholder((num_step, batch_size, 4, num_hidden), name="Xi2h")
    # Only handle hidden transition, saves space.
    Wh2h = te.placeholder((4, num_hidden, num_hidden), name="Wh2h")
    # h: output hidden state, c: cell state.
    s_state_h = te.placeholder((num_step, batch_size, num_hidden))
    s_state_c = te.placeholder((num_step, batch_size, num_hidden))
    s_init_c = te.compute((1, batch_size, num_hidden), lambda *i: 0.0, name="init_c")
    s_init_h = te.compute((1, batch_size, num_hidden), lambda *i: 0.0, name="init_h")
    # LSTM transition
    k = te.reduce_axis((0, num_hidden), name="ki2h")
    s_h2h = te.compute(
        (num_step, batch_size, 4, num_hidden),
        lambda t, i, x, j: te.sum(s_state_h[t - 1, i, k] * Wh2h[x, j, k], axis=k),
        name="s_h2h",
    )
    # Gate rules
    gates = te.compute(Xi2h.shape, lambda *i: Xi2h(*i) + s_h2h(*i), name="gates")
    gshape = (num_step, batch_size, num_hidden)
    in_gate = te.compute(gshape, lambda t, i, j: te.sigmoid(gates[t, i, 0, j]), name="in_gate")
    in_transform = te.compute(
        gshape, lambda t, i, j: te.tanh(gates[t, i, 1, j]), name="in_transform"
    )
    forget_gate = te.compute(
        gshape, lambda t, i, j: te.sigmoid(gates[t, i, 2, j]), name="forget_gate"
    )
    out_gate = te.compute(gshape, lambda t, i, j: te.sigmoid(gates[t, i, 3, j]), name="out_gate")
    next_c = te.compute(
        gshape,
        lambda t, i, j: forget_gate[t, i, j] * s_state_c[t - 1, i, j]
        + in_gate[t, i, j] * in_transform[t, i, j],
        name="next_c",
    )
    next_h = te.compute(
        gshape, lambda t, i, j: out_gate[t, i, j] * te.tanh(next_c[t, i, j]), name="next_h"
    )
    update_c = te.compute(gshape, lambda *i: next_c(*i), name="update_c")
    update_h = te.compute(gshape, lambda *i: next_h(*i), name="update_h")
    # schedule
    scan_h, scan_c = tvm.te.scan(
        [s_init_h, s_init_c],
        [update_h, update_c],
        [s_state_h, s_state_c],
        inputs=[Xi2h],
        name="lstm_scan",
    )
    # schedule
    s = te.create_schedule(scan_h.op)
    # Inline gate computations
    s[gates].compute_inline()
    s[in_gate].compute_inline()
    s[in_transform].compute_inline()
    s[forget_gate].compute_inline()
    s[out_gate].compute_inline()

    block_x = te.thread_axis((0, num_sm), "blockIdx.x")
    thread_x = te.thread_axis((0, num_thread_x), "threadIdx.x")
    thread_y = te.thread_axis((0, num_thread_y), "threadIdx.y")

    s_state_h_S = s.cache_read(s_state_h, "shared", [s_h2h])
    s_state_c_S = s.cache_read(s_state_c, "shared", [next_c])
    Wh2hL = s.cache_read(Wh2h, "local", [s_h2h])

    ko, ki = s[s_h2h].split(s[s_h2h].op.reduce_axis[0], nparts=num_thread_y)
    s_h2h_rf = s.rfactor(s_h2h, ko)
    s[s_h2h].bind(s[s_h2h].op.reduce_axis[0], thread_y)
    s[s_h2h_rf].compute_at(s[s_h2h], s[s_h2h].op.reduce_axis[0])

    if PERSIST_KERNEL:
        s[scan_h.op].env_threads([block_x, thread_y, thread_x])
        s[Wh2hL].compute_at(s[scan_h.op], thread_x)
    else:
        s[Wh2hL].compute_at(s[s_h2h], s[s_h2h].op.axis[3])

    if UNROLL_WLOAD:
        s[Wh2hL].unroll(Wh2hL.op.axis[0])
        s[Wh2hL].unroll(Wh2hL.op.axis[2])

    s[s_state_h_S].compute_at(s[s_h2h_rf], s[s_h2h_rf].op.axis[3])
    s[s_state_c_S].compute_at(s[scan_h.op], s[scan_h].op.scan_axis)

    for ss in [s_state_h_S]:
        xo, xi = s[ss].split(ss.op.axis[2], factor=num_thread_x * num_thread_y)
        ty, xi = s[ss].split(xi, nparts=num_thread_y)
        tx, xi = s[ss].split(xi, nparts=num_thread_x)
        s[ss].bind(ty, thread_y)
        s[ss].bind(tx, thread_x)

    for init in [s_init_c, s_init_h]:
        bx, xi = s[init].split(init.op.axis[2], nparts=num_sm)
        tx, xi = s[init].split(xi, nparts=num_thread_x)
        s[init].bind(bx, block_x)
        s[init].bind(tx, thread_x)

    s[next_c].set_store_predicate(thread_y.equal(0))
    s[next_h].set_store_predicate(thread_y.equal(0))

    for update in [update_c, update_h]:
        bx, xi = s[update].split(s[update].op.axis[2], nparts=num_sm)
        tx, xi = s[update].split(xi, nparts=num_thread_x)
        s[update].bind(bx, block_x)
        s[update].bind(tx, thread_x)
        s[update].set_store_predicate(thread_y.equal(0))

    # verify we can lower correctly
    def check_device(target):
        num_step = n_num_step
        flstm = tvm.build(s, [Xi2h, Wh2h, scan_h, scan_c], target)
        dev = tvm.cuda(0) if target == "cuda" else tvm.cl(0)
        # launch the kernel.
        scan_h_np = np.zeros((num_step, batch_size, num_hidden)).astype("float32")
        scan_c_np = np.zeros((num_step, batch_size, num_hidden)).astype("float32")
        Xi2h_np = np.random.normal(size=(num_step, batch_size, 4, num_hidden)).astype("float32")
        Wh2h_np = np.random.normal(size=(4, num_hidden, num_hidden)).astype("float32")
        scan_h_a = tvm.nd.array(scan_h_np, dev)
        scan_c_a = tvm.nd.array(scan_c_np, dev)
        Xi2h_a = tvm.nd.array(Xi2h_np, dev)
        Wh2h_a = tvm.nd.array(Wh2h_np, dev)
        flstm(Xi2h_a, Wh2h_a, scan_h_a, scan_c_a)
        dev.sync()
        # measure time cost of second step.
        evaluator = flstm.time_evaluator(flstm.entry_name, dev, 1, repeat=1000)
        eval_result = evaluator(Xi2h_a, Wh2h_a, scan_h_a, scan_c_a)
        print("Time cost=%g" % eval_result.mean)

    # set unroll_explicit for more readable code.
    with tvm.transform.PassContext(
        config={
            "tir.UnrollLoop": {
                "auto_max_step": 128,
            },
            "tir.detect_global_barrier": DETECT_GLOBAL_BARRIER,
        }
    ):
        check_device("cuda")


if __name__ == "__main__":
    lstm()
