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
"""Unit test VTA's instructions """
import tvm
from tvm import te
import numpy as np
from tvm import topi
from tvm.contrib import utils

import vta
import vta.testing
from vta.testing import simulator

np.random.seed(0xDEADB)


def test_save_load_out():
    """Test save/store output command"""

    def _run(env, remote):
        n = 6
        x = te.placeholder((n, n, env.BATCH, env.BLOCK_OUT), name="x", dtype=env.acc_dtype)
        x_buf = te.compute((n, n, env.BATCH, env.BLOCK_OUT), lambda *i: x(*i), "x_buf")
        # insert no-op that won't be optimized away
        y_buf = te.compute((n, n, env.BATCH, env.BLOCK_OUT), lambda *i: x_buf(*i) >> 0, "y_buf")
        y = te.compute(
            (n, n, env.BATCH, env.BLOCK_OUT), lambda *i: y_buf(*i).astype(env.inp_dtype), "y"
        )
        # schedule
        s = te.create_schedule(y.op)
        s[x_buf].set_scope(env.acc_scope)
        s[x_buf].pragma(x_buf.op.axis[0], env.dma_copy)
        s[y_buf].set_scope(env.acc_scope)
        s[y_buf].pragma(y_buf.op.axis[0], env.alu)
        s[y].pragma(y.op.axis[0], env.dma_copy)

        # verification
        with vta.build_config():
            m = vta.build(s, [x, y], tvm.target.Target("ext_dev", host=env.target_host))

        if not remote:
            return
        temp = utils.tempdir()
        m.save(temp.relpath("load_act.o"))
        remote.upload(temp.relpath("load_act.o"))
        f = remote.load_module("load_act.o")
        # verify
        dev = remote.ext_dev(0)
        x_np = np.random.randint(1, 10, size=(n, n, env.BATCH, env.BLOCK_OUT)).astype(x.dtype)
        y_np = x_np.astype(y.dtype)
        x_nd = tvm.nd.array(x_np, dev)
        y_nd = tvm.nd.empty(y_np.shape, device=dev, dtype=y_np.dtype)

        if env.TARGET in ["sim", "tsim"]:
            simulator.clear_stats()

        f(x_nd, y_nd)

        np.testing.assert_equal(y_np, y_nd.numpy())

        if env.TARGET in ["sim", "tsim"]:
            sim_stats = simulator.stats()
            print("Save load execution statistics:")
            for k, v in sim_stats.items():
                print("\t{:<16}: {:>16}".format(k, v))

    vta.testing.run(_run)


def test_padded_load():
    """Test padded load."""

    def _run(env, remote):
        def check_padded_load(pad_before, pad_after, test_name=None):
            # declare
            n = 3
            m = 5
            x = te.placeholder((n, m, env.BATCH, env.BLOCK_OUT), name="x", dtype=env.acc_dtype)
            x_buf = topi.nn.pad(x, pad_before, pad_after, name="y")
            # insert no-op that won't be optimized away
            y_buf = te.compute(
                (
                    n + pad_before[0] + pad_after[0],
                    m + pad_before[1] + pad_after[1],
                    env.BATCH,
                    env.BLOCK_OUT,
                ),
                lambda *i: x_buf(*i) >> 0,
                "y_buf",
            )
            y = te.compute(
                (
                    n + pad_before[0] + pad_after[0],
                    m + pad_before[1] + pad_after[1],
                    env.BATCH,
                    env.BLOCK_OUT,
                ),
                lambda *i: y_buf(*i).astype(env.inp_dtype),
                "y",
            )
            # schedule
            s = te.create_schedule(y.op)
            s[x_buf].set_scope(env.acc_scope)
            s[x_buf].pragma(x_buf.op.axis[0], env.dma_copy)
            s[y_buf].set_scope(env.acc_scope)
            s[y_buf].pragma(y_buf.op.axis[0], env.alu)
            s[y].pragma(y.op.axis[0], env.dma_copy)
            # build
            with vta.build_config():
                mod = vta.build(s, [x, y], tvm.target.Target("ext_dev", host=env.target_host))

            if not remote:
                return
            temp = utils.tempdir()
            mod.save(temp.relpath("padded_load.o"))
            remote.upload(temp.relpath("padded_load.o"))
            f = remote.load_module("padded_load.o")
            # verify
            dev = remote.ext_dev(0)
            x_np = np.random.randint(0, 10, size=(n, m, env.BATCH, env.BLOCK_OUT)).astype(x.dtype)
            y_np = np.zeros(
                (
                    n + pad_before[0] + pad_after[0],
                    m + pad_before[1] + pad_after[1],
                    env.BATCH,
                    env.BLOCK_OUT,
                )
            ).astype(y.dtype)
            y_np[pad_before[0] : pad_before[0] + n, pad_before[1] : pad_before[1] + m, :] = x_np
            x_nd = tvm.nd.array(x_np, dev)
            y_nd = tvm.nd.empty(y_np.shape, device=dev, dtype=y_np.dtype)

            if env.TARGET in ["sim", "tsim"]:
                simulator.clear_stats()

            f(x_nd, y_nd)

            np.testing.assert_equal(y_np, y_nd.numpy())

            if env.TARGET in ["sim", "tsim"]:
                sim_stats = simulator.stats()
                print("Padded {} load execution statistics:".format(test_name))
                for k, v in sim_stats.items():
                    print("\t{:<16}: {:>16}".format(k, v))

        check_padded_load([2, 0, 0, 0], [0, 0, 0, 0], test_name="Y0")
        check_padded_load([0, 2, 0, 0], [0, 0, 0, 0], test_name="Y1")
        check_padded_load([0, 0, 0, 0], [2, 0, 0, 0], test_name="X0")
        check_padded_load([0, 0, 0, 0], [0, 2, 0, 0], test_name="X1")
        check_padded_load([1, 1, 0, 0], [1, 1, 0, 0], test_name="all")

    vta.testing.run(_run)


def test_gemm():
    """Test GEMM."""

    def _run(env, remote):
        # declare
        o = 4
        n = 1
        m = 4
        x = te.placeholder((o, n, env.BATCH, env.BLOCK_IN), name="x", dtype=env.inp_dtype)
        w = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN), name="w", dtype=env.wgt_dtype)
        x_buf = te.compute((o, n, env.BATCH, env.BLOCK_IN), lambda *i: x(*i), "x_buf")
        w_buf = te.compute((m, n, env.BLOCK_OUT, env.BLOCK_IN), lambda *i: w(*i), "w_buf")
        ko = te.reduce_axis((0, n), name="ko")
        ki = te.reduce_axis((0, env.BLOCK_IN), name="ki")
        y_gem = te.compute(
            (o, m, env.BATCH, env.BLOCK_OUT),
            lambda bo, co, bi, ci: te.sum(
                x_buf[bo, ko, bi, ki].astype(env.acc_dtype)
                * w_buf[co, ko, ci, ki].astype(env.acc_dtype),
                axis=[ko, ki],
            ),
            name="y_gem",
        )
        y_shf = te.compute(
            (o, m, env.BATCH, env.BLOCK_OUT), lambda *i: y_gem(*i) >> 8, name="y_shf"
        )
        y_max = te.compute(
            (o, m, env.BATCH, env.BLOCK_OUT), lambda *i: tvm.te.max(y_shf(*i), 0), "y_max"
        )  # relu
        y_min = te.compute(
            (o, m, env.BATCH, env.BLOCK_OUT),
            lambda *i: tvm.te.min(y_max(*i), (1 << (env.INP_WIDTH - 1)) - 1),
            "y_min",
        )  # relu
        y = te.compute(
            (o, m, env.BATCH, env.BLOCK_OUT), lambda *i: y_min(*i).astype(env.inp_dtype), name="y"
        )

        if not remote:
            return

        def verify(s, name=None):
            # Build with the CSE pass disabled as otherwise it would complicate the test
            with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
                mod = vta.build(s, [x, w, y], tvm.target.Target("ext_dev", host=env.target_host))
            temp = utils.tempdir()
            mod.save(temp.relpath("gemm.o"))
            remote.upload(temp.relpath("gemm.o"))
            f = remote.load_module("gemm.o")
            # verify
            dev = remote.ext_dev(0)
            x_np = np.random.randint(-128, 128, size=(o, n, env.BATCH, env.BLOCK_IN)).astype(
                x.dtype
            )
            w_np = np.random.randint(-128, 128, size=(m, n, env.BLOCK_OUT, env.BLOCK_IN)).astype(
                w.dtype
            )
            y_np = np.zeros((o, m, env.BATCH, env.BLOCK_OUT)).astype(y.dtype)
            x_nd = tvm.nd.array(x_np, dev)
            w_nd = tvm.nd.array(w_np, dev)
            y_nd = tvm.nd.array(y_np, dev)
            y_np = y_np.astype(env.acc_dtype)
            for b in range(o):
                for i in range(m):
                    for j in range(n):
                        y_np[b, i, :] += np.dot(
                            x_np[b, j, :].astype(env.acc_dtype), w_np[i, j].T.astype(env.acc_dtype)
                        )
            y_np = np.right_shift(y_np, 8)
            y_np = np.clip(y_np, 0, (1 << (env.INP_WIDTH - 1)) - 1).astype(y.dtype)

            if env.TARGET in ["sim", "tsim"]:
                simulator.clear_stats()

            f(x_nd, w_nd, y_nd)

            np.testing.assert_equal(y_np, y_nd.numpy())

            if env.TARGET in ["sim", "tsim"]:
                sim_stats = simulator.stats()
                print("GEMM schedule:{} execution statistics:".format(name))
                for k, v in sim_stats.items():
                    print("\t{:<16}: {:>16}".format(k, v))

        def test_schedule1():
            # default schedule with no smt
            s = te.create_schedule(y.op)
            # set the scope of the SRAM buffers
            s[x_buf].set_scope(env.inp_scope)
            s[w_buf].set_scope(env.wgt_scope)
            s[y_gem].set_scope(env.acc_scope)
            s[y_shf].set_scope(env.acc_scope)
            s[y_max].set_scope(env.acc_scope)
            s[y_min].set_scope(env.acc_scope)
            # set pragmas for DMA transfer and ALU ops
            s[x_buf].compute_at(s[y_gem], ko)
            s[x_buf].pragma(s[x_buf].op.axis[0], env.dma_copy)
            s[w_buf].compute_at(s[y_gem], ko)
            s[w_buf].pragma(s[w_buf].op.axis[0], env.dma_copy)
            s[y_shf].pragma(s[y_shf].op.axis[0], env.alu)
            s[y_max].pragma(s[y_max].op.axis[0], env.alu)
            s[y_min].pragma(s[y_min].op.axis[0], env.alu)
            s[y].pragma(s[y].op.axis[0], env.dma_copy)
            # tensorization
            s[y_gem].reorder(
                ko,
                s[y_gem].op.axis[0],
                s[y_gem].op.axis[1],
                s[y_gem].op.axis[2],
                s[y_gem].op.axis[3],
                ki,
            )
            s[y_gem].tensorize(s[y_gem].op.axis[2], env.gemm)
            verify(s, name="default")

        def test_smt():
            # test smt schedule
            s = te.create_schedule(y.op)
            s[x_buf].set_scope(env.inp_scope)
            s[w_buf].set_scope(env.wgt_scope)
            s[y_gem].set_scope(env.acc_scope)
            s[y_shf].set_scope(env.acc_scope)
            s[y_max].set_scope(env.acc_scope)
            s[y_min].set_scope(env.acc_scope)
            abo, aco, abi, aci = s[y].op.axis
            abo1, abo2 = s[y].split(abo, nparts=2)
            s[y].bind(abo1, te.thread_axis("cthread"))
            s[y_gem].compute_at(s[y], abo1)
            s[y_shf].compute_at(s[y], abo1)
            s[y_max].compute_at(s[y], abo1)
            s[y_min].compute_at(s[y], abo1)
            s[y_gem].reorder(
                ko,
                s[y_gem].op.axis[0],
                s[y_gem].op.axis[1],
                s[y_gem].op.axis[2],
                s[y_gem].op.axis[3],
                ki,
            )
            s[y_gem].tensorize(s[y_gem].op.axis[2], env.gemm)
            s[y_shf].pragma(s[y_shf].op.axis[0], env.alu)
            s[y_max].pragma(s[y_max].op.axis[0], env.alu)
            s[y_min].pragma(s[y_min].op.axis[0], env.alu)
            s[x_buf].compute_at(s[y_gem], ko)
            s[x_buf].pragma(s[x_buf].op.axis[0], env.dma_copy)
            s[w_buf].compute_at(s[y_gem], ko)
            s[w_buf].pragma(s[w_buf].op.axis[0], env.dma_copy)
            s[y].pragma(abo2, env.dma_copy)
            verify(s, name="smt")

        test_schedule1()
        test_smt()

    vta.testing.run(_run)


def test_alu():
    def _run(env, remote):
        def check_alu(tvm_op, np_op=None, use_imm=False, test_name=None):
            """Test ALU"""
            m = 8
            n = 8
            imm = np.random.randint(1, 5)
            # compute
            a = te.placeholder((m, n, env.BATCH, env.BLOCK_OUT), name="a", dtype=env.acc_dtype)
            a_buf = te.compute(
                (m, n, env.BATCH, env.BLOCK_OUT), lambda *i: a(*i), "a_buf"
            )  # DRAM->SRAM
            if use_imm:
                res_buf = te.compute(
                    (m, n, env.BATCH, env.BLOCK_OUT), lambda *i: tvm_op(a_buf(*i), imm), "res_buf"
                )  # compute
            else:
                b = te.placeholder((m, n, env.BATCH, env.BLOCK_OUT), name="b", dtype=env.acc_dtype)
                b_buf = te.compute(
                    (m, n, env.BATCH, env.BLOCK_OUT), lambda *i: b(*i), "b_buf"
                )  # DRAM->SRAM
                res_buf = te.compute(
                    (m, n, env.BATCH, env.BLOCK_OUT),
                    lambda *i: tvm_op(a_buf(*i), b_buf(*i)),
                    "res_buf",
                )  # compute5B
            res = te.compute(
                (m, n, env.BATCH, env.BLOCK_OUT),
                lambda *i: res_buf(*i).astype(env.inp_dtype),
                "res",
            )  # SRAM->DRAM
            # schedule
            s = te.create_schedule(res.op)
            s[a_buf].set_scope(env.acc_scope)  # SRAM
            s[a_buf].pragma(a_buf.op.axis[0], env.dma_copy)  # DRAM->SRAM
            s[res_buf].set_scope(env.acc_scope)  # SRAM
            s[res_buf].pragma(res_buf.op.axis[0], env.alu)  # compute
            s[res].pragma(res.op.axis[0], env.dma_copy)  # SRAM->DRAM
            if not use_imm:
                s[b_buf].set_scope(env.acc_scope)  # SRAM
                s[b_buf].pragma(b_buf.op.axis[0], env.dma_copy)  # DRAM->SRAM

            if not remote:
                return

            # build
            with vta.build_config():
                if use_imm:
                    mod = vta.build(s, [a, res], tvm.target.Target("ext_dev", host=env.target_host))
                else:
                    mod = vta.build(
                        s, [a, b, res], tvm.target.Target("ext_dev", host=env.target_host)
                    )
            temp = utils.tempdir()
            mod.save(temp.relpath("load_act.o"))
            remote.upload(temp.relpath("load_act.o"))
            f = remote.load_module("load_act.o")
            # verify
            dev = remote.ext_dev(0)
            a_np = np.random.randint(-16, 16, size=(m, n, env.BATCH, env.BLOCK_OUT)).astype(a.dtype)
            if use_imm:
                res_np = np_op(a_np, imm) if np_op else tvm_op(a_np, imm)
            else:
                b_np = np.random.randint(-16, 16, size=(m, n, env.BATCH, env.BLOCK_OUT)).astype(
                    b.dtype
                )
                res_np = np_op(a_np, b_np) if np_op else tvm_op(a_np, b_np)
            res_np = res_np.astype(res.dtype)
            a_nd = tvm.nd.array(a_np, dev)
            res_nd = tvm.nd.array(np.zeros((m, n, env.BATCH, env.BLOCK_OUT)).astype(res.dtype), dev)

            if env.TARGET in ["sim", "tsim"]:
                simulator.clear_stats()

            if use_imm:
                f(a_nd, res_nd)
            else:
                b_nd = tvm.nd.array(b_np, dev)
                f(a_nd, b_nd, res_nd)

            np.testing.assert_equal(res_np, res_nd.numpy())

            if env.TARGET in ["sim", "tsim"]:
                sim_stats = simulator.stats()
                print("ALU {} execution statistics:".format(test_name))
                for k, v in sim_stats.items():
                    print("\t{:<16}: {:>16}".format(k, v))

        check_alu(lambda x, y: x << y, np.left_shift, use_imm=True, test_name="SHL")
        check_alu(tvm.te.max, np.maximum, use_imm=True, test_name="MAX")
        check_alu(tvm.te.max, np.maximum, test_name="MAX")
        check_alu(lambda x, y: x + y, use_imm=True, test_name="ADD")
        check_alu(lambda x, y: x + y, test_name="ADD")
        check_alu(lambda x, y: x >> y, np.right_shift, use_imm=True, test_name="SHR")

    vta.testing.run(_run)


def test_relu():
    """Test RELU on ALU"""

    def _run(env, remote):
        m = 8
        n = 10
        # compute
        a = te.placeholder((m, n, env.BATCH, env.BLOCK_OUT), name="a", dtype=env.acc_dtype)
        a_buf = te.compute(
            (m, n, env.BATCH, env.BLOCK_OUT), lambda *i: a(*i), "a_buf"
        )  # DRAM->SRAM
        max_buf = te.compute(
            (m, n, env.BATCH, env.BLOCK_OUT), lambda *i: tvm.te.max(a_buf(*i), 0), "res_buf"
        )  # relu
        min_buf = te.compute(
            (m, n, env.BATCH, env.BLOCK_OUT),
            lambda *i: tvm.te.min(max_buf(*i), (1 << (env.INP_WIDTH - 1)) - 1),
            "max_buf",
        )  # relu
        res = te.compute(
            (m, n, env.BATCH, env.BLOCK_OUT),
            lambda *i: min_buf(*i).astype(env.inp_dtype),
            "min_buf",
        )  # SRAM->DRAM
        # schedule
        s = te.create_schedule(res.op)
        s[a_buf].set_scope(env.acc_scope)  # SRAM
        s[a_buf].pragma(a_buf.op.axis[0], env.dma_copy)  # DRAM->SRAM
        s[max_buf].set_scope(env.acc_scope)  # SRAM
        s[min_buf].set_scope(env.acc_scope)  # SRAM
        s[max_buf].pragma(max_buf.op.axis[0], env.alu)  # compute
        s[min_buf].pragma(min_buf.op.axis[0], env.alu)  # compute
        s[res].pragma(res.op.axis[0], env.dma_copy)  # SRAM->DRAM
        # build
        with vta.build_config():
            mod = vta.build(s, [a, res], tvm.target.Target("ext_dev", host=env.target_host))
        if not remote:
            return
        temp = utils.tempdir()
        mod.save(temp.relpath("load_act.o"))
        remote.upload(temp.relpath("load_act.o"))
        f = remote.load_module("load_act.o")
        # verify
        dev = remote.ext_dev(0)
        a_np = np.random.randint(-256, 256, size=(m, n, env.BATCH, env.BLOCK_OUT)).astype(a.dtype)
        res_np = np.clip(a_np, 0, (1 << (env.INP_WIDTH - 1)) - 1).astype(res.dtype)
        a_nd = tvm.nd.array(a_np, dev)
        res_nd = tvm.nd.array(np.zeros((m, n, env.BATCH, env.BLOCK_OUT)).astype(res.dtype), dev)

        if env.TARGET in ["sim", "tsim"]:
            simulator.clear_stats()

        f(a_nd, res_nd)

        np.testing.assert_equal(res_np, res_nd.numpy())

        if env.TARGET in ["sim", "tsim"]:
            sim_stats = simulator.stats()
            print("Relu execution statistics:")
            for k, v in sim_stats.items():
                print("\t{:<16}: {:>16}".format(k, v))

    vta.testing.run(_run)


def test_shift_and_scale():
    """Test shift and scale on ALU"""

    def _run(env, remote):
        m = 2
        n = 8
        imm_shift = np.random.randint(0, 8)
        imm_scale = np.random.randint(1, 5)
        # compute
        a = te.placeholder((m, n, env.BATCH, env.BLOCK_OUT), name="a", dtype=env.acc_dtype)
        a_buf = te.compute(
            (m, n, env.BATCH, env.BLOCK_OUT), lambda *i: a(*i), "a_buf"
        )  # DRAM->SRAM
        res_shift = te.compute(
            (m, n, env.BATCH, env.BLOCK_OUT), lambda *i: a_buf(*i) + imm_shift, "res_shift"
        )  # compute
        res_scale = te.compute(
            (m, n, env.BATCH, env.BLOCK_OUT), lambda *i: res_shift(*i) >> imm_scale, "res_scale"
        )  # compute
        res = te.compute(
            (m, n, env.BATCH, env.BLOCK_OUT), lambda *i: res_scale(*i).astype(env.inp_dtype), "res"
        )  # SRAM->DRAM
        # schedule
        s = te.create_schedule(res.op)
        s[a_buf].set_scope(env.acc_scope)  # SRAM
        s[res_shift].set_scope(env.acc_scope)  # SRAM
        s[res_scale].set_scope(env.acc_scope)  # SRAM
        s[a_buf].pragma(a_buf.op.axis[0], env.dma_copy)  # DRAM->SRAM
        s[res_shift].pragma(res_shift.op.axis[0], env.alu)  # compute
        s[res_scale].pragma(res_scale.op.axis[0], env.alu)  # compute
        s[res].pragma(res.op.axis[0], env.dma_copy)  # SRAM->DRAM
        # build
        mod = vta.build(s, [a, res], tvm.target.Target("ext_dev", host=env.target_host))
        if not remote:
            return
        temp = utils.tempdir()
        mod.save(temp.relpath("load_act.o"))
        remote.upload(temp.relpath("load_act.o"))
        f = remote.load_module("load_act.o")
        # verify
        dev = remote.ext_dev(0)
        a_np = np.random.randint(-10, 10, size=(m, n, env.BATCH, env.BLOCK_OUT)).astype(a.dtype)
        res_np = np.right_shift((a_np + imm_shift), imm_scale)
        res_np = res_np.astype(res.dtype)
        a_nd = tvm.nd.array(a_np, dev)
        res_nd = tvm.nd.array(np.zeros((m, n, env.BATCH, env.BLOCK_OUT)).astype(res.dtype), dev)

        if env.TARGET in ["sim", "tsim"]:
            simulator.clear_stats()

        f(a_nd, res_nd)

        np.testing.assert_equal(res_np, res_nd.numpy())

        if env.TARGET in ["sim", "tsim"]:
            sim_stats = simulator.stats()
            print("Shift and scale execution statistics:")
            for k, v in sim_stats.items():
                print("\t{:<16}: {:>16}".format(k, v))

    vta.testing.run(_run)


def test_runtime_array():
    def _run(env, remote):
        n = 100
        dev = remote.ext_dev(0)
        x_np = np.random.randint(1, 10, size=(n, n, env.BATCH, env.BLOCK_OUT)).astype("int8")
        x_nd = tvm.nd.array(x_np, dev)
        np.testing.assert_equal(x_np, x_nd.numpy())

    vta.testing.run(_run)


if __name__ == "__main__":
    test_runtime_array()
    test_save_load_out()
    test_padded_load()
    test_gemm()
    test_alu()
    test_relu()
    test_shift_and_scale()
