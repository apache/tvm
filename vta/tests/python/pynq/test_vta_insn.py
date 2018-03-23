"""Unit test TPU's instructions """
import tvm
import vta
import mxnet as mx
import numpy as np
import topi
from tvm.contrib import rpc, util

host = "pynq"
port = 9091
target = "llvm -target=armv7-none-linux-gnueabihf"
out_dtype = "int%d" % vta.VTA_OUT_WIDTH
inp_dtype = "int%d" % vta.VTA_INP_WIDTH
wgt_dtype = "int%d" % vta.VTA_WGT_WIDTH
do_verify = True
print_ir = False


def test_save_load_out():
    """Test save/store output command"""
    n = 4
    x = tvm.placeholder(
        (n, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        name="x",
        dtype=out_dtype)
    x_buf = tvm.compute(
        (n, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: x(*i),
        "x_buf")
    # insert no-op that won't be optimized away
    y_buf = tvm.compute(
        (n, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: x_buf(*i)>>0,
        "y_buf")
    y = tvm.compute(
        (n, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: y_buf(*i).astype(inp_dtype),
        "y")
    # schedule
    s = tvm.create_schedule(y.op)
    s[x_buf].set_scope(vta.SCOPE_OUT)
    s[x_buf].pragma(x_buf.op.axis[0], vta.DMA_COPY)
    s[y_buf].set_scope(vta.SCOPE_OUT)
    s[y_buf].pragma(y_buf.op.axis[0], vta.ALU)
    s[y].pragma(y.op.axis[0], vta.DMA_COPY)

    def verify():
        # build
        m = tvm.build(s, [x, y], "ext_dev", target)
        temp = util.tempdir()
        remote = rpc.connect(host, port)
        m.save(temp.relpath("load_act.o"))
        remote.upload(temp.relpath("load_act.o"))
        f = remote.load_module("load_act.o")
        # verify
        ctx = remote.ext_dev(0)
        x_np = np.random.randint(
            1, 10, size=(n, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT)).astype(x.dtype)
        y_np = x_np.astype(y.dtype)
        x_nd = tvm.nd.array(x_np, ctx)
        y_nd = tvm.nd.empty(y_np.shape, ctx=ctx, dtype=y_np.dtype)
        f(x_nd, y_nd)
        np.testing.assert_equal(y_np, y_nd.asnumpy())
        print("\tFinished verification...")
    if do_verify:
        verify()

def test_padded_load():
    """Test padded load."""
    # declare
    n = 21
    m = 20
    pad_before = [0, 1, 0, 0]
    pad_after = [1, 3, 0, 0]
    x = tvm.placeholder(
        (n, m, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        name="x",
        dtype=out_dtype)
    x_buf = topi.nn.pad(x, pad_before, pad_after, name="y")
    # insert no-op that won't be optimized away
    y_buf = tvm.compute((n + pad_before[0] + pad_after[0],
                         m + pad_before[1] + pad_after[1],
                         vta.VTA_BATCH,
                         vta.VTA_BLOCK_OUT), lambda *i: x_buf(*i)>>0, "y_buf")
    y = tvm.compute((n + pad_before[0] + pad_after[0],
                     m + pad_before[1] + pad_after[1],
                     vta.VTA_BATCH,
                     vta.VTA_BLOCK_OUT), lambda *i: y_buf(*i).astype(inp_dtype), "y")
    # schedule
    s = tvm.create_schedule(y.op)
    s[x_buf].set_scope(vta.SCOPE_OUT)
    s[x_buf].pragma(x_buf.op.axis[0], vta.DMA_COPY)
    s[y_buf].set_scope(vta.SCOPE_OUT)
    s[y_buf].pragma(y_buf.op.axis[0], vta.ALU)
    s[y].pragma(y.op.axis[0], vta.DMA_COPY)

    def verify():
        # build
        mod = tvm.build(s, [x, y], "ext_dev", target)
        temp = util.tempdir()
        remote = rpc.connect(host, port)
        mod.save(temp.relpath("padded_load.o"))
        remote.upload(temp.relpath("padded_load.o"))
        f = remote.load_module("padded_load.o")
        # verify
        ctx = remote.ext_dev(0)
        x_np = np.random.randint(1, 2, size=(
            n, m, vta.VTA_BATCH, vta.VTA_BLOCK_OUT)).astype(x.dtype)
        y_np = np.zeros((n + pad_before[0] + pad_after[0],
                         m + pad_before[1] + pad_after[1],
                         vta.VTA_BATCH,
                         vta.VTA_BLOCK_OUT)).astype(y.dtype)
        y_np[pad_before[0]:pad_before[0] + n,
             pad_before[1]:pad_before[1] + m,
             :] = x_np
        x_nd = tvm.nd.array(x_np, ctx)
        y_nd = tvm.nd.empty(y_np.shape, ctx=ctx, dtype=y_np.dtype)
        f(x_nd, y_nd)
        np.testing.assert_equal(y_np, y_nd.asnumpy())
        print("\tFinished verification...")
    if print_ir:
        print(tvm.lower(s, [y, x], simple_mode=True))
    if do_verify:
        with tvm.build_config(add_lower_pass=vta.debug_mode(
                vta.DEBUG_DUMP_INSN)):
            verify()

def test_gemm():
    """Test GEMM."""
    # declare
    o = 4
    n = 4
    m = 4
    x = tvm.placeholder((o, n, vta.VTA_BATCH, vta.VTA_BLOCK_IN), name="x", dtype=inp_dtype)
    w = tvm.placeholder((m, n, vta.VTA_BLOCK_OUT, vta.VTA_BLOCK_IN), name="w", dtype=wgt_dtype)
    x_buf = tvm.compute((o, n, vta.VTA_BATCH, vta.VTA_BLOCK_IN), lambda *i: x(*i), "x_buf")
    w_buf = tvm.compute((m, n, vta.VTA_BLOCK_OUT, vta.VTA_BLOCK_IN), lambda *i: w(*i), "w_buf")
    ko = tvm.reduce_axis((0, n), name="ko")
    ki = tvm.reduce_axis((0, vta.VTA_BLOCK_IN), name="ki")
    y_gem = tvm.compute(
        (o, m, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda bo, co, bi, ci:
            tvm.sum(x_buf[bo, ko, bi, ki].astype(out_dtype) *
                    w_buf[co, ko, ci, ki].astype(out_dtype),
                    axis=[ko, ki]),
        name="y_gem")
    y_shf = tvm.compute(
        (o, m, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: y_gem(*i)>>8,
        name="y_shf")
    y_max = tvm.compute(
        (o, m, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: tvm.max(y_shf(*i), 0),
        "y_max") #relu
    y_min = tvm.compute(
        (o, m, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: tvm.min(y_max(*i), (1<<(vta.VTA_INP_WIDTH-1))-1),
        "y_min") #relu
    y = tvm.compute(
        (o, m, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: y_min(*i).astype(inp_dtype),
        name="y")

    def verify(s):
        mod = tvm.build(s, [x, w, y], "ext_dev", target)
        temp = util.tempdir()
        remote = rpc.connect(host, port)
        mod.save(temp.relpath("gemm.o"))
        remote.upload(temp.relpath("gemm.o"))
        f = remote.load_module("gemm.o")
        # verify
        ctx = remote.ext_dev(0)
        x_np = np.random.randint(
            -128, 128, size=(o, n, vta.VTA_BATCH, vta.VTA_BLOCK_IN)).astype(x.dtype)
        w_np = np.random.randint(
            -128, 128, size=(m, n, vta.VTA_BLOCK_OUT, vta.VTA_BLOCK_IN)).astype(w.dtype)
        y_np = np.zeros((o, m, vta.VTA_BATCH, vta.VTA_BLOCK_OUT)).astype(y.dtype)
        x_nd = tvm.nd.array(x_np, ctx)
        w_nd = tvm.nd.array(w_np, ctx)
        y_nd = tvm.nd.array(y_np, ctx)
        y_np = y_np.astype(out_dtype)
        for b in range(o):
            for i in range(m):
                for j in range(n):
                    y_np[b,i,:] += np.dot(x_np[b,j,:].astype(out_dtype),
                                          w_np[i,j].T.astype(out_dtype))
        y_np = np.right_shift(y_np, 8)
        y_np = np.clip(y_np, 0, (1<<(vta.VTA_INP_WIDTH-1))-1).astype(y.dtype)
        f(x_nd, w_nd, y_nd)
        np.testing.assert_equal(y_np, y_nd.asnumpy())
        print("\tFinished verification...")

    def test_schedule1():
        # default schedule with no smt
        s = tvm.create_schedule(y.op)
        # set the scope of the SRAM buffers
        s[x_buf].set_scope(vta.SCOPE_INP)
        s[w_buf].set_scope(vta.SCOPE_WGT)
        s[y_gem].set_scope(vta.SCOPE_OUT)
        s[y_shf].set_scope(vta.SCOPE_OUT)
        s[y_max].set_scope(vta.SCOPE_OUT)
        s[y_min].set_scope(vta.SCOPE_OUT)
        # set pragmas for DMA transfer and ALU ops
        s[x_buf].pragma(s[x_buf].op.axis[0], vta.DMA_COPY)
        s[w_buf].pragma(s[w_buf].op.axis[0], vta.DMA_COPY)
        s[y_shf].pragma(s[y_shf].op.axis[0], vta.ALU)
        s[y_max].pragma(s[y_max].op.axis[0], vta.ALU)
        s[y_min].pragma(s[y_min].op.axis[0], vta.ALU)
        s[y].pragma(s[y].op.axis[0], vta.DMA_COPY)
        # tensorization
        s[y_gem].reorder(
            ko,
            s[y_gem].op.axis[0],
            s[y_gem].op.axis[1],
            s[y_gem].op.axis[2],
            s[y_gem].op.axis[3],
            ki)
        s[y_gem].tensorize(s[y_gem].op.axis[2], vta.GEMM)
        if print_ir:
            print(tvm.lower(s, [x, w, y], simple_mode=True))
        if do_verify:
            with tvm.build_config(
                    add_lower_pass=vta.debug_mode(vta.DEBUG_DUMP_INSN)):
                verify(s)

    def test_smt():
        # test smt schedule
        s = tvm.create_schedule(y.op)
        s[x_buf].set_scope(vta.SCOPE_INP)
        s[w_buf].set_scope(vta.SCOPE_WGT)
        s[y_gem].set_scope(vta.SCOPE_OUT)
        s[y_shf].set_scope(vta.SCOPE_OUT)
        s[y_max].set_scope(vta.SCOPE_OUT)
        s[y_min].set_scope(vta.SCOPE_OUT)
        abo, aco, abi, aci = s[y].op.axis
        abo1, abo2 = s[y].split(abo, nparts=2)
        s[y].bind(abo1, tvm.thread_axis("cthread"))
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
            ki)
        s[y_gem].tensorize(s[y_gem].op.axis[2], vta.GEMM)
        s[y_shf].pragma(s[y_shf].op.axis[0], vta.ALU)
        s[y_max].pragma(s[y_max].op.axis[0], vta.ALU)
        s[y_min].pragma(s[y_min].op.axis[0], vta.ALU)
        s[x_buf].compute_at(s[y_gem], ko)
        s[x_buf].pragma(s[x_buf].op.axis[0], vta.DMA_COPY)
        s[w_buf].compute_at(s[y_gem], ko)
        s[w_buf].pragma(s[w_buf].op.axis[0], vta.DMA_COPY)
        s[y].pragma(abo2, vta.DMA_COPY)
        if print_ir:
            print(tvm.lower(s, [x, y, w], simple_mode=True))
        if do_verify:
            with tvm.build_config(
                    add_lower_pass=vta.debug_mode(vta.DEBUG_DUMP_INSN)):
                verify(s)

    test_schedule1()
    test_smt()

def test_alu(tvm_op, np_op=None, use_imm=False):
    """Test ALU"""
    m = 8
    n = 8
    imm = np.random.randint(1,5)
    # compute
    a = tvm.placeholder(
        (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        name="a",
        dtype=out_dtype)
    a_buf = tvm.compute(
        (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: a(*i),
        "a_buf") #DRAM->SRAM
    if use_imm:
        res_buf = tvm.compute(
            (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
            lambda *i: tvm_op(a_buf(*i), imm),
            "res_buf") #compute
    else:
        b = tvm.placeholder(
            (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
            name="b",
            dtype=out_dtype)
        b_buf = tvm.compute(
            (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
            lambda *i: b(*i),
            "b_buf") #DRAM->SRAM
        res_buf = tvm.compute(
            (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
            lambda *i: tvm_op(a_buf(*i), b_buf(*i)),
            "res_buf") #compute
    res = tvm.compute(
        (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: res_buf(*i).astype(inp_dtype),
        "res") #SRAM->DRAM
    # schedule
    s = tvm.create_schedule(res.op)
    s[a_buf].set_scope(vta.SCOPE_OUT) # SRAM
    s[a_buf].pragma(a_buf.op.axis[0], vta.DMA_COPY) # DRAM->SRAM
    s[res_buf].set_scope(vta.SCOPE_OUT) # SRAM
    s[res_buf].pragma(res_buf.op.axis[0], vta.ALU) # compute
    s[res].pragma(res.op.axis[0], vta.DMA_COPY) # SRAM->DRAM
    if use_imm:
        if print_ir:
            print(tvm.lower(s, [a, res], simple_mode=True))
    else:
        s[b_buf].set_scope(vta.SCOPE_OUT) # SRAM
        s[b_buf].pragma(b_buf.op.axis[0], vta.DMA_COPY) # DRAM->SRAM
        if print_ir:
            print(tvm.lower(s, [a, b, res], simple_mode=True))

    def verify():
        # build
        if use_imm:
            mod = tvm.build(s, [a, res], "ext_dev", target)
        else:
            mod = tvm.build(s, [a, b, res], "ext_dev", target)
        temp = util.tempdir()
        remote = rpc.connect(host, port)
        mod.save(temp.relpath("load_act.o"))
        remote.upload(temp.relpath("load_act.o"))
        f = remote.load_module("load_act.o")
        # verify
        ctx = remote.ext_dev(0)
        a_np = np.random.randint(
            -16, 16, size=(m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT)).astype(a.dtype)
        if use_imm:
            res_np = np_op(a_np, imm) if np_op else tvm_op(a_np, imm)
        else:
            b_np = np.random.randint(
                -16, 16, size=(m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT)).astype(b.dtype)
            res_np = np_op(a_np, b_np) if np_op else tvm_op(a_np, b_np)
        res_np = res_np.astype(res.dtype)
        a_nd = tvm.nd.array(a_np, ctx)
        res_nd = tvm.nd.array(
            np.zeros((m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT)).astype(res.dtype), ctx)
        if use_imm:
            f(a_nd, res_nd)
        else:
            b_nd = tvm.nd.array(b_np, ctx)
            f(a_nd, b_nd, res_nd)
        np.testing.assert_equal(res_np, res_nd.asnumpy())
        print("\tFinished verification...")

    if do_verify:
        verify()

def test_relu():
    """Test RELU on ALU"""
    m = 8
    n = 8
    # compute
    a = tvm.placeholder(
        (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        name="a",
        dtype=out_dtype)
    a_buf = tvm.compute(
        (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: a(*i),
        "a_buf") # DRAM->SRAM
    max_buf = tvm.compute(
        (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: tvm.max(a_buf(*i), 0),
        "res_buf") # relu
    min_buf = tvm.compute(
        (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: tvm.min(max_buf(*i), (1<<(vta.VTA_INP_WIDTH-1))-1),
        "max_buf") # relu
    res = tvm.compute(
        (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: min_buf(*i).astype(inp_dtype),
        "min_buf") # SRAM->DRAM
    # schedule
    s = tvm.create_schedule(res.op)
    s[a_buf].set_scope(vta.SCOPE_OUT) # SRAM
    s[a_buf].pragma(a_buf.op.axis[0], vta.DMA_COPY) # DRAM->SRAM
    s[max_buf].set_scope(vta.SCOPE_OUT) # SRAM
    s[min_buf].set_scope(vta.SCOPE_OUT) # SRAM
    s[max_buf].pragma(max_buf.op.axis[0], vta.ALU) # compute
    s[min_buf].pragma(min_buf.op.axis[0], vta.ALU) # compute
    s[res].pragma(res.op.axis[0], vta.DMA_COPY) # SRAM->DRAM
    if print_ir:
        print(tvm.lower(s, [a, res], simple_mode=True))

    def verify():
        # build
        mod = tvm.build(s, [a, res], "ext_dev", target)
        temp = util.tempdir()
        remote = rpc.connect(host, port)
        mod.save(temp.relpath("load_act.o"))
        remote.upload(temp.relpath("load_act.o"))
        f = remote.load_module("load_act.o")
        # verify
        ctx = remote.ext_dev(0)
        a_np = np.random.randint(
            -256, 256, size=(m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT)).astype(a.dtype)
        res_np = np.clip(a_np, 0, (1<<(vta.VTA_INP_WIDTH-1))-1).astype(res.dtype)
        a_nd = tvm.nd.array(a_np, ctx)
        res_nd = tvm.nd.array(
            np.zeros((m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT)).astype(res.dtype), ctx)
        f(a_nd, res_nd)
        np.testing.assert_equal(res_np, res_nd.asnumpy())
        print("\tFinished verification...")

    if do_verify:
        verify()

def test_shift_and_scale():
    """Test shift and scale on ALU"""
    m = 8
    n = 8
    imm_shift = np.random.randint(-10,10)
    imm_scale = np.random.randint(1,5)
    # compute
    a = tvm.placeholder(
        (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        name="a", dtype=out_dtype)
    a_buf = tvm.compute(
        (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: a(*i),
        "a_buf") # DRAM->SRAM
    res_shift = tvm.compute(
        (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: a_buf(*i)+imm_shift,
        "res_shift") # compute
    res_scale = tvm.compute(
        (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: res_shift(*i)>>imm_scale,
        "res_scale") # compute
    res = tvm.compute(
        (m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT),
        lambda *i: res_scale(*i).astype(inp_dtype),
        "res") # SRAM->DRAM
    # schedule
    s = tvm.create_schedule(res.op)
    s[a_buf].set_scope(vta.SCOPE_OUT) # SRAM
    s[res_shift].set_scope(vta.SCOPE_OUT) # SRAM
    s[res_scale].set_scope(vta.SCOPE_OUT) # SRAM
    s[a_buf].pragma(a_buf.op.axis[0], vta.DMA_COPY) # DRAM->SRAM
    s[res_shift].pragma(res_shift.op.axis[0], vta.ALU) # compute
    s[res_scale].pragma(res_scale.op.axis[0], vta.ALU) # compute
    s[res].pragma(res.op.axis[0], vta.DMA_COPY) # SRAM->DRAM
    if print_ir:
        print(tvm.lower(s, [a, res], simple_mode=True))

    def verify():
        # build
        mod = tvm.build(s, [a, res], "ext_dev", target)
        temp = util.tempdir()
        remote = rpc.connect(host, port)
        mod.save(temp.relpath("load_act.o"))
        remote.upload(temp.relpath("load_act.o"))
        f = remote.load_module("load_act.o")
        # verify
        ctx = remote.ext_dev(0)
        a_np = np.random.randint(
            -10, 10, size=(m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT)).astype(a.dtype)
        res_np = np.right_shift((a_np + imm_shift), imm_scale)
        res_np = res_np.astype(res.dtype)
        a_nd = tvm.nd.array(a_np, ctx)
        res_nd = tvm.nd.array(
            np.zeros((m, n, vta.VTA_BATCH, vta.VTA_BLOCK_OUT)).astype(res.dtype), ctx)
        f(a_nd, res_nd)
        np.testing.assert_equal(res_np, res_nd.asnumpy())
        print("\tFinished verification...")

    if do_verify:
        verify()

if __name__ == "__main__":
    print("Padded load test")
    test_padded_load()
    print("Load/store test")
    test_save_load_out()
    print("GEMM test")
    test_gemm()
    print("Max immediate")
    test_alu(tvm.max, np.maximum, use_imm=True)
    print("Max")
    test_alu(tvm.max, np.maximum)
    print("Add immediate")
    test_alu(lambda x, y: x + y, use_imm=True)
    print("Add")
    test_alu(lambda x, y: x + y)
    print("Shift right immediate")
    test_alu(lambda x, y: x >> y, np.right_shift, use_imm=True)
    print("Relu")
    test_relu()
    # print("Shift and scale")
    # test_shift_and_scale()
