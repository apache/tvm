# pylint: disable=invalid-name
"""Schedule for conv2d_hwcn with auto fusion"""
import tvm


def _schedule_conv2d_hwcn(op, sch):
    assert len(op.input_tensors) == 2
    Apad = op.input_tensors[0]
    W = op.input_tensors[1]
    B = op.output(0)

    sch[Apad].compute_inline()
    AA = sch.cache_read(Apad, "shared", [B])
    WW = sch.cache_read(W, "shared", [B])
    AL = sch.cache_read(AA, "local", [B])
    WL = sch.cache_read(WW, "local", [B])

    if op in sch.outputs:
        Out = op.output(0)
        BL = sch.cache_write(Out, "local")
    else:
        Out = sch.outputs[0].output(0)
        sch[B].set_scope("local")
        BL = B

    tile = 8
    num_thread = 8
    block_factor = tile * num_thread
    step = 8
    vthread = 2

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
    thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
    thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

    hi, wi, fi, ni = sch[Out].op.axis
    bz = sch[Out].fuse(hi, wi)
    by, fi = sch[Out].split(fi, factor=block_factor)
    bx, ni = sch[Out].split(ni, factor=block_factor)
    tyz, fi = sch[Out].split(fi, nparts=vthread)
    txz, ni = sch[Out].split(ni, nparts=vthread)
    ty, fi = sch[Out].split(fi, nparts=num_thread)
    tx, ni = sch[Out].split(ni, nparts=num_thread)
    sch[Out].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)
    sch[Out].bind(bz, block_z)
    sch[Out].bind(by, block_y)
    sch[Out].bind(bx, block_x)
    sch[Out].bind(tyz, thread_yz)
    sch[Out].bind(txz, thread_xz)
    sch[Out].bind(ty, thread_y)
    sch[Out].bind(tx, thread_x)

    # Schedule BL local write
    sch[BL].compute_at(sch[Out], tx)
    yi, xi, fi, ni = sch[BL].op.axis
    ry, rx, rc = sch[BL].op.reduce_axis
    rco, rci = sch[BL].split(rc, factor=step)
    sch[BL].reorder(rco, ry, rx, rci, fi, ni)
    fuse_index = sch[BL].fuse(ry, rx)
    fuse_index = sch[BL].fuse(fuse_index, rco)
    rx = fuse_index

    sch[AA].compute_at(sch[BL], rx)
    sch[WW].compute_at(sch[BL], rx)
    sch[AL].compute_at(sch[BL], rci)
    sch[WL].compute_at(sch[BL], rci)
    # Schedule for A's shared memory load
    yi, xi, ci, ni = sch[AA].op.axis
    ty, ci = sch[AA].split(ci, nparts=num_thread)
    tx, ni = sch[AA].split(ni, nparts=num_thread)
    _, ni = sch[AA].split(ni, factor=4)
    sch[AA].reorder(ty, tx, yi, xi, ci, ni)
    sch[AA].bind(ty, thread_y)
    sch[AA].bind(tx, thread_x)
    sch[AA].vectorize(ni)
    # Schedule for W's shared memory load
    yi, xi, ci, fi = sch[WW].op.axis
    ty, ci = sch[WW].split(ci, nparts=num_thread)
    tx, fi = sch[WW].split(fi, nparts=num_thread)
    _, fi = sch[WW].split(fi, factor=4)
    sch[WW].reorder(ty, tx, yi, xi, ci, fi)
    sch[WW].bind(ty, thread_y)
    sch[WW].bind(tx, thread_x)
    sch[WW].vectorize(fi)

    return sch


def schedule_conv2d_hwcn_map(op):
    """Schedule for conv2d_hwcn map ops.

    Parameters
    ----------
    op: tvm.tensor.Operation
        The symbolic description of the operation, should be conv2d_hwcn or
        conv2d_hwcn followed by a sequence of one-to-one-mapping operators.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    def traverse(operator):
        if operator.tag == 'ewise' or operator.tag == 'scale_shift':
            if operator not in sch.outputs:
                sch[operator].compute_inline()
            for tensor in operator.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        elif operator.tag == 'conv2d_hwcn':
            _schedule_conv2d_hwcn(operator, sch)
        else:
            raise RuntimeError("Unsupported operator: %s" % operator.tag)

    sch = tvm.create_schedule(op)
    traverse(op)
    return sch
