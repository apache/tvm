#pylint: disable=invalid-name, no-member, too-many-locals, too-many-statements, too-many-arguments
"""Schedule for conv2d_nchw with auto fusion"""
import tvm
from .. import util

def conv2d_224_3_64(s, temp_S, Filter_S, Out, Out_L):
    """Schedule conv2d for specific feature_in_out_filter pattern"""
    # scheduler params
    ofactor = 16
    hfactor = 2
    ow_size = util.get_const_int(Out.shape[3])
    num_thread = ow_size*hfactor
    vthread = hfactor
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")

    i, oc, h, w = s[Out].op.axis
    ooc, ioc = s[Out].split(oc, factor=ofactor)
    oh, ih = s[Out].split(h, factor=hfactor)
    s[Out].reorder(ooc, oh, ioc, ih, w)
    oc = s[Out].fuse(ooc, oh)
    w = s[Out].fuse(w, ih)

    s[Out].bind(w, thread_x)
    s[Out].bind(ioc, thread_xz)
    s[Out].bind(oc, block_x)

    s[Out_L].compute_at(s[Out], w)

    # schedule Out_L local write
    i, oc, h, w = s[Out_L].op.axis
    ic, dh, dw = s[Out_L].op.reduce_axis
    s[Out_L].reorder(i, oc, h, w, ic, dh, dw)
    s[temp_S].compute_at(s[Out_L], ic)
    s[Filter_S].compute_at(s[Out_L], w)

    #schedule temp_S shared mem load
    i, ic, h, w = s[temp_S].op.axis
    tx, ih = s[temp_S].split(w, nparts=num_thread)
    s[temp_S].bind(tx, thread_x)

    #schedule Filter_S shared mem load
    i, oc, h, w = s[Filter_S].op.axis
    fuse_index = s[Filter_S].fuse(w, h)
    w = s[Filter_S].fuse(fuse_index, oc)
    tx, _ = s[Filter_S].split(w, nparts=num_thread)
    s[Filter_S].bind(tx, thread_x)

def conv2d_56_64_128(s, temp_S, Filter_S, Out, Out_L, flag):
    """Schedule conv2d for specific feature_in_out_filter pattern"""
    if util.get_const_int(Filter_S.shape[0]) == util.get_const_int(Filter_S.shape[1]):
        num_thread_x = 8
        num_thread_y = 8
        vthread_x = 7
        vthread_y = 2
        ifactor = 8

        block_x = tvm.thread_axis("blockIdx.x")
        block_y = tvm.thread_axis("blockIdx.y")
        thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
        thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")
        thread_xz = tvm.thread_axis((0, vthread_x), "vthread", name="vx")
        thread_yz = tvm.thread_axis((0, vthread_y), "vthread", name="vy")

        i, oc, h, w = s[Out].op.axis
        w = s[Out].fuse(h, w)
        ow, iw = s[Out].split(w, factor=num_thread_x*vthread_x)
        ooc, ioc = s[Out].split(oc, factor=num_thread_y*vthread_y)
        oiw, iiw = s[Out].split(iw, nparts=vthread_x)
        oioc, iioc = s[Out].split(ioc, nparts=vthread_y)
        s[Out].reorder(i, ooc, ow, oioc, oiw, iioc, iiw)
        s[Out].bind(iiw, thread_x)
        s[Out].bind(iioc, thread_y)
        s[Out].bind(oiw, thread_xz)
        s[Out].bind(oioc, thread_yz)
        s[Out].bind(ow, block_x)
        s[Out].bind(ooc, block_y)

        s[Out_L].compute_at(s[Out], iiw)

        # schedule Out_L local write
        i, oc, h, w = s[Out_L].op.axis
        ic, dh, dw = s[Out_L].op.reduce_axis
        oic, iic = s[Out_L].split(ic, factor=ifactor)
        s[Out_L].reorder(oic, dh, dw, iic, h, w)

        s[temp_S].compute_at(s[Out_L], oic)
        s[Filter_S].compute_at(s[Out_L], dw)

        #schedule temp_S shared mem load
        i, ic, h, w = s[temp_S].op.axis
        _, iic = s[temp_S].split(ic, factor=num_thread_y)
        w = s[temp_S].fuse(h, w)
        _, iw = s[temp_S].split(w, factor=num_thread_x)
        s[temp_S].bind(iic, thread_y)
        s[temp_S].bind(iw, thread_x)

        i, oc, h, w = s[Filter_S].op.axis
        _, ioc = s[Filter_S].split(oc, factor=num_thread_y)
        _, ii = s[Filter_S].split(i, factor=num_thread_x)
        s[Filter_S].bind(ioc, thread_y)
        s[Filter_S].bind(ii, thread_x)
    else:
        # scheduler params
        num_thread = 8
        vthread = 2
        opart2 = 4
        ofactor = 64
        wfactor = 28
        ifactor = 8
        if flag > 256:
            wfactor = 14
        sfactor = max(1, ofactor//(opart2*2))
        spart = max(1, (wfactor + vthread-1) // vthread)
        block_x = tvm.thread_axis("blockIdx.x")
        block_y = tvm.thread_axis("blockIdx.y")
        block_z = tvm.thread_axis("blockIdx.z")
        thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
        thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
        thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
        thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

        i, oc, h, w = s[Out].op.axis
        ooc, ioc = s[Out].split(oc, factor=ofactor)
        ow, iw = s[Out].split(w, factor=wfactor)
        ow = s[Out].fuse(ow, h)
        oioc, iioc = s[Out].split(ioc, nparts=vthread)
        oiw, iiw = s[Out].split(iw, nparts=vthread)
        oiioc, iiioc = s[Out].split(iioc, nparts=opart2)
        s[Out].reorder(i, ooc, ow, oioc, oiw, oiioc, iiw, iiioc)
        s[Out].bind(iiioc, thread_x)
        s[Out].bind(iiw, thread_y)
        s[Out].bind(oiioc, thread_xz)
        s[Out].bind(oiw, thread_yz)
        s[Out].bind(oioc, block_x)
        s[Out].bind(ow, block_y)
        s[Out].bind(ooc, block_z)

        s[Out_L].compute_at(s[Out], iiioc)

        if util.get_const_int(Filter_S.shape[1]) == 128:
            # schedule Out_L local write
            i, oc, h, w = s[Out_L].op.axis
            ic, dh, dw = s[Out_L].op.reduce_axis
            oic, iic = s[Out_L].split(ic, factor=ifactor)
            s[Out_L].reorder(oic, dh, dw, iic, h, w)
            oic = s[Out_L].fuse(dh, oic)
            s[temp_S].compute_at(s[Out_L], oic)
            s[Filter_S].compute_at(s[Out_L], oic)

            #schedule temp_S shared mem load
            i, ic, h, w = s[temp_S].op.axis
            _, iic = s[temp_S].split(ic, factor=sfactor)
            _, iw = s[temp_S].split(w, factor=spart)
            s[temp_S].bind(iic, thread_x)
            s[temp_S].bind(iw, thread_y)

            #schedule Filter_S shared mem load
            i, oc, h, w = s[Filter_S].op.axis
            _, ioc = s[Filter_S].split(oc, factor=sfactor)
            _, ii = s[Filter_S].split(i, factor=spart)
            s[Filter_S].bind(ioc, thread_x)
            s[Filter_S].bind(ii, thread_y)
        else:
            # schedule Out_L local write
            i, oc, h, w = s[Out_L].op.axis
            ic, dh, dw = s[Out_L].op.reduce_axis
            oic, iic = s[Out_L].split(ic, factor=ifactor)
            s[Out_L].reorder(oic, dh, dw, iic, h, w)

            s[temp_S].compute_at(s[Out_L], oic)
            s[Filter_S].compute_at(s[Out_L], dw)

            #schedule temp_S shared mem load
            i, ic, h, w = s[temp_S].op.axis
            _, iic = s[temp_S].split(ic, factor=sfactor)
            _, iw = s[temp_S].split(w, factor=spart)
            s[temp_S].bind(iic, thread_x)
            s[temp_S].bind(iw, thread_y)

            #schedule Filter_S shared mem load
            i, oc, h, w = s[Filter_S].op.axis
            _, ioc = s[Filter_S].split(oc, factor=sfactor)
            _, ii = s[Filter_S].split(i, factor=spart)
            s[Filter_S].bind(ioc, thread_x)
            s[Filter_S].bind(ii, thread_y)

def conv2d_14_256_256(s, Filter, temp_S, Filter_S, Out, Out_L):
    """Schedule conv2d for specific feature_in_out_filter pattern"""
    if util.get_const_int(Filter.shape[1]) == 256:
        # scheduler params
        vthread_x = util.get_const_int(Out.shape[3])
        num_thread_x = 64
        ofactor = 8
        if util.get_const_int(Filter.shape[3]) == 1:
            ofactor = 64
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
        thread_xz = tvm.thread_axis((0, vthread_x), "vthread", name="vx")

        i, oc, h, w = s[Out].op.axis
        ooc, ioc = s[Out].split(oc, factor=num_thread_x)
        s[Out].reorder(i, ooc, h, w, ioc)
        ooc = s[Out].fuse(h, ooc)
        s[Out].bind(ioc, thread_x)
        s[Out].bind(w, thread_xz)
        s[Out].bind(ooc, block_x)

        s[Out_L].compute_at(s[Out], ioc)

        # schedule Out_L local write
        i, oc, h, w = s[Out_L].op.axis
        ic, dh, dw = s[Out_L].op.reduce_axis
        oic, iic = s[Out_L].split(ic, ofactor)
        s[Out_L].reorder(oic, dh, dw, iic, h, w)

        s[temp_S].compute_at(s[Out_L], oic)
        s[Filter_S].compute_at(s[Out_L], oic)

        #schedule temp_S shared mem load
        i, ic, h, w = s[temp_S].op.axis
        s[temp_S].reorder(i, ic, w, h)
        ic = s[temp_S].fuse(w, ic)
        _, iic = s[temp_S].split(ic, factor=num_thread_x)
        s[temp_S].bind(iic, thread_x)

        #schedule Filter_S shared mem load
        i, oc, h, w = s[Filter_S].op.axis
        _, ii = s[Filter_S].split(i, factor=num_thread_x)
        s[Filter_S].bind(ii, thread_x)
        s[Filter_S].storage_align(s[Filter_S].op.axis[0], 2, 1)

    else:
        # scheduler params
        vthread_x = util.get_const_int(Out.shape[2])
        num_thread_x = 16
        num_thread_y = util.get_const_int(Out.shape[3])
        ofactor = 8
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
        thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")
        thread_xz = tvm.thread_axis((0, vthread_x), "vthread", name="vx")

        i, oc, h, w = s[Out].op.axis
        ooc, ioc = s[Out].split(oc, factor=num_thread_x)
        s[Out].reorder(i, ooc, h, w, ioc)
        s[Out].bind(ioc, thread_x)
        s[Out].bind(w, thread_y)
        s[Out].bind(h, thread_xz)
        s[Out].bind(ooc, block_x)

        s[Out_L].compute_at(s[Out], ioc)

        # schedule Out_L local write
        i, oc, h, w = s[Out_L].op.axis
        ic, dh, dw = s[Out_L].op.reduce_axis
        oic, iic = s[Out_L].split(ic, ofactor)
        s[Out_L].reorder(oic, dh, dw, iic, h, w)

        s[temp_S].compute_at(s[Out_L], oic)
        s[Filter_S].compute_at(s[Out_L], oic)

        #schedule temp_S shared mem load
        i, ic, h, w = s[temp_S].op.axis
        ic = s[temp_S].fuse(w, h, ic)
        oic, iic = s[temp_S].split(ic, factor=num_thread_x)
        _, ioic = s[temp_S].split(oic, factor=num_thread_y)
        s[temp_S].bind(iic, thread_x)
        s[temp_S].bind(ioic, thread_y)

        #schedule Filter_S shared mem load
        i, oc, h, w = s[Filter_S].op.axis
        _, ii = s[Filter_S].split(i, factor=num_thread_x)
        h = s[Filter_S].fuse(h, w)
        _, ih = s[Filter_S].split(h, factor=num_thread_y)
        s[Filter_S].bind(ii, thread_x)
        s[Filter_S].bind(ih, thread_y)
        s[Filter_S].storage_align(s[Filter_S].op.axis[0], 2, 1)

def conv2d_56_64_64(s, Filter, temp_S, Filter_S, Out, Out_L):
    """Schedule conv2d for specific feature_in_out_filter pattern"""
    # scheduler params
    num_thread = 8
    vthread = 2
    opart2 = 4
    ofactor = 64
    wfactor = 56
    ifactor = 8
    if util.get_const_int(Filter.shape[0]) == 64:
        opart2 = 8
        ifactor = 16
    sfactor = max(1, ofactor//(opart2*2))
    spart = max(1, (wfactor + vthread-1) // vthread)

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
    thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
    thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

    i, oc, h, w = s[Out].op.axis
    ooc, ioc = s[Out].split(oc, factor=ofactor)
    ow, iw = s[Out].split(w, factor=wfactor)
    ow = s[Out].fuse(ow, h)
    oioc, iioc = s[Out].split(ioc, nparts=vthread)
    oiw, iiw = s[Out].split(iw, nparts=vthread)
    oiioc, iiioc = s[Out].split(iioc, nparts=opart2)
    s[Out].reorder(i, ooc, ow, oioc, oiw, oiioc, iiw, iiioc)
    s[Out].bind(iiioc, thread_x)
    s[Out].bind(iiw, thread_y)
    s[Out].bind(oiioc, thread_xz)
    s[Out].bind(oiw, thread_yz)
    s[Out].bind(oioc, block_x)
    s[Out].bind(ow, block_y)
    s[Out].bind(ooc, block_z)

    s[Out_L].compute_at(s[Out], iiioc)

    # schedule Out_L local write
    i, oc, h, w = s[Out_L].op.axis
    ic, dh, dw = s[Out_L].op.reduce_axis
    oic, iic = s[Out_L].split(ic, factor=ifactor)
    s[Out_L].reorder(oic, dh, dw, iic, h, w)

    fuse_index = s[Out_L].fuse(dw, dh)
    fuse_index = s[Out_L].fuse(fuse_index, oic)
    dw = fuse_index
    s[temp_S].compute_at(s[Out_L], dw)
    s[Filter_S].compute_at(s[Out_L], dw)

    #schedule temp_S shared mem load
    i, ic, h, w = s[temp_S].op.axis
    _, iic = s[temp_S].split(ic, factor=sfactor)
    _, iw = s[temp_S].split(w, factor=spart)
    s[temp_S].bind(iic, thread_x)
    s[temp_S].bind(iw, thread_y)

    #schedule Filter_S shared mem load
    i, oc, h, w = s[Filter_S].op.axis
    _, ioc = s[Filter_S].split(oc, factor=sfactor)
    _, ii = s[Filter_S].split(i, factor=spart)
    s[Filter_S].bind(ioc, thread_x)
    s[Filter_S].bind(ii, thread_y)

def schedule_conv2d_small_batch(outs):
    """Create schedule for tensors or return error if batch size is larager than 1"""
    s = tvm.create_schedule([x.op for x in outs])

    def schedule(temp, Filter, Output):
        """Schedule conv2d_nchw"""
        block_h = util.get_const_int(Output.shape[3])
        block_w = util.get_const_int(temp.shape[1])
        if block_h % 48 == 0:
            block_h = 48
        elif block_h % 32 == 0:
            block_h = 32
        if block_w % 48 == 0:
            block_w = 48
        elif block_w % 32 == 0:
            block_w = 32

        s[temp].compute_inline()

        temp_S = s.cache_read(temp, "shared", [Output])
        Filter_S = s.cache_read(Filter, "shared", [Output])

        if Output.op in s.outputs:
            Out = Output
            Out_L = s.cache_write(Out, "local")
        else:
            Out = outs[0].op.output(0)
            s[Output].set_scope("local")
            Out_L = Output

        flag = util.get_const_int(Filter.shape[0])+util.get_const_int(Filter.shape[1])

        if util.get_const_int(Filter.shape[3]) == 7:
            conv2d_224_3_64(s, temp_S, Filter_S, Out, Out_L)
        elif 128 < flag < 512:
            conv2d_56_64_128(s, temp_S, Filter_S, Out, Out_L, flag)
        elif flag >= 512:
            conv2d_14_256_256(s, Filter, temp_S, Filter_S, Out, Out_L)
        else:
            conv2d_56_64_64(s, Filter, temp_S, Filter_S, Out, Out_L)

    def traverse(OP):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if 'ewise' in OP.tag or 'bcast' in OP.tag:
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule conv2d
        if 'conv2d_nchw' in OP.tag:
            temp = OP.input_tensors[0]
            Filter = OP.input_tensors[1]
            Output = OP.output(0)
            schedule(temp, Filter, Output)

    traverse(outs[0].op)
    return s

def schedule_conv2d_nchw(outs):
    """Schedule for conv2d_nchw.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_nchw
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d_nchw.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    batch_size = util.get_const_int(outs[0].op.output(0).shape[0])
    if batch_size > 1:
        raise RuntimeError("Batch size: %d is too large for this schedule" % batch_size)
    return  schedule_conv2d_small_batch(outs)
