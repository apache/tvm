# pylint: disable=invalid-name
"""Schedule for conv2d_nchw with auto fusion, optimized for batch_size(n)=1."""
import tvm
import math


@tvm.register_func("topi.schedule.cuda.conv2d_nchw")
def schedule_conv2d_nchw(outs, target):
    """Schedule for conv2d_nchw.

    Parameters
    ----------
    outs: tvm.Array<tvm::Tensor>
        The computation graph description of conv2d_nchw in the format 
        of a list of tensors. 

    traget: str
        Compilation target ('cuda' for gpu)

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d_nchw.
    """
    s = tvm.create_schedule([x.op for x in outs])
    def schedule(temp, Filter, Output):
        out_height = tvm.ir_pass.Simplify(Output.shape[2]).value
        out_width = tvm.ir_pass.Simplify(Output.shape[3]).value
        channel_multiplier = tvm.ir_pass.Simplify(Filter.shape[1]).value

        block_h = out_width
        block_w = tvm.ir_pass.Simplify(temp.shape[1]).value
        if block_h % 48 == 0:
            block_h = 48
        elif block_h % 32 == 0:
            block_h = 32
        if block_w % 48 == 0:
            block_w = 48
        elif block_w % 32 == 0:
            block_w = 32

        s[temp].compute_inline()
        
        temp_S   = s.cache_read(temp, "shared", [Output])
        Filter_S = s.cache_read(Filter, "shared", [Output])
        
        if outs[0].op in s.outputs:
            Out = Output
            Out_L = s.cache_write(Out, "local")
        else:
            Out = outs[0].op.output(0)
            s[Output].set_scope("local")
            Out_L = Output

        # sheduler params
        tile = 8
        num_thread = 8
        step = 16
        vthread = 2
        out_filter = tvm.ir_pass.Simplify(Filter.shape[0]).value
        in_filter = tvm.ir_pass.Simplify(Filter.shape[1]).value
        opart2 = out_filter/8
        ofactor=out_filter
        wfactor=block_h
        ifactor=in_filter/4
        sfactor=max(1, ofactor/(opart2*2))
        spart = int(math.ceil(wfactor/vthread))

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
        oioc, iioc = s[Out].split(ioc, nparts = vthread)
        oiw, iiw = s[Out].split(iw, nparts=vthread)
        oiioc, iiioc = s[Out].split(iioc, nparts = opart2)
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
        oic, iic = s[temp_S].split(ic, factor=sfactor)
        ow, iw = s[temp_S].split(w, factor=spart)
        s[temp_S].bind(iic, thread_x)
        s[temp_S].bind(iw, thread_y)
        
        #schedule Filter_S shared mem load
        i, oc, h, w = s[Filter_S].op.axis
        ooc, ioc = s[Filter_S].split(oc, factor=sfactor)
        oi, ii = s[Filter_S].split(i, factor=spart)
        s[Filter_S].bind(ioc, thread_x)
        s[Filter_S].bind(ii, thread_y)
        
    def traverse(OP):
        # inline all one-to-one-mapping operators except the last stage (output)
        if 'ewise' in OP.tag or 'bcast' in OP.tag:
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if str(tensor.op.input_tensors) != str([]):
                    traverse(tensor.op)
        # schedule conv2d
        if 'conv2d_nchw' in OP.tag:
            temp = OP.input_tensors[0]
            Filter = OP.input_tensors[1]
            Output = OP.output(0)
            schedule(temp, Filter, Output)

    traverse(outs[0].op)
    return s
