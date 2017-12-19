# pylint: disable=invalid-name,unused-variable,invalid-name
"""Conv2D schedule on x86"""
import tvm
from .. import generic
from .. import tag

@generic.schedule_conv2d_nchw.register(["cpu"])
def schedule_conv2d(outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            else: # inject custom schedule
                if len(op.axis) == 4:
                    n, c, h, w = op.axis
                    fused = s[op].fuse(n, c)
                    s[op].parallel(fused)
                    s[op].vectorize(w)
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        if 'conv2d_nchw' in op.tag:
            conv = op.output(0)
            kernel = op.input_tensors[1]
            data = op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            n_pad, c_pad, h_pad, w_pad = data_pad.op.axis
            pad_fused = s[data_pad].fuse(n_pad, c_pad)
            s[data_pad].parallel(pad_fused)
            C = conv
            n, c, h, w = C.op.axis
            rc, ry, rx = C.op.reduce_axis
            fused = s[C].fuse(n, c)
            s[C].parallel(fused)
            wo, wi = s[C].split(w, factor=16)
            #ho, hi = s[C].split(h, factor=4)
            s[C].reorder(fused, h,  wo, rc, ry, rx, wi)
            s[C].unroll(rx)
            s[C].unroll(ry)
            s[C].vectorize(wi)

    traverse(outs[0].op)
    return s
