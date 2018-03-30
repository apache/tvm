#pylint: disable=invalid-name, line-too-long
"""Schedule for conv2d_transpose_nchw with auto fusion"""
import tvm
from .. import util
from .. import tag
from .. import generic
from .conv2d_nchw import conv2d_224_3_64, conv2d_56_64_128, conv2d_14_256_256, conv2d_56_64_64


def schedule_conv2d_transpose_small_batch(outs):
    """Create schedule for tensors or return error if batch size is larger than 1"""
    s = tvm.create_schedule([x.op for x in outs])

    def schedule(temp, Filter, Output):
        """Schedule conv2d_transpose_nchw"""
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

        flag = util.get_const_int(Filter.shape[0])+util.get_const_int(Filter.shape[1])

        if flag > 768:
            temp_G = s.cache_read(temp, "global", [Output])
            s[temp_G].compute_inline()
            i, ic, h, w = s[temp_G].op.axis
            oic, iic = s[temp_G].split(ic, factor=4)
            s[temp_G].reorder(i, h, w, oic, iic)
            temp_R = s.cache_write(temp_G, "global")
            temp_S = s.cache_read(temp_R, "shared", [temp_G])
        elif 128 < flag < 512:
            temp_G = s.cache_read(temp, "global", [Output])
            s[temp_G].compute_inline()
            i, ic, h, w = s[temp_G].op.axis
            oic, iic = s[temp_G].split(ic, factor=4)
            s[temp_G].reorder(i, oic, h, w, iic)
            temp_R = s.cache_write(temp_G, "global")
            temp_S = s.cache_read(temp_R, "shared", [temp_G])
        elif util.get_const_int(Filter.shape[3]) == 7 or (util.get_const_int(Output.shape[2] == 224) and flag < 128):
            temp_G = s.cache_read(temp, "global", [Output])
            s[temp_G].compute_inline()
            i, ic, h, w = s[temp_G].op.axis
            s[temp_G].split(w, factor=4)
            temp_R = s.cache_write(temp_G, "global")
            temp_S = s.cache_read(temp_R, "shared", [temp_G])
        else:
            s[temp].compute_inline()
            temp_S = s.cache_read(temp, "shared", [Output])
            temp_R = temp_S

        Filter_S = s.cache_read(Filter, "shared", [Output])

        if Output.op in s.outputs:
            Out = Output
            Out_L = s.cache_write(Out, "local")
        else:
            Out = outs[0].op.output(0)
            s[Output].set_scope("local")
            Out_L = Output

        if util.get_const_int(Filter.shape[3]) == 7 or (util.get_const_int(Output.shape[2] == 224) and flag < 128):
            conv2d_224_3_64(s, temp, temp_R, temp_S, Filter_S, Out, Out_L, flag)
        elif 128 < flag < 512:
            conv2d_56_64_128(s, temp, temp_R, temp_S, Filter_S, Out, Out_L, flag)
        elif flag >= 512:
            conv2d_14_256_256(s, temp, temp_R, temp_S, Filter, Filter_S, Out, Out_L)
        else:
            conv2d_56_64_64(s, Filter, temp_S, Filter_S, Out, Out_L)

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_injective(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule conv2d_transpose_nchw
        if 'conv2d_transpose_nchw' in OP.tag:
            temp = OP.input_tensors[0]
            DilatedInput = temp.op.input_tensors[0]
            s[DilatedInput].compute_inline()
            Filter = OP.input_tensors[1]
            Output = OP.output(0)
            schedule(temp, Filter, Output)

    traverse(outs[0].op)
    return s


@generic.schedule_conv2d_transpose_nchw.register(["cuda", "gpu"])
def schedule_conv2d_transpose_nchw(outs):
    """Schedule for conv2d_transpose_nchw.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_transpose_nchw
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d_transpose_nchw.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    batch_size = util.get_const_int(outs[0].op.output(0).shape[0])
    if batch_size > 1:
        raise RuntimeError("Batch size: %d is too large for this schedule" % batch_size)
    return schedule_conv2d_transpose_small_batch(outs)
