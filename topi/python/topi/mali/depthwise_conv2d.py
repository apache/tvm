# pylint: disable=invalid-name,unused-variable,unused-argument
"""depthwise_conv2d schedule on ARM Mali GPU"""

import tvm
from tvm import autotvm

from ..generic import schedule_depthwise_conv2d_nchw
from ..nn import depthwise_conv2d_nchw
from ..util import traverse_inline

# register original implementation of depthwise_conv2d_nchw since we don't need to change this part
autotvm.register_topi_compute(depthwise_conv2d_nchw, 'mali', 'direct',
                              depthwise_conv2d_nchw.fdefault)

# register customized schedule for arm cpu.
@autotvm.register_topi_schedule(schedule_depthwise_conv2d_nchw, 'mali', 'direct')
def schedule_depthwise_conv2d_nchw_mali(cfg, outs):
    """Schedule depthwise conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The configuration of this template
    outs: Array of Tensor
        The computation graph description of depthwise convolution2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nchw.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _schedule(pad_data, kernel, conv):
        """schedule depthwise_conv2d"""
        max_unroll = 16
        vec_size = [1, 2, 4, 8, 16]

        ##### space definition begin #####
        n, c, y, x = s[conv].op.axis
        bc, tc, ci = cfg.define_split("tile_c", c, num_outputs=3)
        by, ty, yi = cfg.define_split('tile_y', y, num_outputs=3)
        bx, tx, xi = cfg.define_split("tile_x", x, num_outputs=3)
        cfg.define_annotate('ann_spatial', [ci, yi, xi], policy='try_unroll_vec')

        # fallback support
        if cfg.is_fallback:
            ref_log = autotvm.tophub.load_reference_log(
                'mali', 'rk3399', 'depthwise_conv2d_nchw', 'direct')
            cfg.fallback_with_reference_log(ref_log)
        ###### space definition end ######


        # schedule padding
        n, c, y, x = s[pad_data].op.axis
        tile_and_bind3d(s, pad_data, c, y, x, cfg["tile_c"].size[1], 1, 1)

        # schedule dilation
        if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
            s[kernel].compute_inline()

        # schedule conv
        if conv.op not in s.outputs:
            s[conv].set_scope('local')
            OL = conv
            output = s.outputs[0].output(0)
        else:
            OL = s.cache_write(conv, 'local')
            output = conv

        n, c, y, x = s[output].op.axis
        bc, tc, ci = cfg['tile_c'].apply(s, output, c)
        by, ty, yi = cfg['tile_y'].apply(s, output, y)
        bx, tx, xi = cfg['tile_x'].apply(s, output, x)

        bc = s[output].fuse(n, bc)
        s[output].bind(bc, tvm.thread_axis("blockIdx.z"))
        s[output].bind(tc, tvm.thread_axis("threadIdx.z"))
        s[output].bind(by, tvm.thread_axis("blockIdx.y"))
        s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
        s[output].bind(tx, tvm.thread_axis("threadIdx.x"))

        di, dj = s[OL].op.reduce_axis
        s[OL].unroll(di)
        s[OL].unroll(dj)

        s[OL].compute_at(s[output], tx)
        n, ci, yi, xi = s[OL].op.axis

        cfg["ann_spatial"].apply(s, OL, [ci, yi, xi],
                                 axis_lens=[cfg['tile_c'].size[2], cfg['tile_y'].size[2],
                                            cfg['tile_x'].size[2]],
                                 max_unroll=max_unroll,
                                 vec_size=vec_size,
                                 cfg=cfg)

    def _callback(op):
        """traverse to find op to schedule"""
        # schedule depthwise_conv2d
        if op.tag == 'depthwise_conv2d_nchw':
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            conv = op.output(0)
            _schedule(pad_data, kernel, conv)

    traverse_inline(s, outs[0].op, _callback)
    return s


def tile_and_bind3d(s, tensor, z, y, x, z_factor=2, y_factor=None, x_factor=None):
    """ tile and bind 3d """
    y_factor = y_factor or z_factor
    x_factor = x_factor or y_factor
    zo, zi = s[tensor].split(z, z_factor)
    yo, yi = s[tensor].split(y, y_factor)
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].bind(zo, tvm.thread_axis("blockIdx.z"))
    s[tensor].bind(zi, tvm.thread_axis("threadIdx.z"))
    s[tensor].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, tvm.thread_axis("threadIdx.y"))
    s[tensor].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, tvm.thread_axis("threadIdx.x"))
    return zo, zi, yo, yi, xo, xi
