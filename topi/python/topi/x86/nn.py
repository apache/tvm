# pylint: disable=invalid-name,too-many-locals
"""x86 nn operators"""
from __future__ import absolute_import as _abs
import tvm
from tvm import autotvm
from .. import generic, tag
from ..util import traverse_inline, get_const_tuple
from .. import nn
from .util import get_fp32_len
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity

@generic.schedule_softmax.register(["cpu"])
def schedule_softmax(outs):
    """Schedule for softmax

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of softmax
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)
    if len(s[x].op.axis) >= 5:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1], s[x].op.axis[2])
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 3:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1])
        s[x].parallel(fused)
    else:
        s[x].parallel(s[x].op.axis[0])
    return s


@generic.schedule_dense.register(["cpu", 'test'])
def schedule_dense(outs):
    """Schedule for dense

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    #print('schedule dense generic')
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'dense' in op.tag:
            output = outs[0]
            dense = op.output(0)

            # Write cache for blocks
            if dense.op in s.outputs:
                CC = s.cache_write(dense, 'local')
            else:
                CC = dense

            # Tile
            bnx = 1
            bny = 4
            x, y = output.op.axis
            xo, yo, xi, yi = s[output].tile(x, y, bnx, bny)

            xc, yc = s[CC].op.axis
            k, = s[CC].op.reduce_axis
            ko, ki = s[CC].split(k, factor=4)
            s[CC].reorder(ko, xc, ki, yc)

            s[CC].unroll(ki)
            s[CC].vectorize(yc)

            s[output].unroll(xi)
            s[output].vectorize(yi)

            fused = s[output].fuse(xo, yo)
            s[output].parallel(fused)
            s[CC].compute_at(s[output], fused)

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute(nn.dense, 'cpu', 'direct')
def _declaration_dense(cfg, data, weight, bias=None):
    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)

    # create tuning space
    cfg.define_split('tile_y', batch, num_outputs=3)
    cfg.define_split('tile_x', out_dim, num_outputs=3)
    cfg.define_split('tile_k', in_dim, num_outputs=2)
    # cfg.define_knob("auto_unroll_max_step", [0, 32, 512, 1500])
    # cfg.define_knob("unroll_explicit", [0, 1])

    packw_bn = cfg['tile_x'].size[-1]
    packw_shape = (out_dim // packw_bn, in_dim, packw_bn)
    packw = tvm.compute(packw_shape,
                        lambda z, y, x: weight[z * packw_bn + x, y], name='packed_weight')

    k = tvm.reduce_axis((0, in_dim), name='k')
    matmul = tvm.compute((batch, out_dim),
                         lambda y, x: tvm.sum(data[y, k] * packw[x // packw_bn, k, x % packw_bn], axis=k),
                         tag='dense')
    if bias is not None:
        matmul = tvm.compute((batch, out_dim),
                             lambda i, j: matmul[i, j] + bias[j],
                             tag=tag.BROADCAST)

    if cfg.is_fallback:
        _fallback_schedule(cfg, batch, out_dim, in_dim)

    return matmul


@autotvm.register_topi_schedule(generic.schedule_dense, 'cpu', 'direct')
def _schedule_dense(cfg, outs):
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def _callback(op):
        if 'dense' in op.tag:
            _schedule_dense_template(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _fallback_schedule(cfg, M, N, K):
    vec_width = get_fp32_len()

    tilex_ii = 1
    for bn in range(vec_width*2, 0, -1):
        if N % bn == 0:
            tilex_ii = bn
            break
    NN = N // tilex_ii
    tilex_oi = 1
    while NN // tilex_oi > 4:
        if (NN // tilex_oi) % 2 == 1:
            break
        tilex_oi *= 2

    tiley_ii = 8
    while M % tiley_ii != 0:
        tiley_ii //= 2
    MM = M // tiley_ii
    tiley_oi = 1
    while MM // tiley_oi > 4:
        if (MM // tiley_oi) % 2 == 1:
            break
        tiley_oi *= 2

    cfg["tile_y"] = SplitEntity([MM // tiley_oi, tiley_oi, tiley_ii])
    cfg["tile_x"] = SplitEntity([NN // tilex_oi, tilex_oi, tilex_ii])
    cfg["tile_k"] = SplitEntity([K // 4, 4])
    # cfg["auto_unroll_max_step"] = OtherOptionEntity(512)
    # cfg["unroll_explicit"] = OtherOptionEntity(False)


def _schedule_dense_template(cfg, s, C):
    A, packedB = s[C].op.input_tensors

    CC = s.cache_write(C, 'global')
    y, x = s[C].op.axis
    k, = s[CC].op.reduce_axis

    yt, yo, yi = cfg['tile_y'].apply(s, C, y)
    xt, xo, xi = cfg['tile_x'].apply(s, C, x)
    s[C].reorder(yt, xt, yo, xo, yi, xi)

    xyt = s[C].fuse(yt, xt)
    s[C].parallel(xyt)
    xyo = s[C].fuse(yo, xo)
    s[C].vectorize(xi)

    s[CC].compute_at(s[C], xyo)
    y, x = s[CC].op.axis

    ko, ki = cfg['tile_k'].apply(s, CC, k)
    s[CC].reorder(ko, ki, y, x)
    s[CC].vectorize(x)
    s[CC].unroll(y)
    s[CC].unroll(ki)

    z, y, x = s[packedB].op.axis
    s[packedB].reorder(z, x, y)
    s[packedB].parallel(z)
    s[packedB].vectorize(y)

    # s[C].pragma(xyo, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    # s[C].pragma(xyo, 'unroll_explicit', cfg['unroll_explicit'].val)

    return s
