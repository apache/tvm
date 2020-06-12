"""All AutoSchedule Supported Operators"""
from __future__ import absolute_import as _abs
from tvm import ansor

@ansor.register_topi_schedule()
def schedule_dense_nopack(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_conv2d_nhwc(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_conv2d_NCHWc(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_reduce(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_pool(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_adaptive_pool(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_softmax(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_conv2d_nchw_int8(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_conv2d_nchw(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_depthwise_conv2d_nchw(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_depthwise_conv2d_nhwc(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_conv2d_NCHWc_int8(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_depthwise_conv2d_NCHWc(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_conv2d_transpose_nchw(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_conv3d_ncdhw(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_conv3d_ndhwc(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_conv1d_ncw(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_conv1d_nwc(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_dense_pack(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_batch_matmul(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_bitserial_conv2d_nchw(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_bitserial_conv2d_nhwc(cfg, outs):
    return ansor.gen_schedule(cfg, outs)

@ansor.register_topi_schedule()
def schedule_bitserial_dense(cfg, outs):
    return ansor.gen_schedule(cfg, outs)
