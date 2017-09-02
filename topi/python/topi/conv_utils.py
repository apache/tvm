"""Convolution utility"""
from __future__ import absolute_import as _abs
from collections import namedtuple
from . import target as _target

workload_entity = ['height', 'width', 'in_filter', 'out_filter',
            'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride']
Workload = namedtuple('Workload', workload_entity)

spatial_schedule_entity = ['vh', 'vw', 'vc', 'ba', 'bc', 'unroll']
ConvSpatialSchedule = namedtuple('ConvSpatialSchedule', spatial_schedule_entity)

im2col_schedule_entity = ['vp', 'vq', 'ba', 'bc', 'unroll']
ConvIm2ColSchedule = namedtuple('ConvIm2ColSchedule', im2col_schedule_entity)

# workloads of resnet18 on imagenet
workloads = [
    Workload(224, 224,   3,  64, 7, 7, 3, 3, 2, 2),
    Workload( 56,  56,  64,  64, 3, 3, 1, 1, 1, 1),
    Workload( 56,  56,  64,  64, 1, 1, 0, 0, 1, 1),
    Workload( 56,  56,  64, 128, 3, 3, 1, 1, 2, 2),
    Workload( 56,  56,  64, 128, 1, 1, 0, 0, 2, 2),
    Workload( 28,  28, 128, 128, 3, 3, 1, 1, 1, 1),
    Workload( 28,  28, 128, 256, 3, 3, 1, 1, 2, 2),
    Workload( 28,  28, 128, 256, 1, 1, 0, 0, 2, 2),
    Workload( 14,  14, 256, 256, 3, 3, 1, 1, 1, 1),
    Workload( 14,  14, 256, 512, 3, 3, 1, 1, 2, 2),
    Workload( 14,  14, 256, 512, 1, 1, 0, 0, 2, 2),
    Workload(  7,   7, 512, 512, 3, 3, 1, 1, 1, 1),
]

rasp_spatial_schedules = [
    ConvSpatialSchedule( 1,  8,  4,  1,  4,  True),
    ConvSpatialSchedule( 1,  7,  4,  2,  4,  True),
    ConvSpatialSchedule( 1,  4,  8,  4,  1,  True),
    ConvSpatialSchedule( 1,  4,  4,  1, 16, False),
    ConvSpatialSchedule( 1,  4,  8,  4,  8, False),
    ConvSpatialSchedule( 1,  7,  4,  3,  8,  True),
    ConvSpatialSchedule( 1,  2,  8,  1,  8,  True),
    ConvSpatialSchedule( 2,  1, 16,  1,  4,  True),
    ConvSpatialSchedule( 1,  7,  4,  1,  1,  True),
    ConvSpatialSchedule( 1,  1,  8,  4, 16, False),
    ConvSpatialSchedule( 1,  1, 16,  1,  8, False),
    ConvSpatialSchedule( 1,  1,  4,  1, 16,  True),
]

def get_workload(data, kernel, stride, padding):
    _, CI, H, W = map(lambda x: x.value, data.shape)
    CO, _, HK, WK = map(lambda x: x.value, kernel.shape)
    if isinstance(padding, (tuple, list)):
        HPAD, WPAD = padding
    else:
        HPAD, WPAD = padding, padding
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    return Workload(H, W, CI, CO, HK, WK, HPAD, WPAD, HSTR, WSTR)

def get_schedule(wkl, target=None):
    if wkl not in workloads:
        raise ValueError, "no schedule for such workload: {}".format(wkl)
    idx = workloads.index(wkl)
    if target is None:
        target = _target.current_target()
    else:
        target = _target.Target(target)
    if target == _target.rasp():
        return rasp_spatial_schedules[idx]
    else:
        raise ValueError, "no schedule for such target"

def infer_pad(data, data_pad):
    if data_pad is None:
        return 0, 0
    else:
        _, _, H, W = map(lambda x: x.value, data.shape)
        _, _, TH, TW = map(lambda x: x.value, data_pad.shape)
        hpad = (TH - H) / 2
        wpad = (TW - W) / 2
        return hpad, wpad

def infer_stride(data, kernel, out):
    _, _, IH, IW = map(lambda x: x.value, data.shape)
    _, _, KH, KW = map(lambda x: x.value, kernel.shape)
    _, _, OH, OW = map(lambda x: x.value, out.shape)
    hstride = (IH - KH) / (OH - 1)
    wstride = (IW - KW) / (OW - 1)
    return hstride, wstride
