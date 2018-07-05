# pylint: disable=invalid-name,unused-variable,invalid-name
"""Conv2D schedule ported from RASP

Used for CPU conv2d
"""
from __future__ import absolute_import as _abs

from topi.nn.conv2d import conv2d, _get_schedule
from topi.nn.conv2d import SpatialPack, Im2ColPack, Workload
from topi.rasp import conv2d as _rasp_conv2d
from topi import generic

_WORKLOADS = [
    Workload('int8', 'int32', 224, 224, 3, 64, 7, 7, 3, 3, 2, 2),
    Workload('int8', 'int32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
    Workload('int8', 'int32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
    Workload('int8', 'int32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
    Workload('int8', 'int32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
    Workload('int8', 'int32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
    Workload('int8', 'int32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
    Workload('int8', 'int32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
    Workload('int8', 'int32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
    Workload('int8', 'int32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
    Workload('int8', 'int32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
    Workload('int8', 'int32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
]
_SCHEDULES = [
    # float32 imagenet
    SpatialPack(1, 8, 4, 1, 4, True),
    SpatialPack(1, 7, 4, 2, 4, True),
    SpatialPack(1, 4, 8, 4, 1, True),
    SpatialPack(1, 4, 4, 1, 16, False),
    SpatialPack(1, 4, 8, 4, 8, False),
    SpatialPack(1, 7, 4, 3, 8, True),
    SpatialPack(1, 2, 8, 1, 8, True),
    SpatialPack(2, 1, 16, 1, 4, True),
    SpatialPack(1, 7, 4, 1, 1, True),
    Im2ColPack(7, 4, 1, 16, True),
    Im2ColPack(7, 4, 1, 8, False),
    Im2ColPack(7, 4, 1, 16, False),
]

@_get_schedule.register(["vtacpu", "vta"])
def _schedule_conv2d(wkl):
    if wkl not in _WORKLOADS:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    idx = _WORKLOADS.index(wkl)
    sch = _SCHEDULES[idx]
    return sch

conv2d.register(["vtacpu", "vta"], _rasp_conv2d._declaration_conv2d)

generic.schedule_conv2d_nchw.register(
    ["vtacpu", "vta"],
    _rasp_conv2d.schedule_conv2d_nchw)
