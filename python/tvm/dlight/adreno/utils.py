# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Utilis for Adreno operators."""

# pylint: disable=import-outside-toplevel, unused-argument, invalid-name, missing-function-docstring
from typing import List

from tvm.target import Target
from tvm import tir

from ..analysis import BlockInfo


def get_texture_storage(block_info: BlockInfo):
    """
    Returns the texture layout acceptable for the shape

    Parameters
    ----------
    shape: array
        Shape of the tensor to be packed to texture
    """
    # certain limitation of the Qualcomm devices. Subject to be determined for certain device
    # individually, but until we have access to remote device during compilation, we have to
    # define it uniformly for all target devices
    # spatial_limit = 16384, depth_limit = 2048
    # TODO: Check Write Bufs.
    shape = block_info.write_bufs[0].buf_region.buffer.shape

    spatial_limit = Target.current().attrs["texture_spatial_limit"]
    depth_limit = Target.current().attrs["texture_depth_limit"]

    if len(shape) > 4:
        if shape[0] < spatial_limit and shape[1] * shape[2] * shape[3] < spatial_limit:
            return "global.texture-weight"
        elif shape[0] < depth_limit and shape[2] * shape[3] < spatial_limit:
            return "global.texture-nhwc"
        elif (
            shape[0] * shape[1] < depth_limit
            and shape[2] < spatial_limit
            and shape[3] < spatial_limit
        ):
            return "global.texture"
    elif len(shape) > 3:
        if shape[0] < spatial_limit and shape[1] * shape[2] < spatial_limit:
            return "global.texture-weight"
        elif shape[0] < depth_limit and shape[1] < spatial_limit and shape[2] < spatial_limit:
            return "global.texture"
    elif len(shape) == 3:
        if shape[0] < spatial_limit and shape[1] < spatial_limit:
            return "global.texture-weight"

    return "global"


def schedule_inline_blocks(sch: tir.Schedule, blocks: List[tir.schedule.BlockRV] = None):
    from .fallback import Fallback

    return Fallback.schedule_inline_blocks(sch, blocks)


def schedule_default(sch, blocks: List[tir.schedule.BlockRV] = None):
    from .fallback import Fallback

    ret = []
    for blk in blocks:
        ret.append(Fallback.schedule_default(sch, blk))

    return ret


def schedule_storage_annotate(sch: tir.Schedule, func=get_texture_storage):
    # Check the Write Buffer isn't one of input Params and is Texturizable...
    from .fallback import Fallback

    return Fallback.schedule_annotate_storage(sch)


def schedule_fallback(sch, blk):
    from .fallback import Fallback

    return Fallback.schedule_fallback(sch)
