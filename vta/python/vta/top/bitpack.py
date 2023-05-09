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
# pylint: disable=ungrouped-imports, unsupported-binary-operation

"""Bit packing operators"""
from __future__ import absolute_import as _abs

import tvm
from tvm import te
from tvm.topi import utils

from tvm.relay.op.op import register_compute, register_injective_schedule
from tvm.relay.op.op import register_pattern, OpPattern


def bitpack(data, bits, pack_type="int8", name="bitpack"):
    """Packs lowest dimension into format needed by VTA

    Parameters
    ----------
    pack_axis : int
        index of the axis to pack in data
    bit_axis : int
        index of axis to place bit axis in resulting packed data

    Returns
    -------
    packed : Tensor
        The packed tensor.
    """
    shape_vec = list(data.shape)
    if pack_type == "int8":
        data_width = 8
    elif pack_type == "int16":
        data_width = 16
    elif pack_type == "int32":
        data_width = 32
    else:
        raise RuntimeError("Unknown pack type %s" % pack_type)
    assert data_width % bits == 0
    lanes = data_width // bits

    # Data must be in multiples of the data_width
    assert utils.get_const_int(shape_vec[-1]) % lanes == 0, "Not a multiple of word size"
    shape_vec[-1] = shape_vec[-1] // lanes
    oshape = tuple(shape_vec)

    def _bitpack(*indices):
        ret = None
        mask = tvm.tir.const((1 << bits) - 1, pack_type)
        for k in range(lanes):
            idx = list(indices)
            idx[-1] = idx[-1] * lanes + k
            elem = data(*idx).astype(pack_type)
            if k == 0:
                ret = elem & mask
            else:
                val = (elem & mask) << tvm.tir.const(k * bits, pack_type)
                ret = ret | val
        return ret

    return te.compute(oshape, _bitpack, name=name, tag="bitpack")


@register_compute("bitpack", level=15)
def compute_bitpack(attrs, inputs):
    lanes = attrs.lanes
    dtype = inputs[0].dtype
    assert dtype == "int8"
    width = 8
    assert width % lanes == 0
    bits = 8 // lanes
    return bitpack(inputs[0], bits, dtype)


register_injective_schedule("bitpack")
register_pattern("bitpack", OpPattern.INJECTIVE)
