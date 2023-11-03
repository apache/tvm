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
"""Pad the data by constant value """
from __future__ import absolute_import as _abs

import tvm
from tvm import te

from .. import tag
from ..utils import equal_const_int


@tvm.te.tag_scope(tag=tag.INJECTIVE + ",pad")
def pad(data, pad_before, pad_after=None, pad_value=0.0, name="PadInput", attrs=None):
    """Pad Input with zeros.

    Parameters
    ----------
    data : tvm.te.Tensor
        n-D input, can be any layout.

    pad_before : list / tuple of n ints
        Pad width on each dimension to pad the before the axis begin.

    pad_after : list / tuple of n ints, optional
        Pad width each dimension to pad the after the axis end.

    pad_value : float, optional
        The value to be padded.

    name : str, optional
        The name prefix operators generated

    Returns
    -------
    Output : tvm.te.Tensor
        n-D, the same layout as Input.
    """
    n = len(data.shape)
    pad_after = pad_after if pad_after else pad_before
    if len(pad_before) != n:
        raise ValueError(f"Input dimension and pad_before dismatch : {n} vs {len(pad_before)}")
    if len(pad_after) != n:
        raise ValueError(f"Input dimension and pad_after dismatch : {n} vs {len(pad_after)}")
    ana = tvm.arith.Analyzer()
    dshape = []
    for dim in data.shape:
        if isinstance(dim, tvm.tir.Any):
            dshape.append(tvm.te.size_var("dim"))
        else:
            dshape.append(dim)
    out_shape = tuple(ana.simplify(dshape[i] + pad_before[i] + pad_after[i]) for i in range(n))
    pad_value = (
        pad_value
        if isinstance(pad_value, tvm.tir.PrimExpr)
        else tvm.tir.const(pad_value, data.dtype)
    )

    def _pad(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if equal_const_int(pad_before[i], 0) and equal_const_int(pad_after[i], 0):
                index_tuple.append(indices[i])
            else:
                index_tuple.append(indices[i] - pad_before[i])
                not_zero.append(indices[i] >= pad_before[i])
                not_zero.append(indices[i] < data.shape[i] + pad_before[i])
        if not_zero:
            not_zero = tvm.tir.all(*not_zero)
            return tvm.tir.if_then_else(not_zero, data(*index_tuple), pad_value)
        return data(*index_tuple)

    return te.compute(out_shape, _pad, name=name, attrs=attrs)


@tvm.te.tag_scope(tag=tag.INJECTIVE + ",pad")
def mirror_pad(data, pad_before, pad_after=None, mode="SYMMETRIC", name="MirrorPadInput"):
    """Pad Input with mirroring either symmetric or reflected.

    Parameters
    ----------
    data : tvm.te.Tensor
        n-D input, can be any layout.

    pad_before : list / tuple of n ints
        Pad width on each dimension to pad the before the axis begin.

    pad_after : list / tuple of n ints, optional
        Pad width each dimension to pad the after the axis end.

    mode: str, optional
        Type of mirror padding to apply. Must be SYMMETRIC or REFLECT

    name : str, optional
        The name prefix operators generated

    Returns
    -------
    Output : tvm.te.Tensor
        n-D, the same layout as Input.
    """
    n = len(data.shape)
    pad_after = pad_after if pad_after else pad_before
    if len(pad_before) != n:
        raise ValueError(f"Input dimension and pad_before dismatch : {n} vs {len(pad_before)}")
    if len(pad_after) != n:
        raise ValueError(f"Input dimension and pad_after dismatch : {n} vs {len(pad_after)}")
    ana = tvm.arith.Analyzer()
    out_shape = tuple(ana.simplify(data.shape[i] + pad_before[i] + pad_after[i]) for i in range(n))
    assert mode in ("SYMMETRIC", "REFLECT")
    mode = int(mode == "SYMMETRIC")

    def _pad(*indices):
        index_tuple = []
        above = []
        below = []
        for i in range(n):
            if equal_const_int(pad_before[i], 0) and equal_const_int(pad_after[i], 0):
                index_tuple.append(indices[i])
                above.append(False)
                below.append(False)
            else:
                index_tuple.append(indices[i] - pad_before[i])
                above.append(indices[i] >= data.shape[i] + pad_before[i])
                below.append(indices[i] < pad_before[i])
        mapped_tuple = []
        for i, axis in enumerate(index_tuple):
            mapped_axis = tvm.tir.if_then_else(below[i], -axis - mode, axis)
            mapped_axis = tvm.tir.if_then_else(
                above[i], (2 * (data.shape[i] - 1)) - axis + mode, mapped_axis
            )
            mapped_tuple.append(mapped_axis)
        return data(*mapped_tuple)

    return te.compute(out_shape, _pad, name=name)
