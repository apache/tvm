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
# pylint: disable=invalid-name, unused-argument
"""Extract parameters from the DMA operators in TIR."""
import tvm
from .utils import get_outer_loops, get_base_address, get_strides, get_op_attrs
from .spec import SerialFeatureMap, SerialPadding


def get_pad_params(stmt):
    """Get the padding parameters from a pad loop nest.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of a pad loop nest.

    Returns
    -------
    pad : SerialPadding
        The serializable padding.
    input_pointer : tvm.tir.Var
        The pointer consumed by the operation.
    output_pointer : tvm.tir.Var
        The pointer produced by the operation.

    """
    _, body = get_op_attrs(stmt)
    n, h, w, c, _, inner = get_outer_loops(body, "NHWC")
    output_pointer = inner.buffer.data
    pad = SerialPadding(top=0, left=0, bottom=0, right=0)
    if isinstance(inner.value, tvm.tir.Call):
        input_pointer = inner.value.args[1].buffer.data
    else:
        input_pointer = inner.value.buffer.data
        return pad, input_pointer, output_pointer

    padded_shape = [n.extent, h.extent, w.extent, c.extent]

    def _visit(expr):
        if isinstance(expr, tvm.tir.expr.LT):
            var = expr.a
            val = expr.b
            if var == h.loop_var:
                pad.bottom = padded_shape[1] - val
            else:
                pad.right = padded_shape[2] - val
        elif isinstance(expr, tvm.tir.expr.LE):
            var = expr.b
            val = expr.a
            if var == h.loop_var:
                pad.top = val
            else:
                pad.left = val

    cond = inner.value.args[0]
    tvm.tir.stmt_functor.post_order_visit(cond, _visit)
    return (
        pad,
        input_pointer,
        output_pointer,
    )


def get_upscale_params(stmt):
    """Get the upscale parameters from a loop nest.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of an upscale loop nest.

    Returns
    -------
    input_pointer : tvm.tir.Var
        The pointer consumed by the operation.
    output_pointer : tvm.tir.Var
        The pointer produced by the operation.
    """
    _, body = get_op_attrs(stmt)
    _, _, _, _, _, inner = get_outer_loops(body, "NHWC")
    if isinstance(inner.value, tvm.tir.Call):
        input_pointer = inner.value.args[1].buffer.data
    else:
        input_pointer = inner.value.buffer.data
    output_pointer = inner.buffer.data
    return (input_pointer, output_pointer)


def get_convert_to_nhwc_params(stmt):
    """Get the true number of channels from a convert_to_nhwc loop nest.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of a convert_to_nhwc loop nest.

    Returns
    -------
    int
        The true number of channels.
    input_pointer : tvm.tir.Var
        The pointer consumed by the operation.
    output_pointer : tvm.tir.Var
        The pointer produced by the operation.

    """
    attrs, body = get_op_attrs(stmt)
    _, _, _, c, _, inner = get_outer_loops(body, "NHWC")

    # Ignore the reduce sum operation inserted to ensure
    # compute that is deemed uneccesary isn't removed by TVM.
    if attrs["layout"] == "NHCWB16":
        inner = inner.body
        input_pointer = inner.value.b.buffer.data
    else:
        input_pointer = inner.value.buffer.data

    output_pointer = inner.buffer.data
    return c.extent, input_pointer, output_pointer


def get_convert_to_nhcwb16_params(stmt):
    """Get the true number of channels from a convert_to_nhcwb16 loop nest.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of a convert_to_nhcwb16 loop nest.

    Returns
    -------
    out_channels : int
        The true number of channels.
    input_pointer : tvm.tir.Var
        The pointer consumed by the operation.
    output_pointer : tvm.tir.Var
        The pointer produced by the operation.

    """
    attrs, body = get_op_attrs(stmt)
    _, _, _, c, b, inner = get_outer_loops(body, attrs["layout"])
    output_pointer = inner.buffer.data
    if isinstance(inner.value, tvm.tir.Call):
        cond = inner.value.args[0]
        out_channels = cond.b.value
        input_pointer = inner.value.args[1].buffer.data
    else:
        input_pointer = inner.value.buffer.data
        out_channels = c.extent * b.extent if attrs["layout"] == "NHCWB16" else c.extent

    return out_channels, input_pointer, output_pointer


def get_read_params(stmt):
    """Get the feature map parameters from a read loop nest.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of a read loop nest.

    Returns
    -------
    SerialFeatureMap
        The serializable feature map.
    input_pointer : tvm.tir.Var
        The pointer consumed by the operation.
    output_pointer : tvm.tir.Var
        The pointer produced by the operation.

    """
    attrs, body = get_op_attrs(stmt)
    _, h, w, c, _, inner = get_outer_loops(body, attrs["layout"])
    input_pointer = inner.value.buffer.data
    output_pointer = inner.buffer.data

    # Needed for stride calculation, can replace with
    # inner.value.buffer.strides in future.
    assert len(inner.value.indices) == 1, "Ethos-U DMA expects flattened buffers"
    stride_vars = [h.loop_var, w.loop_var, c.loop_var]
    strides = get_strides(inner.value.indices[0], stride_vars)

    base_address = [get_base_address(index) for index in inner.value.indices]
    data_type = inner.buffer.data.type_annotation.element_type.dtype
    return (
        SerialFeatureMap(
            data_type=data_type,
            height=h.extent,
            width=w.extent,
            channels=c.extent,
            tile_height_0=h.extent,
            tile_height_1=0,
            tile_width_0=w.extent,
            tile_address_0=tvm.tir.BufferLoad(inner.value.buffer, base_address),
            tile_address_1=0,
            tile_address_2=0,
            tile_address_3=0,
            scale=attrs["scale"],
            zero_point=attrs["zero_point"],
            layout=attrs["layout"],
            stride_h=strides[0],
            stride_w=strides[1],
            stride_c=strides[2],
        ),
        input_pointer,
        output_pointer,
    )


def get_write_params(stmt):
    """Get the feature map parameters from a write loop nest.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of a write loop nest.

    Returns
    -------
    SerialFeatureMap
        The serializable feature map.
    input_pointer : tvm.tir.Var
        The pointer consumed by the operation.
    output_pointer : tvm.tir.Var
        The pointer produced by the operation.

    """
    attrs, body = get_op_attrs(stmt)
    _, h, w, c, _, inner = get_outer_loops(body, attrs["layout"])
    input_pointer = inner.value.buffer.data
    output_pointer = inner.buffer.data

    # Needed for stride calculation, can replace with
    # inner.value.buffer.strides in future.
    assert len(inner.indices) == 1, "Ethos-U DMA expects flattened buffers"
    stride_vars = [h.loop_var, w.loop_var, c.loop_var]
    strides = get_strides(inner.indices[0], stride_vars)

    base_address = [get_base_address(index) for index in inner.indices]
    data_type = inner.buffer.data.type_annotation.element_type.dtype
    return (
        SerialFeatureMap(
            data_type=data_type,
            height=h.extent,
            width=w.extent,
            channels=c.extent,
            tile_height_0=h.extent,
            tile_height_1=0,
            tile_width_0=w.extent,
            tile_address_0=tvm.tir.BufferLoad(inner.buffer, base_address),
            tile_address_1=0,
            tile_address_2=0,
            tile_address_3=0,
            scale=attrs["scale"],
            zero_point=attrs["zero_point"],
            layout=attrs["layout"],
            stride_h=strides[0],
            stride_w=strides[1],
            stride_c=strides[2],
        ),
        input_pointer,
        output_pointer,
    )


def get_ifm_params(pointer, producers):
    """Get the parameters associated with the DMA capabilities for an IFM.

    Parameters
    ----------
    pointer : tvm.tir.Var
        The pointer that the IFM DMA pipeline produces.
    producers : dict of tvm.tir.Var to tvm.tir.AttrStmt
        A dictionary to associate pointers with the loop nest
        that produces their values.

    Returns
    -------
    serial_ifm : SerialFeatureMap
        The serializable IFM.
    serial_padding : SerialPadding
        The serializable padding.

    """
    pad = producers[pointer]
    serial_padding, input_pointer, _ = get_pad_params(pad)
    upscale = producers[input_pointer]
    input_pointer, _ = get_upscale_params(upscale)
    convert_to_nhwc = producers[input_pointer]
    in_channels, input_pointer, _ = get_convert_to_nhwc_params(convert_to_nhwc)
    read = producers[input_pointer]
    serial_ifm, _, _ = get_read_params(read)
    serial_ifm.channels = in_channels
    return serial_ifm, serial_padding


def get_ofm_params(pointer, consumers, producers):
    """Get the parameters associated with the DMA capabilities for an OFM.

    Parameters
    ----------
    pointer : tvm.tir.Var
        The pointer that the OFM DMA pipeline consumes.
    consumers : dict of tvm.tir.Var to tvm.tir.AttrStmt
        A dictionary to associate pointers with the loop nest
        that consumes their values.
    producers : dict of tvm.tir.Var to tvm.tir.AttrStmt
        A dictionary to associate pointers with the loop nest
        that produces their values.

    Returns
    -------
    serial_ifm : SerialFeatureMap
        The serializable OFM.
    output_pointer : tvm.tir.Var
        The pointer that the OFM DMA pipeline produces.
    is_allocator : bool
        Whether this operator allocates its output.

    """
    convert_to_nhcwb16 = consumers[pointer]
    out_channels, _, output_pointer = get_convert_to_nhcwb16_params(convert_to_nhcwb16)
    write = consumers[output_pointer]
    serial_ofm, _, output_pointer = get_write_params(write)
    is_allocator = True
    if output_pointer not in producers:
        is_allocator = False
    elif producers[output_pointer] != write:
        is_allocator = False
    serial_ofm.channels = out_channels
    return serial_ofm, output_pointer, is_allocator
