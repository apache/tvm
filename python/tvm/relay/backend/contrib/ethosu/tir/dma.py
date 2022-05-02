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
from typing import NamedTuple, Union
import tvm
from .utils import get_outer_loops, get_base_address, get_strides, get_op_attrs
from .spec import SerialBlockConfig, SerialFeatureMap, SerialPadding


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


class Tiles(NamedTuple):
    height_0: tvm.tir.expr.IntImm
    height_1: tvm.tir.expr.IntImm
    width_0: tvm.tir.expr.IntImm
    address_0: Union[tvm.tir.expr.BufferLoad, int]
    address_1: Union[tvm.tir.expr.BufferLoad, int]
    address_2: Union[tvm.tir.expr.BufferLoad, int]


def create_tiles(stmt: tvm.tir.stmt.AttrStmt) -> Tiles:
    """Given an AttrStmt this function returns a Tiles instance
    containing the tiles' addresses and dimensions.

    When rolling buffers are not used only tile0 is used.
    Otherwise, when rolling buffers are used, the statement contains
    modulo arithmetic operations, which are unsupported by the NPU.
    To support this scenario more than one tile is used.
    In particular, when the rolling variable is the height one
    tile0 and tile2 are used, otherwise, when the rolling variable
    is the width one, tile0 and tile1 are used.

    As an example consider this statement:

    // attr [iter_var(i0, )] pragma_op = "ethosu_read"
    // attr [iter_var(i0, )] pragma_zero_point = 0
    // attr [iter_var(i0, )] pragma_layout = "NHCWB16"
    // attr [iter_var(i0, )] pragma_scale = 1f
    for (i0, 0, 1) {
      for (i1, 0, 6) {
        for (i2, 0, 1) {
          for (i3, 0, 1) {
            for (i4, 0, 16) {
              ethosu_read[((i1*16) + i4)] = ethosu_write[((floormod((i1 + 4), 6)*16) + i4)]
            }
          }
        }
      }
    }

    You can see from the floormod expression floormod((i1 + 4), 6)
    that the rolling variable is i1, that is, the height one.
    In this case tile0 and tile2 are used.
    The height of tile0 will be 6 - 4 = 2, and height of tile2 will be 4.
    Both the width of tile0 and tile2 will be equal to the extent of the width variable.
    Also, the addresses are set accordingly.
    When the rolling variable is the width one a simmetric approach will be used.

    It is worth mentioning that only the height of tile0, the height of tile1,
    and the width of tile0 must be computed, the other ones can be inferred.
    """
    attrs, body = get_op_attrs(stmt)
    _, h, w, _, _, inner = get_outer_loops(body, attrs["layout"])
    base_address = [get_base_address(index) for index in inner.value.indices]
    read_stmt = inner.value
    floor_mod_mul = None

    def _compute_stride(for_stmt):
        stride = 1
        while isinstance(for_stmt.body, tvm.tir.For):
            for_stmt = for_stmt.body
            stride *= for_stmt.extent
        return stride

    def _get_floor_mod_mul(stmt):
        nonlocal floor_mod_mul
        if (
            isinstance(stmt, tvm.tir.expr.Mul)
            and isinstance(stmt.b, tvm.tir.expr.IntImm)
            and isinstance(stmt.a, tvm.tir.FloorMod)
            and isinstance(stmt.a.b, tvm.tir.expr.IntImm)
            and isinstance(stmt.a.a, tvm.tir.expr.Add)
            and isinstance(stmt.a.a.a, tvm.tir.expr.Var)
            and isinstance(stmt.a.a.b, tvm.tir.expr.IntImm)
        ):
            floor_mod_mul = stmt

    tvm.tir.stmt_functor.post_order_visit(read_stmt, _get_floor_mod_mul)
    if floor_mod_mul is not None:
        rolling_var = floor_mod_mul.a.a.a
        count = 0

        def _count_var(var):
            nonlocal count
            if var == rolling_var:
                count += 1

        tvm.tir.stmt_functor.ir_transform(inner, _count_var, None, ["tir.Var"])
        if count == 2:
            stride = floor_mod_mul.b
            tile_length = floor_mod_mul.a.b - floor_mod_mul.a.a.b
            if rolling_var == h.loop_var and _compute_stride(h) == stride:
                return Tiles(
                    height_0=tile_length,
                    height_1=0,
                    width_0=w.extent,
                    address_0=tvm.tir.BufferLoad(inner.value.buffer, base_address),
                    address_1=0,
                    address_2=tvm.tir.BufferLoad(inner.value.buffer, [0]),
                )
            if rolling_var == w.loop_var and _compute_stride(w) == stride:
                return Tiles(
                    height_0=h.extent,
                    height_1=h.extent,
                    width_0=tile_length,
                    address_0=tvm.tir.BufferLoad(inner.value.buffer, base_address),
                    address_1=tvm.tir.BufferLoad(inner.value.buffer, [0]),
                    address_2=0,
                )

    return Tiles(
        height_0=h.extent,
        height_1=0,
        width_0=w.extent,
        address_0=tvm.tir.BufferLoad(inner.value.buffer, base_address),
        address_1=0,
        address_2=0,
    )


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

    data_type = inner.buffer.data.type_annotation.element_type.dtype
    tiles = create_tiles(stmt)
    return (
        SerialFeatureMap(
            data_type=data_type,
            height=h.extent,
            width=w.extent,
            channels=c.extent,
            tile_height_0=tiles.height_0,
            tile_height_1=tiles.height_1,
            tile_width_0=tiles.width_0,
            tile_address_0=tiles.address_0,
            tile_address_1=tiles.address_1,
            tile_address_2=tiles.address_2,
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
    if "block_config_height" in attrs:
        block_config = SerialBlockConfig(
            height=int(attrs["block_config_height"]),
            width=int(attrs["block_config_width"]),
            depth=int(attrs["block_config_depth"]),
        )
    else:
        block_config = SerialBlockConfig(0, 0, 0)
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
        block_config,
        input_pointer,
        output_pointer,
    )


def get_ifm_params(pointer, producers_consumers, stmt):
    """Get the parameters associated with the DMA capabilities for an IFM.

    Parameters
    ----------
    pointer : tvm.tir.Var
        The pointer that the IFM DMA pipeline produces.
    producers_consumers: ProducersConsumers
        It associates pointers with the loop nest that produces
        their values and with the loop nest that consumes their values.

    Returns
    -------
    serial_ifm : SerialFeatureMap
        The serializable IFM.
    serial_padding : SerialPadding
        The serializable padding.

    """
    pad = producers_consumers.get_producer(pointer, stmt)
    serial_padding, input_pointer, _ = get_pad_params(pad)
    upscale = producers_consumers.get_producer(input_pointer, pad)
    input_pointer, _ = get_upscale_params(upscale)
    convert_to_nhwc = producers_consumers.get_producer(input_pointer, upscale)
    in_channels, input_pointer, _ = get_convert_to_nhwc_params(convert_to_nhwc)
    read = producers_consumers.get_producer(input_pointer, convert_to_nhwc)
    serial_ifm, _, _ = get_read_params(read)
    serial_ifm.channels = in_channels

    floor_mod_stmt = None
    for_stmt = None

    def _get_buffer_var(stmt):
        nonlocal for_stmt
        nonlocal floor_mod_stmt
        if isinstance(stmt, tvm.tir.For):
            for_stmt = stmt
        if isinstance(stmt, tvm.tir.FloorMod):
            floor_mod_stmt = stmt

    tvm.tir.stmt_functor.post_order_visit(stmt, _get_buffer_var)

    if floor_mod_stmt is not None:
        layout = get_op_attrs(read)[0]["layout"]
        channels = serial_ifm.channels
        if for_stmt.body.loop_var == floor_mod_stmt.a.a.a:
            height_a = floor_mod_stmt.b - floor_mod_stmt.a.b
            height_b = serial_ifm.height
            serial_ifm.height = height_a + height_b
            serial_ifm.tile_height_0 = serial_ifm.height
            address = serial_ifm.tile_address_0
            offset = (
                height_a * (channels // 16 + 1) * serial_ifm.width * 16
                if layout == "NHCWB16"
                else height_a * serial_ifm.width * channels
            )
            serial_ifm.tile_address_0 = tvm.tir.BufferLoad(
                address.buffer, [address.indices[0] - offset]
            )
        else:
            width_a = floor_mod_stmt.b - floor_mod_stmt.a.b
            width_b = serial_ifm.width
            serial_ifm.width = width_a + width_b
            serial_ifm.tile_width_0 = serial_ifm.width
            address = serial_ifm.tile_address_0
            offset = width_a * 16 if layout == "NHCWB16" else width_a * channels
            serial_ifm.tile_address_0 = tvm.tir.BufferLoad(
                address.buffer, [address.indices[0] - offset]
            )
    return serial_ifm, serial_padding


def get_ofm_params(pointer, producers_consumers, stmt):
    """Get the parameters associated with the DMA capabilities for an OFM.

    Parameters
    ----------
    pointer : tvm.tir.Var
        The pointer that the OFM DMA pipeline consumes.
    producers_consumers: ProducersConsumers
        It associates pointers with the loop nest that produces
        their values and with the loop nest that consumes their values.

    Returns
    -------
    serial_ifm : SerialFeatureMap
        The serializable OFM.
    serial_block_config : SerialBlockConfig
        The serializable block config.
    output_pointer : tvm.tir.Var
        The pointer that the OFM DMA pipeline produces.
    is_allocator : bool
        Whether this operator allocates its output.

    """
    convert_to_nhcwb16 = producers_consumers.get_consumer(pointer, stmt)
    out_channels, _, output_pointer = get_convert_to_nhcwb16_params(convert_to_nhcwb16)
    write = producers_consumers.get_consumer(output_pointer, convert_to_nhcwb16)
    serial_ofm, serial_block_config, _, output_pointer = get_write_params(write)
    is_allocator = True

    producer = producers_consumers.get_producer(output_pointer, write)
    if producer is None or producer != write:
        is_allocator = False
    serial_ofm.channels = out_channels
    return serial_ofm, serial_block_config, output_pointer, is_allocator
