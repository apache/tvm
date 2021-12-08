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
"""Extract information from the identity operator in TIR."""
from typing import Dict, Tuple
import tvm
from .spec import SerialKernel, SerialActivation, SerialPooling, SerialPadding, SerialFeatureMap
from .utils import get_op_attrs, get_base_address, get_strides, get_loads


def _get_feature_map(stmt: tvm.tir.AttrStmt, fm_type: str) -> Tuple[SerialFeatureMap, tvm.tir.Var]:
    """Get the feature map parameters from a loop nest of any shape (as long there are at
    most 4 nested loops).

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of a loop nest.
    fm_type: str
        Either "ifm" or "ofm", depending on whether it is an input or output feature map

    Returns
    -------
    SerialFeatureMap
        The serializable feature map.
    output_pointer : tvm.tir.Var
        The pointer produced by the operation.

    """
    assert fm_type in ("ifm", "ofm")

    attrs, body = get_op_attrs(stmt)

    loops = []
    inner = body
    # extact the loops and the innermost statement
    while hasattr(inner, "body"):
        loops.append(inner)
        inner = inner.body

    # If the batch size loop is present, we need to remove it
    if len(loops) > 3:
        assert loops[0].extent == 1
        loops = loops[1:]

    fm_inner = inner.value if fm_type == "ifm" else inner

    stride_vars = [l.loop_var for l in loops]
    strides = get_strides(fm_inner.index, stride_vars)

    base_address = get_base_address(fm_inner.index)
    data_type = inner.buffer_var.type_annotation.element_type.dtype
    pointer = fm_inner.buffer_var

    serial_feature_map = SerialFeatureMap(
        data_type=data_type,
        height=loops[0].extent,
        width=loops[1].extent if len(loops) > 1 else 1,
        channels=loops[2].extent if len(loops) > 2 else 1,
        tile_height_0=loops[0].extent,
        tile_height_1=0,
        tile_width_0=loops[1].extent if len(loops) > 1 else 1,
        tile_address_0=tvm.tir.Load(data_type, pointer, base_address),
        tile_address_1=0,
        tile_address_2=0,
        tile_address_3=0,
        scale=attrs["scale"],
        zero_point=attrs["zero_point"],
        layout="NHWC",
        stride_h=strides[0] if len(strides) > 0 else 1,
        stride_w=strides[1] if len(strides) > 1 else 1,
        stride_c=strides[2] if len(strides) > 2 else 1,
    )

    output_pointer = inner.buffer_var

    return serial_feature_map, output_pointer


def get_identity_params(
    stmt: tvm.tir.AttrStmt,
    producers: Dict[tvm.tir.Var, tvm.tir.AttrStmt],
    consumers: Dict[tvm.tir.Var, tvm.tir.AttrStmt],
) -> Tuple[SerialPooling, tvm.tir.Var, tvm.tir.Var]:
    """Get the parameters necessary to construct a call_extern for an identity pooling.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of an identity pooling loop nest.
    producers : Dict[tvm.tir.Var, tvm.tir.AttrStmt]
        A dictionary to associate pointers with the loop nest
        that produces their values.
    consumers : Dict[tvm.tir.Var, tvm.tir.AttrStmt]
        A dictionary to associate pointers with the loop nest
        that consumes their values.

    Returns
    -------
    SerialPooling
        The parameters needed to construct a 2D pooling.
    output_pointer : tvm.tir.Var
        The output pointer of the pooling operation.
    replace_pointer : tvm.tir.Var
        The output pointer of the DMA write operation, which is to replace
        the pooling output pointer.
    is_allocator : bool
        Whether this operator allocates its output.

    """
    attrs, _ = get_op_attrs(stmt)
    # Find the inner loop
    while hasattr(stmt, "body"):
        stmt = stmt.body

    # loads = [input, LUT, LUT]
    loads = get_loads(stmt)

    input_pointer = loads[0].buffer_var
    output_pointer = stmt.buffer_var

    read = producers[input_pointer]
    write = consumers[output_pointer]

    serial_ifm, _ = _get_feature_map(read, "ifm")
    serial_ofm, write_output_pointer = _get_feature_map(write, "ofm")

    replace_pointer = write_output_pointer

    is_allocator = True
    if write_output_pointer not in producers:
        is_allocator = False
    elif producers[write_output_pointer] != write:
        is_allocator = False

    # TODO: We might want to support stand alone ReLU in the future by adding clip_min and
    # clip max attributes to the identity operator
    serial_activation = SerialActivation(op=attrs["activation"], clip_min=0, clip_max=0)

    # Create a serialized identity pooling to be run on the NPU
    return (
        SerialPooling(
            ifm=serial_ifm,
            ofm=serial_ofm,
            pooling_type="AVG",
            pool_shape=SerialKernel(1, 1, 1, 1, 1, 1),
            padding=SerialPadding(0, 0, 0, 0),
            activation=serial_activation,
            upscale="NONE",
            rounding_mode="TFL",
        ),
        output_pointer,
        replace_pointer,
        is_allocator,
    )
