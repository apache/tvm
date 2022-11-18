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
import pytest

pytest.importorskip("ethosu.vela")
import tvm.script
from tvm.relay.backend.contrib.ethosu.tir.dma import Tiles, create_tiles
from tvm.script import tir as T


def check_tiles_equal(tiles, expected):
    assert tiles.height_0 == expected.height_0
    assert tiles.height_1 == expected.height_1
    assert tiles.width_0 == expected.width_0
    if isinstance(tiles.address_0, int):
        assert tiles.address_0 == expected.address_0
    else:
        assert tiles.address_0.buffer == expected.address_0.buffer
        assert tiles.address_0.indices[0] == expected.address_0.indices[0]
    if isinstance(tiles.address_1, int):
        assert tiles.address_1 == expected.address_1
    else:
        assert tiles.address_1.buffer == expected.address_1.buffer
        assert tiles.address_1.indices[0] == expected.address_1.indices[0]
    if isinstance(tiles.address_2, int):
        assert tiles.address_2 == expected.address_2
    else:
        assert tiles.address_2.buffer == expected.address_2.buffer
        assert tiles.address_2.indices[0] == expected.address_2.indices[0]


def test_create_tiles_h():
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main(placeholder1: T.Buffer[(100,), "int8"], placeholder2: T.Buffer[(100,), "int8"]) -> None:
            T.attr("i0", "pragma_layout", "NHCWB16")
            for i0 in T.serial(0, 1):
                for i1 in T.serial(0, 6):
                    for i2 in T.serial(0, 1):
                        for i3 in T.serial(0, 1):
                            for i4 in T.serial(0, 16):
                                placeholder1[((i1*16) + i4)] = placeholder2[((T.floormod((i1 + 4), 6)*16) + i4)]

        __tvm_meta__ = None
    # fmt: on

    stmt = Module["main"].body
    tiles = create_tiles(stmt)
    buffer = stmt.body.body.body.body.body.body.value.buffer
    expected = Tiles(
        height_0=tvm.tir.expr.IntImm("int32", 2),
        height_1=tvm.tir.expr.IntImm("int32", 0),
        width_0=tvm.tir.expr.IntImm("int32", 1),
        address_0=tvm.tir.BufferLoad(buffer, [tvm.tir.expr.IntImm("int32", 64)]),
        address_1=tvm.tir.expr.IntImm("int32", 0),
        address_2=tvm.tir.BufferLoad(buffer, [tvm.tir.expr.IntImm("int32", 0)]),
    )
    check_tiles_equal(tiles, expected)


def test_create_tiles_w():
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main(placeholder1: T.Buffer[(100,), "int8"], placeholder2: T.Buffer[(100,), "int8"]) -> None:
            T.attr("i0", "pragma_layout", "NHCWB16")
            for i0 in T.serial(0, 1):
                for i1 in T.serial(0, 1):
                    for i2 in T.serial(0, 1):
                        for i3 in T.serial(0, 6):
                            for i4 in T.serial(0, 16):
                                placeholder1[((i3*16) + i4)] = placeholder2[((T.floormod((i3 + 4), 6)*16) + i4)]

        __tvm_meta__ = None
    # fmt: on

    stmt = Module["main"].body
    tiles = create_tiles(stmt)
    buffer = stmt.body.body.body.body.body.body.value.buffer
    expected = Tiles(
        height_0=tvm.tir.expr.IntImm("int32", 1),
        height_1=tvm.tir.expr.IntImm("int32", 1),
        width_0=tvm.tir.expr.IntImm("int32", 2),
        address_0=tvm.tir.BufferLoad(buffer, [tvm.tir.expr.IntImm("int32", 64)]),
        address_1=tvm.tir.BufferLoad(buffer, [tvm.tir.expr.IntImm("int32", 0)]),
        address_2=tvm.tir.expr.IntImm("int32", 0),
    )
    check_tiles_equal(tiles, expected)


def test_create_tiles_wrong_var_stride():
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main(placeholder1: T.Buffer[(100,), "int8"], placeholder2: T.Buffer[(100,), "int8"]) -> None:
            T.attr("i0", "pragma_layout", "NHCWB16")
            for i0 in T.serial(0, 1):
                for i1 in T.serial(0, 6):
                    for i2 in T.serial(0, 1):
                        for i3 in T.serial(0, 1):
                            for i4 in T.serial(0, 16):
                                placeholder1[((i1*16) + i4)] = placeholder2[((T.floormod((i1 + 4), 6)*8) + i4)]

        __tvm_meta__ = None
    # fmt: on

    stmt = Module["main"].body
    tiles = create_tiles(stmt)
    buffer = stmt.body.body.body.body.body.body.value.buffer
    expected = Tiles(
        height_0=tvm.tir.expr.IntImm("int32", 6),
        height_1=tvm.tir.expr.IntImm("int32", 0),
        width_0=tvm.tir.expr.IntImm("int32", 1),
        address_0=tvm.tir.BufferLoad(buffer, [tvm.tir.expr.IntImm("int32", 32)]),
        address_1=tvm.tir.expr.IntImm("int32", 0),
        address_2=tvm.tir.expr.IntImm("int32", 0),
    )
    check_tiles_equal(tiles, expected)


def test_create_tiles_multiple_var_occurrences():
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main(placeholder1: T.Buffer[(100,), "int8"], placeholder2: T.Buffer[(100,), "int8"]) -> None:
            T.attr("i0", "pragma_layout", "NHWC")
            for i0 in T.serial(0, 1):
                for i1 in T.serial(0, 5):
                    for i2 in T.serial(0, 6):
                        for i3 in T.serial(0, 4):
                            placeholder1[(((i1*24) + (i2*4)) + i3)] = placeholder2[(((((T.floordiv((i1 - 1), 2)*48) + (T.floormod((i1 + 1), 2)*24)) + (i2*4)) + i3) + 96)]

        __tvm_meta__ = None
    # fmt: on

    stmt = Module["main"].body
    tiles = create_tiles(stmt)
    buffer = stmt.body.body.body.body.body.value.buffer
    expected = Tiles(
        height_0=tvm.tir.expr.IntImm("int32", 5),
        height_1=tvm.tir.expr.IntImm("int32", 0),
        width_0=tvm.tir.expr.IntImm("int32", 6),
        address_0=tvm.tir.BufferLoad(buffer, [tvm.tir.expr.IntImm("int32", 72)]),
        address_1=tvm.tir.expr.IntImm("int32", 0),
        address_2=tvm.tir.expr.IntImm("int32", 0),
    )
    check_tiles_equal(tiles, expected)


if __name__ == "__main__":
    pytest.main([__file__])
