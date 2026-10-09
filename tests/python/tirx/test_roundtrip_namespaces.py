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

import tvm
from tvm.ir import assert_structural_equal
from tvm.script import tirx as T


def from_source(code):
    return tvm.script.from_source(code, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})


def test_roundtrip_tir_namespaces_minimal():
    # Exercise a selection of namespace ops and ensure round-trip consistency
    @T.function
    def func(A: T.Tensor((2, 2), "float16")) -> None:
        T.ptx.wgmma.commit_group.sync.aligned()
        T.cuda.cluster_sync()
        T.ptx.cp.async_.wait_group(0)
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.printf("ok")
        T.nvshmem.quiet()
        T.nki.identity(A[0, 0], 1)

    code = func.script()
    roundtripped = from_source(code)
    assert roundtripped.script() == code
    assert_structural_equal(func, roundtripped)


def test_explicit_refresh_preserves_region_and_tile_forms():
    from tvm import tirx
    from tvm.ir import Op, register_op_attr
    from tvm.tirx import script

    region_name = "tirx.test_refresh_region"
    register_op_attr(
        region_name,
        "FRegionGetBodyParams",
        Op.get("tirx.device_entry").get_attr("FRegionGetBodyParams"),
    )
    register_op_attr(region_name, "TIRxOpCategory", "builtin")
    Op.get(region_name).set_signature()
    tile_name = "tirx.tile.test_refresh_tile"
    register_op_attr(tile_name, "TIRxOpCategory", "tile_primitive")
    Op.get(tile_name).set_signature(["dst", "src"])
    original = T.sqrt
    script._refresh_op_api()
    tirx.op._refresh_op_api()
    assert T.sqrt is original
    assert "test_refresh_region" in T.__all__
    assert "test_refresh_tile" in T.tile.__all__
    native = tirx.op.test_refresh_region(body=[], body_params=[])
    assert isinstance(native, tirx.RegionStmt)

    @T.function
    def func(A: T.Tensor((4,), "float32")):
        with T.test_refresh_region():
            T.tile.warp.test_refresh_tile(A[:], A[:], dispatch="example", max_inst_size=4)

    assert_structural_equal(func, from_source(func.script()))
    assert_structural_equal(func, tvm.ir.load_json(tvm.ir.save_json(func)))
    script._refresh_op_api()
    assert T.sqrt is original


def test_explicit_tile_refresh_preserves_op_operands():
    from tvm.ir import Op, register_op_attr
    from tvm.tirx import script

    name = "tirx.tile.test_refresh_reduction"
    register_op_attr(name, "TIRxOpCategory", "tile_primitive")
    Op.get(name).set_signature(["reduction"])
    script._refresh_op_api()

    @T.function
    def func():
        T.tile.test_refresh_reduction(Op.get("tirx.tile.sum"))

    assert_structural_equal(func, from_source(func.script()))


def test_explicit_refresh_backend_native_forms():
    from tvm import tirx
    from tvm.backend.cuda import op as cuda_op
    from tvm.ir import Op, register_op_attr
    from tvm.ir.op import _init_op_api

    region_name = "tirx.cuda.test_refresh_region"
    register_op_attr(
        region_name,
        "FRegionGetBodyParams",
        Op.get("tirx.device_entry").get_attr("FRegionGetBodyParams"),
    )
    register_op_attr(region_name, "TIRxOpCategory", "builtin")
    Op.get(region_name).set_signature()
    tile_name = "tirx.tile.test_refresh_cuda_factory"
    register_op_attr(tile_name, "TIRxOpCategory", "tile_primitive")
    Op.get(tile_name).set_signature()
    original = cuda_op.cuda_func_call
    _init_op_api("tirx.cuda", cuda_op.__name__)
    assert cuda_op.cuda_func_call is original
    assert isinstance(cuda_op.test_refresh_region(body=[]), tirx.RegionStmt)
    factory = cuda_op._op_api_factory(Op.get(tile_name), cuda_op.__name__)
    assert isinstance(factory(), tirx.TilePrimitiveCall)


def test_explicit_refresh_updates_initialized_s_tir_exports():
    from tvm.ir import Op, PrimType, register_op_attr
    from tvm.s_tir import script as S
    from tvm.tirx import script

    S._initialize()
    original = S.function
    name = "tirx.test_refresh_shared"
    register_op_attr(name, "TIRxOpCategory", "builtin")
    register_op_attr(name, "TFixedReturnType", PrimType("int32"))
    Op.get(name).set_signature()
    script._refresh_op_api()
    S._refresh_op_api()
    assert S.test_refresh_shared is T.test_refresh_shared
    assert "test_refresh_shared" in S.__all__
    assert S.function is original
