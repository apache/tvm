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

from tvm.ir import Op
from tvm.tirx.buffer import decl_buffer
from tvm.tirx.stmt import TilePrimitiveCall


def _test(op: str, *args):
    return TilePrimitiveCall(*args, op=Op.get("tirx.tile." + op), workspace={}, config={})


def test_copy():
    A = decl_buffer((64, 64), "float32", scope="global")
    A_sm = decl_buffer((64, 64), "float32", scope="shared")
    _test("copy", A[0:64, 0:64], A_sm[0:64, 0:64])


def test_fill():
    A = decl_buffer((64, 64), "float32", scope="global")
    _test("fill", A[0:64, 0:64], 1.0)


def test_gemm():
    A = decl_buffer((64, 64), "float32", scope="global")
    B = decl_buffer((64, 64), "float32", scope="global")
    C = decl_buffer((64, 64), "float32", scope="global")
    D = decl_buffer((64, 64), "float32", scope="global")
    _test("gemm", D[:, :], A[:, :], B[:, :], C[:, :], True, False, 1.0, 0.0)


def test_buffer_replacer_no_shared_default():
    """Regression test for F4: BufferReplacer default dicts must not be shared."""
    from tvm.tirx.transform.common import BufferReplacer

    r1 = BufferReplacer()
    r2 = BufferReplacer()
    A = decl_buffer((64,), "float32")
    B = decl_buffer((64,), "float32")
    r1.buffer_map[A] = B
    # r2 must not see r1's mutation
    assert len(r2.buffer_map) == 0


def test_gemm_async_partial_scale_factor():
    """Regression test for F7: gemm_async must reject partial scale factors."""
    from tvm.tirx.script.builder.tirx import gemm_async

    A = decl_buffer((64, 64), "float16", scope="shared")
    B = decl_buffer((64, 64), "float16", scope="shared")
    C = decl_buffer((64, 64), "float16", scope="shared")
    SF = decl_buffer((64,), "float16", scope="shared")

    with pytest.raises(ValueError, match="SFA and SFB must both be provided or both be None"):
        gemm_async(C[:, :], A[:, :], B[:, :], SFA=SF[:])

    with pytest.raises(ValueError, match="SFA and SFB must both be provided or both be None"):
        gemm_async(C[:, :], A[:, :], B[:, :], SFB=SF[:])


if __name__ == "__main__":
    test_copy()
    test_fill()
    test_gemm()
    test_buffer_replacer_no_shared_default()
    test_gemm_async_partial_scale_factor()
