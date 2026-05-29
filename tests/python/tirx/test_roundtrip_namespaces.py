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
from tvm.script import tirx as Tx


def from_source(code):
    return tvm.script.from_source(code)


def test_roundtrip_tir_namespaces_minimal():
    # Exercise a selection of namespace ops and ensure round-trip consistency
    @Tx.prim_func
    def func(a_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(a_ptr, (2, 2), "float16")
        Tx.ptx.wgmma.commit_group()
        Tx.cuda.cluster_sync()
        Tx.ptx.cp_async.wait_group(0)
        Tx.ptx.fence.proxy_async("shared::cta")
        Tx.cuda.printf("ok")
        Tx.nvshmem.quiet()
        Tx.nki.identity(A[0, 0], 1)

    code = func.script()
    roundtripped = from_source(code)
    assert roundtripped.script() == code
    assert_structural_equal(func, roundtripped)
