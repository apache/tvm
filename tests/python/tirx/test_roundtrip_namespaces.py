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
    return tvm.script.from_source(code)


def test_roundtrip_tir_namespaces_minimal():
    # Exercise a selection of namespace ops and ensure round-trip consistency
    @T.prim_func
    def func(a_ptr: T.handle) -> None:
        A = T.match_buffer(a_ptr, (2, 2), "float16")
        T.ptx.wgmma.commit_group()
        T.cuda.cluster_sync()
        T.ptx.cp_async.wait_group(0)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.printf("ok")
        T.nvshmem.quiet()
        T.nki.identity(A[0, 0], 1)

    code = func.script()
    roundtripped = from_source(code)
    assert roundtripped.script() == code
    assert_structural_equal(func, roundtripped)
