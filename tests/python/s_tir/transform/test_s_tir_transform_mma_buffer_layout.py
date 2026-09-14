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

import tvm
import tvm.testing
from tvm import s_tir, tirx


@pytest.mark.parametrize(
    "scope,shape", [("m16n8k8.matrixA", (32, 8)), ("m16n8k8.matrixB", (8, 32))]
)
@pytest.mark.parametrize("access_kind", ["load", "store"])
def test_explicit_matrix_ab_access_is_rejected(scope, shape, access_kind):
    buffer = tirx.decl_buffer(shape, "float32", scope=scope)
    if access_kind == "load":
        body = tirx.Evaluate(tirx.BufferLoad(buffer, [0, 0]))
    else:
        body = tirx.BufferStore(buffer, 0.0, [0, 0])
    block = tirx.SBlock([], [], [], "root", body, alloc_buffers=[buffer])
    func = tirx.PrimFunc([], tirx.SBlockRealize([], True, block))

    with pytest.raises(tvm.error.InternalError, match=f"{scope}.*explicit"):
        s_tir.transform.TransformMmaBufferLayout()(tvm.IRModule.from_expr(func))


if __name__ == "__main__":
    tvm.testing.main()
