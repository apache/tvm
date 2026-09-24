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
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


def test_reject_blocks():
    @Ts.prim_func(private=True)
    def before(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.serial(0, T.int64(16)):
            for j in T.serial(0, T.int64(8)):
                with Ts.sblock():
                    vi = Ts.axis.spatial(T.int64(128), i * T.int64(8) + j)
                    B[vi] = A[vi] + T.float32(1)

    mod = tvm.IRModule.from_expr(before)
    with pytest.raises(ValueError, match="requires a function without S-TIR blocks"):
        tvm.tirx.transform.ForceNarrowIndexToInt32()(mod)
