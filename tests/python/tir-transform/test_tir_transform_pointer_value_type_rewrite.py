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
import tvm.testing
from tvm import te
from tvm.driver.build_module import schedule_to_module
from tvm.script import tir as T


class BaseCompare(tvm.testing.CompareBeforeAfter):
    transform = tvm.tir.transform.PointerValueTypeRewrite()


class TestRewriteToShuffle(BaseCompare):
    @T.prim_func
    def before(A: T.Buffer((16,), "float32"), B: T.Buffer((4,), "float32")):
        A_local_data = T.allocate([16], "float32", scope="local")
        A_local = T.Buffer((16,), "float32", data=A_local_data, scope="local")
        for i in range(4):
            A_local[i * 4 : i * 4 + 4] = A[i * 4 : i * 4 + 4]
        for i in range(4):
            B[i] = A_local[i * 4] + A_local[i * 4 + 1] + A_local[i * 4 + 2] + A_local[i * 4 + 3]

    @T.prim_func
    def expected(A: T.Buffer((4,), "float32x4"), B: T.Buffer((4,), "float32")):
        A_local_data = T.allocate([4], "float32x4", scope="local")
        A_local = T.Buffer((4,), "float32x4", data=A_local_data, scope="local")
        for i in range(4):
            A_local[T.Div(i * 4, 4)] = A[T.Div(i * 4, 4)]
        for i in range(4):
            B[i] = (
                T.Shuffle([A_local[T.Div(i * 4, 4)]], [0])
                + T.Shuffle([A_local[T.Div(i * 4 + 1, 4)]], [1])
                + T.Shuffle([A_local[T.Div(i * 4 + 2, 4)]], [2])
                + T.Shuffle([A_local[T.Div(i * 4 + 3, 4)]], [3])
            )


class TestAddressOf(BaseCompare):
    @T.prim_func
    def before(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
        for i in range(4):
            T.evaluate(T.address_of(A[i * 4]))
            B[i * 4 : i * 4 + 4] = A[i * 4 : i * 4 + 4]

    @T.prim_func
    def expected(A: T.Buffer((16,), "float32"), B: T.Buffer((4,), "float32x4")):
        for i in range(4):
            T.evaluate(T.address_of(A[i * 4]))
            B[T.Div(i * 4, 4)] = A[i * 4 : i * 4 + 4]


class TestScalarReadWithoutWrite(BaseCompare):
    @T.prim_func
    def before(A: T.Buffer((16,), "float32")):
        for i in range(4):
            T.evaluate(A[i * 4])

    expected = before
