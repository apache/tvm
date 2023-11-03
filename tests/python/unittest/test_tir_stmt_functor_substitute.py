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

from tvm.script import tir as T
from tvm.tir.stmt_functor import substitute


class BaseCompare(tvm.testing.CompareBeforeAfter):
    def transform(self):
        def inner(mod):
            func = mod["main"]
            vmap = {func.params[0]: 16}
            new_func = tvm.tir.PrimFunc(params=[], body=substitute(func.body, vmap))
            return tvm.IRModule.from_expr(new_func)

        return inner


class TestBasicSubstitute(BaseCompare):
    def before(n: T.int32):
        for i in range(n):
            T.evaluate(i)

    def expected():
        for i in range(16):
            T.evaluate(i)


class TestSubstituteAllocate(BaseCompare):
    def before(n: T.int32):
        A_data = T.allocate([n], "float32")
        T.evaluate(A_data)

    def expected():
        A_data = T.allocate([16], "float32")
        T.evaluate(A_data)


class TestSubstituteBufferLoad(BaseCompare):
    def before(n: T.int32):
        A_data = T.allocate([n], "float32")
        A = T.Buffer(n, "float32", data=A_data)
        for i in range(n):
            T.evaluate(A[i])

    def expected():
        A_data = T.allocate([16], "float32")
        A = T.Buffer(16, "float32", data=A_data)
        for i in range(16):
            T.evaluate(A[i])


class TestSubstituteDeclBuffer(BaseCompare):
    def before(n: T.int32):
        A_data = T.allocate([n], "float32")
        A = T.decl_buffer(n, "float32", data=A_data)
        T.evaluate(A.data)

    def expected():
        A_data = T.allocate([16], "float32")
        A = T.decl_buffer(16, "float32", data=A_data)
        T.evaluate(A.data)


if __name__ == "__main__":
    tvm.testing.main()
