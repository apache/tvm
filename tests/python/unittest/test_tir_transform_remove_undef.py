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


class BaseBeforeAfter(tvm.testing.CompareBeforeAfter):
    @tvm.testing.fixture
    def transform(self):
        return tvm.tir.transform.RemoveStoreUndef()


class TestRemoveStoreUndef(BaseBeforeAfter):
    """Remove a store whose value is T.undef()"""

    def before(A: T.Buffer[1, "int32"]):
        A[0] = T.undef(dtype="int32")

    def expected(A: T.Buffer[1, "int32"]):
        T.evaluate(0)


class TestRemoveStoreUndefExpression(BaseBeforeAfter):
    """Expressions containing T.undef() are removed"""

    def before(A: T.Buffer[1, "int32"]):
        A[0] = 1 + T.undef(dtype="int32")

    def expected(A: T.Buffer[1, "int32"]):
        T.evaluate(0)


class TestKeepOtherCallNodes(BaseBeforeAfter):
    """Expressions containing other CallNodes are not removed"""

    def before(A: T.Buffer[1, "int32"], n: T.int32):
        A[0] = T.shift_left(n, 1, dtype="int32")

    expected = before


if __name__ == "__main__":
    tvm.testing.main()
