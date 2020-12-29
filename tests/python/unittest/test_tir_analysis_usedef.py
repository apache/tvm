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
from tvm import te


@pytest.mark.xfail
def test_loop_dependent_allocate():
    N = te.size_var("N")
    A = te.placeholder((2 * N,), "float32", "A")
    C = te.compute((N,), lambda i: A[2 * i] + A[i + 1], name="C")
    s = te.create_schedule(C.op)
    AA = s.cache_read(A, "local", [C])
    s[AA].compute_at(s[C], s[C].op.axis[0])
    # this line should fail due to IRUseDefAnalysis sees an allocate statement
    # referencing undefined variable
    tvm.lower(s, [A, C])


if __name__ == "__main__":
    test_loop_dependent_allocate()
