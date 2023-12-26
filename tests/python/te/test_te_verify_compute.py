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
from tvm import te


def test_verify_compute():
    n = te.size_var("n")
    m = te.size_var("m")
    A = te.placeholder((n, m), name="A")
    k = te.reduce_axis((0, m), "k")
    k_ = te.reduce_axis((0, m - 1), "k_")
    f1 = lambda i: te.sum(A[i, k], axis=k)
    f2 = lambda i: A[i, 0] + 1
    f3 = lambda i: te.sum(A[i, k], axis=k) + 1
    f4 = lambda i: A[i, 0] * (te.sum(A[i, k], axis=k) + 1)
    f5 = lambda i: (te.sum(A[i, k], axis=k), A[i, 0] + 1)
    f6 = lambda i: (te.sum(A[i, k], axis=k), te.sum(A[i, k_], axis=k_))

    #
    # Valid compute
    try:
        B = te.compute((n,), f1, name="B")
    except tvm._ffi.base.TVMError as ex:
        assert False

    #
    # Valid compute
    try:
        B = te.compute((n,), f2, name="B")
    except tvm._ffi.base.TVMError as ex:
        assert False

    #
    # Invalid compute with non top level reduction
    try:
        B = te.compute((n,), f3, name="B")
        assert False
    except tvm._ffi.base.TVMError as ex:
        pass

    #
    # Invalid compute with non top level reduction
    try:
        B = te.compute((n,), f4, name="B")
        assert False
    except tvm._ffi.base.TVMError as ex:
        pass

    #
    # Invalid compute with reduction and non-reduction batch ops
    try:
        B0, B1 = te.compute((n,), f5, name="B")
        assert False
    except tvm._ffi.base.TVMError as ex:
        pass

    #
    # Invalid compute with unequal batch reduction ops
    try:
        B0, B1 = te.compute((n,), f6, name="B")
        assert False
    except tvm._ffi.base.TVMError as ex:
        pass


if __name__ == "__main__":
    test_verify_compute()
