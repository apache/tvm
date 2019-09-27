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

def test_equal_expr():
    x = tvm.var('x')
    y = tvm.var('y')

    def func1():
        return x + y + 1

    def func2():
        return tvm.exp(tvm.truncdiv((x + y + 1) * y, 4))

    assert tvm.ir_pass.Equal(func1(), func1())
    assert tvm.ir_pass.Equal(func2(), func2())
    assert not tvm.ir_pass.Equal(func2(), func1())


def test_equal_compute():
    x = tvm.var('x')
    y = tvm.var('y')
    n = 128
    A = tvm.placeholder((n, n), name='A')
    B = tvm.placeholder((n, n), name='B')
    ii = tvm.var('i')
    jj = tvm.var('j')

    def func1():
        k = tvm.reduce_axis((0, n), name='k')
        return tvm.sum(A[ii, k] * B[jj, k], axis=k)

    Ab = tvm.decl_buffer((n,), name='A')
    n = tvm.var("n")
    def func2():
        ib = tvm.ir_builder.create()
        A = ib.buffer_ptr(Ab)
        with ib.for_range(0, n, name="i") as i:
            A[i] = A[i] + 1
            with ib.for_range(0, 10, name="j") as j:
                A[j] = A[j] + 2
        return ib.get()

    assert tvm.ir_pass.Equal(func1(), func1())
    assert tvm.ir_pass.Equal(func2(), func2())


if __name__ == "__main__":
    test_equal_expr()
    test_equal_compute()
