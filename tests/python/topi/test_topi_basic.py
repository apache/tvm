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
from tvm import topi
from tvm.topi import utils


def test_util():
    x = tvm.tir.const(100, "int32")
    assert utils.get_const_int(x) == 100
    assert utils.get_const_tuple((x, x)) == (100, 100)


def test_ewise():
    m = te.var("m")
    l = te.var("l")
    A = te.placeholder((m, l), name="A")

    def test_apply(func, name):
        B = func(A)
        assert tuple(B.shape) == tuple(A.shape)
        assert B.op.body[0].op.name == "tir." + name

    test_apply(topi.exp, "exp")
    test_apply(topi.erf, "erf")
    test_apply(topi.tanh, "tanh")
    test_apply(topi.sigmoid, "sigmoid")
    test_apply(topi.log, "log")
    test_apply(topi.sqrt, "sqrt")
    test_apply(topi.rsqrt, "rsqrt")
    test_apply(topi.sin, "sin")
    test_apply(topi.cos, "cos")
    test_apply(topi.tan, "tan")
    test_apply(topi.atan, "atan")


if __name__ == "__main__":
    test_util()
    test_ewise()
