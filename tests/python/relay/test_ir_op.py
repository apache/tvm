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
from tvm import relay

def test_op_attr():
    log_op = relay.op.get("log")

    @relay.op.register("exp", "ftest")
    def test(x):
        return x + 1

    assert log_op.num_inputs  == 1
    assert log_op.get_attr("ftest") is None
    assert relay.op.get("exp").get_attr("ftest")(1) == 2

def test_op_level1():
    x = relay.Var("x")

    for op_name in ["log", "exp", "sqrt", "rsqrt","tanh"]:
        y = getattr(relay, op_name)(x)
        assert y.op.name == op_name
        assert y.op.support_level == 1
        assert y.args[0] == x

def test_op_level3():
    x = relay.Var("x")

    for op_name in ["ceil", "floor", "trunc", "round", "abs", "negative"]:
        y = getattr(relay, op_name)(x)
        assert y.op.name == op_name
        assert y.op.support_level == 3
        assert y.args[0] == x

if __name__ == "__main__":
    test_op_attr()
    test_op_level1()
    test_op_level3()
