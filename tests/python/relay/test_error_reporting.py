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
from tvm import relay

def check_type_err(expr, msg):
    try:
        mod = relay.Module.from_expr(expr)
        mod = relay.transform.InferType()(mod)
        entry = mod["main"]
        expr = entry if isinstance(expr, relay.Function) else entry.body
        assert False
    except tvm.TVMError as err:
        assert msg in str(err)

def test_too_many_args():
    x = relay.var('x', shape=(10, 10))
    f = relay.Function([x], x)
    y = relay.var('y', shape=(10, 10))
    check_type_err(
        f(x, y),
        "the function is provided too many arguments expected 1, found 2;")

def test_too_few_args():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('y', shape=(10, 10))
    f = relay.Function([x, y], x)
    check_type_err(f(x), "the function is provided too few arguments expected 2, found 1;")

def test_rel_fail():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('y', shape=(11, 10))
    f = relay.Function([x, y], x + y)
    check_type_err(f(x, y), "Incompatible broadcast type TensorType([10, 10], float32) and TensorType([11, 10], float32);")

if __name__ == "__main__":
    test_too_many_args()
    test_too_few_args()
    test_rel_fail()
