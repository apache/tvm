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


def test_decorate_device():
    x = te.var("x")
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([x], tvm.tir.Evaluate(x)))

    stmt = tvm.tir.transform.DecorateDeviceScope()(mod)["main"].body
    assert stmt.attr_key == "device_scope"


if __name__ == "__main__":
    test_decorate_device()
