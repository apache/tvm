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
from tvm.relay.backend.contrib.ethosu.tir.compiler import lower_to_tir


def test_lower_to_tir():
    data = relay.var("data", shape=(1, 1, 1, 1024), dtype="uint8")
    weight = relay.var("weight", shape=(1, 1, 1024, 1001), dtype="int8")
    p2 = relay.var("p2", shape=(1, 1, 1, 1), dtype="int32")
    conv = relay.nn.conv2d(
        data,
        weight,
        kernel_size=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="int32",
    )
    multiply = relay.multiply(relay.const(-22, dtype="int32"), p2)
    tile = relay.tile(multiply, reps=(1, 1, 1, 1001))
    subtract = relay.subtract(conv, tile)
    func = subtract
    expr = relay.Function(relay.analysis.free_vars(func), func)
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    lower_to_tir(mod["main"])


if __name__ == "__main__":
    test_lower_to_tir()
