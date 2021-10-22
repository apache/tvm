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


def test_simple_cast_fold():
    data = relay.var("data", shape=[1, 3, 224, 224], dtype="float32")
    out = relay.cast(data, "float16")
    out = relay.add(out, out)
    mod = tvm.IRModule.from_expr(out)
    mod = tvm.relay.transform.InferType()(mod)
    mod = tvm.relay.transform.FoldTypeTransformation()(mod)

    data_fp16 = relay.var("data", shape=[1, 3, 224, 224], dtype="float16")
    out = relay.add(data_fp16, data_fp16)
    expected_mod = tvm.IRModule.from_expr(out)
    expected_mod = tvm.relay.transform.InferType()(expected_mod)

    assert tvm.ir.structural_equal(mod, expected_mod)


def test_simple_quantize_fold():
    data = relay.var("data", shape=[1, 3, 224, 224], dtype="float32")
    out = relay.qnn.op.quantize(data, relay.const(2.0), relay.const(0), out_dtype="uint8")
    out = relay.add(out, out)

    mod = tvm.IRModule.from_expr(out)
    mod = tvm.relay.transform.InferType()(mod)
    mod = tvm.relay.transform.FoldTypeTransformation()(mod)

    data_fp16 = relay.var("data", shape=[1, 3, 224, 224], dtype="uint8")
    out = relay.add(data_fp16, data_fp16)
    expected_mod = tvm.IRModule.from_expr(out)
    expected_mod = tvm.relay.transform.InferType()(expected_mod)

    assert tvm.ir.structural_equal(mod, expected_mod)
