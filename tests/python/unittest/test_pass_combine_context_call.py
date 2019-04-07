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

def test_for():
    dev_type = tvm.var("dev_type")
    def device_context(dev_id):
        ctx = tvm.call_extern("handle", "device_context", dev_type, dev_id)
        return tvm.make.Call(
            "handle", "tvm_thread_context", [ctx], tvm.expr.Call.Intrinsic, None, 0)

    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    A = ib.allocate("float32", n, name="A", scope="global")
    with ib.for_range(0, n, name="i") as i:
        ib.emit(tvm.call_extern
                ("int32", "fadd", device_context(0), A))
        with ib.for_range(0, 10, name="j") as j:
            ib.emit(tvm.call_extern
                    ("int32", "fadd", device_context(1), A))
            ib.emit(tvm.call_extern
                    ("int32", "fadd", device_context(0), A))
    body = ib.get()
    f = tvm.ir_pass.MakeAPI(body, "func", [dev_type, n], 2, True)
    f = tvm.ir_pass.CombineContextCall(f)
    assert f.body.value.dtype == "handle"
    assert f.body.body.value.dtype == "handle"


if __name__ == "__main__":
    test_for()
