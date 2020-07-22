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
import tvm.relay as relay
import tvm.relay.transform as _transform


def test_canonicalize_cast():
    def before(data, conv_weight, bias1, bias2):
        x = relay.nn.conv2d(data, conv_weight,
                          channels=16,
                          kernel_size=(3, 3),
                          padding=(1, 1),
                          out_dtype="int8")
        x1 = relay.cast(x, dtype="int32")
        y1 = relay.add(x1, bias1)
        y2 = relay.add(x1, bias2)
        y = relay.add(y1, y2)
        return relay.Function([data, conv_weight, bias1, bias2], y)

    def expected(data, conv_weight, bias1, bias2):
        x = relay.nn.conv2d(data, conv_weight,
                          channels=16,
                          kernel_size=(3, 3),
                          padding=(1, 1),
                          out_dtype="int8")
        x1 = relay.cast(x, dtype="int32")
        x2 = relay.cast(x, dtype="int32")
        y1 = relay.add(x1, bias1)
        y2 = relay.add(x2, bias2)
        y = relay.add(y1, y2)
        return relay.Function([data, conv_weight, bias1, bias2], y)

    def check(shape):
        data = relay.var("data", shape=shape, dtype="int8")
        conv_weight = relay.var("weight")
        bias1 = relay.var("bias1", shape=(16, 1, 1), dtype="int32")
        bias2 = relay.var("bias2", shape=(16, 1, 1), dtype="int32")
        y = before(data, conv_weight, bias1, bias2)
        mod = tvm.IRModule.from_expr(y)
        seq = tvm.transform.Sequential([_transform.InferType(), _transform.CanonicalizeCast(),
                                     _transform.InferType()])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
        y = mod["main"]
        y_expected = expected(data, conv_weight, bias1, bias2)
        gv = relay.GlobalVar("expected")
        mod[gv] = y_expected
        mod = _transform.InferType()(mod)
        y_expected = mod["expected"]
        assert tvm.ir.structural_equal(y, y_expected)

    check((1, 16, 7, 7))


if __name__ == '__main__':
    test_canonicalize_cast()
