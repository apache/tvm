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

def test_basic():
    a = tvm.var("a")
    b = tvm.var("b")
    c = tvm.var("c")
    m = tvm.arith.DetectClipBound(tvm.all(a * 1 < b * 6,
                                          a - 1 > 0), [a])
    assert tvm.ir_pass.Simplify(m[1] - (b * 6 - 1)).value == 0
    assert m[0].value == 2
    m = tvm.arith.DetectClipBound(tvm.all(a * 1 < b * 6,
                                          a - 1 > 0), [a, b])
    assert len(m) == 0
    m = tvm.arith.DetectClipBound(tvm.all(a + 10 * c <= 20,
                                          b - 1 > 0), [a, b])
    assert tvm.ir_pass.Simplify(m[1] - (20 - 10 * c)).value == 0
    assert tvm.ir_pass.Simplify(m[2] - 2).value == 0


if __name__ == "__main__":
    test_basic()
