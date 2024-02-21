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
import tvm.testing
from tvm.script import tir as T


def test_detect_cs():
    x = T.int32()
    y = T.int32()
    z = T.int32()
    c = T.floor(x + y + 0.5) + x + z * (T.floor(x + y + 0.5))
    m = tvm.arith.detect_common_subexpr(c, 2)
    assert c.a.a in m
    assert m[c.a.a] == 2


if __name__ == "__main__":
    tvm.testing.main()
