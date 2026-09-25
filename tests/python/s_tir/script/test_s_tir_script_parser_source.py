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
"""Source and span tests for the canonical parser"""

import tvm
import tvm.testing
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


def test_nesting_parsing():
    class dummy:
        pass

    for i in range(1):

        @tvm.script.ir_module
        class Module:
            @Ts.prim_func
            def impl(
                A: T.Buffer((12, 196, 64), "float32"),
            ) -> None:
                T.evaluate(0)


if __name__ == "__main__":
    tvm.testing.main()
