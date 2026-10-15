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
from tvm.script import ir as I
from tvm.script import tirx as T


def test_decorate_device():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(x: T.int32):
            T.evaluate(x)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(x: T.int32):
            with T.region("tirx.device_scope", []):
                T.evaluate(x)

    After = tvm.tirx.transform.DecorateDeviceScope()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
