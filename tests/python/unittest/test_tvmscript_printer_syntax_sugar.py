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
import pytest
import tvm.testing
from tvm.script.parser import tir as T
from tvm.script import script


def _test(obj, expected: str):
    assert script(obj).strip() == expected.strip()


def test_remap():
    @T.prim_func
    def block_with_remap_implicitly():
        for i0, i1, i2, i3, i4, i5 in T.grid(128, 128, 128, 128, 128, 128):
            with T.block("update"):
                v0 = T.axis.spatial(128, i0 + 1)
                v1 = T.axis.spatial(128, i1)
                v2 = T.axis.reduce(128, i2)
                v3 = T.axis.spatial(128, i3 - 1)
                v4 = T.axis.reduce(128, i4)
                v5 = T.axis.spatial(128, i5)
                pass

    @T.prim_func
    def block_with_remap_explicitly():
        for i0, i1, i2, i3, i4, i5 in T.grid(128, 128, 128, 128, 128, 128):
            with T.block("update"):
                v0 = T.axis.spatial(128, i0 + 1)
                v1, v2 = T.axis.remap("SR", [i1, i2])
                v3 = T.axis.spatial(128, i3 - 1)
                v4, v5 = T.axis.remap("RS", [i4, i5])
                pass

    expected_output = """@T.prim_func
def main() -> None:
    with T.block("root"):
        T.reads()
        T.writes()
        for i0, i1, i2, i3, i4, i5 in T.grid(128, 128, 128, 128, 128, 128):
            with T.block("update"):
                v0 = T.axis.spatial(128, i0 + 1)
                v1, v2 = T.axis.remap("SR", [i1, i2])
                v3 = T.axis.spatial(128, i3 - 1)
                v4, v5 = T.axis.remap("RS", [i4, i5])
                T.reads()
                T.writes()
                T.evaluate(0)"""
    _test(block_with_remap_implicitly, expected_output)
    _test(block_with_remap_explicitly, expected_output)


if __name__ == "__main__":
    tvm.testing.main()
