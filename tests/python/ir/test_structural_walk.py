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

import tvm_ffi

import tvm_tirx.testing
from tvm_tirx import tirx


def test_mixed_iter_type_detection_interrupts_walk():
    spatial = tirx.Var("spatial", "int32")
    reduction = tirx.Var("reduction", "int32")
    unreachable = tirx.Var("unreachable", "int32")
    visited = []

    def detect(var):
        visited.append(var)
        if var.same_as(reduction):
            return tvm_ffi.VisitInterrupt()
        return None

    result = tvm_ffi.structural_walk(
        (spatial + reduction) + unreachable,
        (tirx.Var, detect),
        order="post",
    )

    assert isinstance(result, tvm_ffi.VisitInterrupt)
    assert any(var.same_as(spatial) for var in visited)
    assert any(var.same_as(reduction) for var in visited)
    assert not any(var.same_as(unreachable) for var in visited)


if __name__ == "__main__":
    tvm_tirx.testing.main()
