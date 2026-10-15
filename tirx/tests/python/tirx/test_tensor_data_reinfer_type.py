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

import tvm
from tvm import ir, tirx


def test_tensor_data_reinfer_type_from_rewritten_argument():
    global_tensor = tirx.decl_tensor((8,), "float32", name="global_tensor", scope="global")
    local_tensor = tirx.decl_tensor((8,), "float16", name="local_tensor", scope="local")
    stale_type = global_tensor.data.ty
    expected_type = local_tensor.data.ty

    call = ir.Call("tirx.tensor_data_ptr", [local_tensor], ty=stale_type)
    ir.assert_structural_equal(ir.reinfer_type(call), expected_type)
    ir.assert_structural_equal(call.ty, stale_type)

    with pytest.raises(TypeError):
        ir.reinfer_type(ir.Call("tirx.tensor_data_ptr", [ir.Var("not_a_tensor")]))


if __name__ == "__main__":
    tvm.testing.main()
