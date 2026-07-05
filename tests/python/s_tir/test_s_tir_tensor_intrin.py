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
from tvm.s_tir.tensor_intrin.cuda import get_wmma_load_intrin, get_wmma_store_intrin
from tvm.s_tir.tensor_intrin.metal import (
    get_simdgroup_load_intrin,
    get_simdgroup_store_intrin,
)


def test_matrix_load_store_access_ptr_types():
    implementations = [
        get_wmma_load_intrin(16, 16, 16, "float16", "shared", False, False)[1],
        get_wmma_store_intrin(16, 16, 16, "float16", "shared")[1],
        get_simdgroup_load_intrin("float16", "shared")[1],
        get_simdgroup_store_intrin("float16", "shared")[1],
    ]
    access_ptr_op = tvm.ir.Op.get("tirx.tvm_access_ptr")

    for implementation in implementations:
        access_ptr_calls = []

        def collect_access_ptr(node):
            if getattr(node, "op", None) is not None and node.op.same_as(access_ptr_op):
                access_ptr_calls.append(node)

        tvm.tirx.stmt_functor.post_order_visit(implementation.body, collect_access_ptr)

        assert len(access_ptr_calls) == 1
        access_ptr = access_ptr_calls[0]
        tvm.ir.assert_structural_equal(access_ptr.ty, access_ptr.args[1].ty)


if __name__ == "__main__":
    tvm.testing.main()
