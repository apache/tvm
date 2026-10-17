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

from tvm_tirx.ir import Call, Evaluate, Op, Stmt


def seek_kernel_replace_point(stmt: Stmt, body: Stmt) -> Stmt:
    """Replace the kernel replacement point in ``stmt`` with ``body``."""

    def replace_evaluate(op: Evaluate):
        value = op.value
        if isinstance(value, Call) and value.op.same_as(Op.get("tirx.kernel_replace_point")):
            return body
        return op

    return tvm_ffi.structural_map(stmt, (Evaluate, replace_evaluate), order="post")
