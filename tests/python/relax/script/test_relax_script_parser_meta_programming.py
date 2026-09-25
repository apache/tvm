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


"""RELAX parser integration for meta programming."""

from __future__ import annotations

from typing import TypeVar

from tvm import ir
from tvm.script import ir as I
from tvm.script import relax as R


def test_return_annotation_keeps_local_symbols_and_unused_captures():
    n = ir.Var("n", "int64")
    unused = TypeVar("unused", bound=int)

    @R.function
    def main(x: R.Tensor((n,), "float32")) -> R.Tensor(
        ((lambda local: local if I.constexpr(True) else unused)(n),), "float32"
    ):
        return x

    assert main.ret_ty.shape[0].same_as(n)
