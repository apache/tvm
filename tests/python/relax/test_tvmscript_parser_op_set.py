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

from typing import Optional, Union

import tvm
import tvm.script
import tvm.testing
from tvm import IRModule, relax
from tvm.script import relax as R


def _check(
    parsed: Union[relax.Function, IRModule],
    expect: Optional[Union[relax.Function, IRModule]],
):
    test = parsed.script(show_meta=True)
    roundtrip_mod = tvm.script.from_source(test)
    tvm.ir.assert_structural_equal(parsed, roundtrip_mod)
    if expect:
        tvm.ir.assert_structural_equal(parsed, expect)


def test_unique():
    @R.function
    def foo(
        x: R.Tensor((2, 3, 4), dtype="float32")
    ) -> R.Tuple(
        R.Tensor(dtype="float32", ndim=3),
        R.Tensor(dtype="int64", ndim=1),
        R.Tensor(dtype="int64", ndim=1),
    ):
        gv: R.Tuple(
            R.Tensor(dtype="float32", ndim=3),
            R.Tensor(dtype="int64", ndim=1),
            R.Tensor(dtype="int64", ndim=1),
        ) = R.unique(
            x, sorted=True, return_index=False, return_inverse=True, return_counts=True, axis=1
        )
        return gv

    x = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(
            relax.op.unique(x, sorted=True, return_inverse=True, return_counts=True, axis=1)
        )
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


if __name__ == "__main__":
    tvm.testing.main()
