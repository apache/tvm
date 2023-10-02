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
import tvm.relax
import tvm.testing

from tvm.script import ir as I, relax as R

from tvm.relax.transform import KillAfterLastUse


def test_basic():
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor([16, 32], "float32")):
            storage = R.memory.alloc_storage(R.shape([2048]), 0, "global", "uint8")
            y = R.memory.alloc_tensor(storage, 0, R.shape([16, 32]), "float32")
            _dummy = R.call_packed("add_tensors", [x, y], sinfo_args=(R.Tuple,))
            z = R.add(x, y)
            return z

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(x: R.Tensor([16, 32], "float32")):
            storage = R.memory.alloc_storage(R.shape([2048]), 0, "global", "uint8")
            y = R.memory.alloc_tensor(storage, 0, R.shape([16, 32]), "float32")
            _ = R.memory.kill_storage(storage)
            _dummy = R.call_packed("add_tensors", [x, y], sinfo_args=(R.Tuple,))
            z = R.add(x, y)
            _ = R.memory.kill_tensor(y)
            return z

    After = KillAfterLastUse()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_track_usage_across_trivial_rebindings():
    """To work around VM de-duplication of register usage"""

    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(w: R.Tensor([16, 32], "float32")):
            x = R.add(w, R.const(1, "float32"))
            y = x
            z = R.add(y, R.const(1, "float32"))
            return z

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(w: R.Tensor([16, 32], "float32")):
            x = R.add(w, R.const(1, "float32"))
            z = R.add(x, R.const(1, "float32"))
            _ = R.memory.kill_tensor(x)
            return z

    After = KillAfterLastUse()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_track_usage_across_trivial_rebindings_in_match_cast():
    """To work around VM de-duplication of register usage"""

    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(w: R.Tensor([16, 32], "float32")):
            x = R.add(w, R.const(1, "float32"))
            y = R.match_cast(x, R.Tensor([16, 32]))
            z = R.add(y, R.const(1, "float32"))
            return z

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(w: R.Tensor([16, 32], "float32")):
            x = R.add(w, R.const(1, "float32"))
            y = R.match_cast(x, R.Tensor([16, 32]))
            _ = R.memory.kill_tensor(x)
            z = R.add(y, R.const(1, "float32"))
            _ = R.memory.kill_tensor(y)
            return z

    After = KillAfterLastUse()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


if __name__ == "__main__":
    tvm.testing.main()
