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
from tvm.relax.transform import LegalizeOps
from tvm.script import ir as I
from tvm.script import relax as R


def test_allreduce():
    # fmt: off
    @tvm.script.ir_module
    class AllReduce:
        @R.function
        def main(x: R.Tensor((10, 10), "float32"))  -> R.Tensor((10, 10), "float32"):
            gv0: R.Tensor((10, 10), "float32") = R.ccl.allreduce(x, "sum")
            gv1: R.Tensor((10, 10), "float32") = R.ccl.allreduce(x, "prod")
            gv2: R.Tensor((10, 10), "float32") = R.ccl.allreduce(x, "min")
            gv3: R.Tensor((10, 10), "float32") = R.ccl.allreduce(x, "max")
            gv4: R.Tensor((10, 10), "float32") = R.ccl.allreduce(x, "avg")
            return x

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
            gv0: R.Tensor((10, 10), dtype="float32") = R.call_dps_packed("runtime.disco.allreduce", [x, R.shape([0])], out_sinfo=R.Tensor((10, 10), dtype="float32"))
            gv1: R.Tensor((10, 10), dtype="float32") = R.call_dps_packed("runtime.disco.allreduce", [x, R.shape([1])], out_sinfo=R.Tensor((10, 10), dtype="float32"))
            gv2: R.Tensor((10, 10), dtype="float32") = R.call_dps_packed("runtime.disco.allreduce", [x, R.shape([2])], out_sinfo=R.Tensor((10, 10), dtype="float32"))
            gv3: R.Tensor((10, 10), dtype="float32") = R.call_dps_packed("runtime.disco.allreduce", [x, R.shape([3])], out_sinfo=R.Tensor((10, 10), dtype="float32"))
            gv4: R.Tensor((10, 10), dtype="float32") = R.call_dps_packed("runtime.disco.allreduce", [x, R.shape([4])], out_sinfo=R.Tensor((10, 10), dtype="float32"))
            return x
    # fmt: on

    mod = LegalizeOps()(AllReduce)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_broadcast_from_zero():
    # fmt: off
    @tvm.script.ir_module
    class BroadcastFromZero:
        @R.function
        def main(x: R.Tensor((10, 10), "float32"))  -> R.Tensor((10, 10), "float32"):
            gv0: R.Tensor((10, 10), "float32") = R.ccl.broadcast_from_worker0(x)
            return x

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
            gv0: R.Tensor((10, 10), dtype="float32") = R.call_dps_packed("runtime.disco.broadcast_from_worker0", x, out_sinfo=R.Tensor((10, 10), dtype="float32"))
            return x
    # fmt: on

    mod = LegalizeOps()(BroadcastFromZero)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_scatter_from_worker0():
    # fmt: off
    @tvm.script.ir_module
    class ScatterFromWorker0:
        @R.function
        def main(x: R.Tensor((10, 10), "float32"))  -> R.Tensor((5, 10), "float32"):
            gv0: R.Tensor((5, 10), "float32") = R.ccl.scatter_from_worker0(x, 2)
            return gv0

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((5, 10), dtype="float32"):
            gv0: R.Tensor((5, 10), dtype="float32") = R.call_dps_packed("runtime.disco.scatter_from_worker0", x, out_sinfo=R.Tensor((5, 10), dtype="float32"))
            return gv0
    # fmt: on

    mod = LegalizeOps()(ScatterFromWorker0)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
