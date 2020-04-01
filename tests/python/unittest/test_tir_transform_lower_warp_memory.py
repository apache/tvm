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
from tvm import te

def test_lower_warp_mem():
    m = 128
    A = te.placeholder((m,), name='A')
    B = te.compute((m,), lambda i: A[i] + 3, name='B')

    s = te.create_schedule(B.op)
    AA = s.cache_read(A, "warp", [B])
    xo, xi = s[B].split(B.op.axis[0], 64)
    xi0, xi1 = s[B].split(xi, factor=32)
    tx = te.thread_axis("threadIdx.x")
    s[B].bind(xi1, tx)
    s[B].bind(xo, te.thread_axis("blockIdx.x"))
    s[AA].compute_at(s[B], xo)
    xo, xi = s[AA].split(s[AA].op.axis[0], 32)
    s[AA].bind(xi, tx)

    f = tvm.lower(s, [A, B])
    fhost, fdevice = tvm.tir.ir_pass.SplitHostDevice(f)

    # temp adapter to convert loweredFunc to IRModule
    # to test passes in the new style.
    fname = fdevice.name
    mod = tvm.testing.LoweredFuncsToIRModule([fdevice])
    cuda_target = tvm.target.create("cuda")
    assert cuda_target.thread_warp_size == 32
    mod = tvm.IRModule.from_expr(mod[fname].with_attr("target", cuda_target))
    fdevice = tvm.tir.transform.LowerWarpMemory()(mod)["main"]
    assert(fdevice.body.body.value.value == "local")
    assert(fdevice.body.body.body.extents[0].value == 2)


if __name__ == "__main__":
    test_lower_warp_mem()
