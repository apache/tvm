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
import tvm.testing
from tvm.script import tir as T, ir as I


@tvm.testing.requires_cuda
def test_split_host_device_func_attr():
    m = te.size_var("m")
    l = te.size_var("l")
    A = te.placeholder((m, l), name="A")

    A1 = te.compute((m, l), lambda i, j: A[i, j], name="A1")
    A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name="A2")

    s = te.create_schedule(A2.op)
    xo, xi = s[A2].split(A2.op.axis[0], factor=8)
    s[A2].bind(xo, te.thread_axis("blockIdx.x"))
    s[A1].compute_at(s[A2], xo)
    s[A1].set_scope("shared")

    mod = tvm.lower(s, [A, A2], name="f")

    cuda_target = tvm.target.Target("cuda")
    mod = tvm.tir.transform.Apply(
        lambda f: f.with_attr({"global_symbol": "test", "target": cuda_target})
    )(mod)
    fdevice = tvm.tir.transform.SplitHostDevice()(mod)["test_kernel0"]

    assert fdevice.attrs["global_symbol"] == "test_kernel0"
    assert fdevice.attrs["calling_conv"].value == 2
    assert fdevice.attrs["target"] == cuda_target
    assert fdevice.attrs["tir.is_global_func"].value


def test_ssa_across_entire_module():
    """The host and device functions should not share TIR vars

    Any arguments that are passed from the host to the device should
    be in terms of independent TIR variables.
    """

    @I.ir_module
    class before:
        @T.prim_func
        def main():
            T.func_attr({"global_symbol": "main", "target": T.target("cuda")})
            for i in range(16):
                T.attr(0, "device_scope", 0)
                for j in range(16):
                    T.evaluate(i)

    after = tvm.tir.transform.SplitHostDevice()(before)
    loop_var = after["main"].body.loop_var
    param_var = after["main_kernel0"].params[0]

    assert not loop_var.same_as(param_var)


if __name__ == "__main__":
    test_split_host_device_func_attr()
