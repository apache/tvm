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
from tvm.script import ir as I
from tvm.script import tir as T


def test_host_func():
    """Test that host functions are not split."""
    # te schedule copied from test_tir_transform_split_host_device.py
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

    assert len(mod.get_global_vars()) == 1, """Before split, expected 1 global function."""
    mod = tvm.tir.transform.Apply(
        lambda f: f.with_attr(
            {
                "global_symbol": "test",
                "target": tvm.target.Target("cuda"),
                "tir.is_host_func": True,
            }
        )
    )(mod)
    mod = tvm.tir.transform.SplitHostDevice()(mod)
    assert len(mod.get_global_vars()) == 1, """Expected host function not to be splited."""


if __name__ == "__main__":
    test_host_func()
