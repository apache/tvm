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
"""Test group effect"""
import tvm
from tvm import te


def test_scan_group():
    m = te.size_var("m")
    n = te.size_var("n")
    x = te.compute((m, n), lambda i, j: tvm.tir.const(1, "float32"), name="x")
    s_state = te.placeholder((m, n))
    s_init = te.compute((1, n), lambda _, i: x[0, i])

    s_update1 = te.compute((m, n), lambda t, i: s_state[t - 1, i] + x[t, i])
    s_update2 = te.compute((m, n), lambda t, i: s_update1[t, i] + 1)
    s_update3 = te.compute((m, n), lambda t, i: s_update2[t, i] + 1)
    res = tvm.te.scan(s_init, s_update3, s_state, inputs=x)

    s = te.create_schedule(res.op)
    assert s[s_update1].group is not None
    assert s[s_update2].group == s[s_update1].group
    # Assign within group, is valid
    s[s_update1].compute_at(s[s_update2], s_update2.op.axis[1])
    # create a new group, for [s_update2 and s_update1]
    g2 = s.create_group(outputs=s_update2, inputs=[s_state, x])
    assert g2.group is not None
    assert g2.group == s[s_update3].group
    assert s[s_update2].group == g2
    assert s[s_update1].group == g2
    g2.compute_at(s[s_update3], s_update3.op.axis[1])
    assert g2.attach_stage == s[s_update3]
    try:
        # compute outside group error.
        s[s_update2].compute_at(s[s_init], s_init.op.axis[0])
        assert False
    except tvm.error.TVMError:
        pass


def test_compute_group():
    m = te.size_var("m")
    n = te.size_var("n")
    x = te.compute((m, n), lambda i, j: tvm.tir.const(1, "float32"), name="x")
    x1 = te.compute(x.shape, lambda *i: x(*i) + 1, name="x1")
    x2 = te.compute(x.shape, lambda *i: x1(*i) + 2, name="x2")
    s = te.create_schedule(x2.op)
    g = s.create_group(outputs=x1, inputs=x, include_inputs=True)
    assert s[x1].group == g
    assert s[x].group == g
    g.compute_at(s[x2], x2.op.axis[1])
    assert g.attach_stage == s[x2]
    assert g.num_child_stages == 2


def test_nest_group():
    m = te.size_var("m")
    n = te.size_var("n")
    x = te.compute((m, n), lambda i, j: tvm.tir.const(1, "float32"), name="x")
    x1 = te.compute(x.shape, lambda *i: x(*i) + 1, name="x1")
    x2 = te.compute(x.shape, lambda *i: x1(*i) + 2, name="x2")
    s = te.create_schedule(x2.op)
    g1 = s.create_group(outputs=x1, inputs=x)
    g2 = s.create_group(outputs=x1, inputs=x, include_inputs=True)
    assert set(s.groups) == set([g1, g2])
    assert s[x].group == g2
    assert s[x1].group == g1
    assert g1.group == g2
    assert g2.num_child_stages == 2
    assert g1.num_child_stages == 1


if __name__ == "__main__":
    test_nest_group()
    test_compute_group()
    test_scan_group()
