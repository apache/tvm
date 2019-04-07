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

def test_scan():
    m = tvm.var("m")
    n = tvm.var("n")
    x = tvm.compute((m, n), lambda i, j: tvm.const(1, "float32"), name="x")
    s_state = tvm.placeholder((m, n))
    s_init = tvm.compute((1, n), lambda _, i: x[0, i], name="s_init")
    x_trans = tvm.compute((m, n), lambda i, j: x[i, j] + 1, name="x_trans")
    s_up1 = tvm.compute((m, n), lambda t, i: s_state[t - 1, i] + 1, name="up1")
    s_update = tvm.compute((m, n), lambda t, i: s_up1[t, i] + x_trans[t, i], name="update")
    s_scan = tvm.scan(s_init, s_update, s_state)

    def test_getbody():
        body = tvm.schedule.ScanGetBody(s_scan.op)
        assert set(body) == set([s_scan.op, s_update.op, s_up1.op])

    def test_attach_path():
        s = tvm.create_schedule(s_scan.op)
        s[x_trans].compute_at(s[s_update], s_update.op.axis[0])
        apath = tvm.schedule.CreateAttachPath(s)
        assert(tuple(apath[s_update.op]) == tuple([s_scan.op.scan_axis]))
        assert(tuple(apath[x_trans.op]) == tuple([s_update.op.axis[0], s_scan.op.scan_axis]))

    def test_fix_pt():
        body = tvm.schedule.ScanGetBody(s_scan.op)
        fxpt = tvm.schedule.ScanFixPointAnalysis(s_scan.op, body)
        assert(fxpt[s_scan.spatial_axis_[0]].value != 0)

def test_scan_fix_point():
    m = tvm.var("m")
    n = tvm.var("n")
    l = tvm.var("l")
    x = tvm.compute((l, m, n), lambda *i: tvm.const(1, "float32"), name="x")
    s_state = tvm.placeholder((l, m, n))
    s_init = tvm.compute((1, m, n), lambda _, i, j: x[0, i, j], name="s_init")

    def test_scan0():
        s_update = tvm.compute((l, m, n),
                               lambda t, i, j: x[t, j, i]  + s_state[t-1, i, j], name="update")
        s_scan = tvm.scan(s_init, s_update, s_state)
        body = tvm.schedule.ScanGetBody(s_scan.op)
        fxpt = tvm.schedule.ScanFixPointAnalysis(s_scan.op, body)
        assert(fxpt[s_scan.op.spatial_axis_[0]].value == 1)
        assert(fxpt[s_scan.op.spatial_axis_[1]].value == 1)

    def test_scan1():
        s_update = tvm.compute((l, m, n),
                               lambda t, i, j: x[t, j, i]  + s_state[t-1, j, i], name="update")
        s_scan = tvm.scan(s_init, s_update, s_state)
        body = tvm.schedule.ScanGetBody(s_scan.op)
        fxpt = tvm.schedule.ScanFixPointAnalysis(s_scan.op, body)
        assert(fxpt[s_scan.op.spatial_axis_[0]].value == 0)
        assert(fxpt[s_scan.op.spatial_axis_[1]].value == 0)

    def test_scan3_not_exact_reach():
        s_h1 = tvm.compute((l, n, m), lambda t, j, i: s_state[t-1, i, j], name="h1")
        s_h2 = tvm.compute((l, m, n), lambda t, i, j: s_state[t-1, i, 10] * 2, name="h1")
        s_update = tvm.compute((l, m, n), lambda t, i, j: s_h1[t, j, i] + s_h2[t, i, j], name="update")
        s_scan = tvm.scan(s_init, s_update, s_state)
        body = tvm.schedule.ScanGetBody(s_scan.op)
        fxpt = tvm.schedule.ScanFixPointAnalysis(s_scan.op)
        assert(fxpt[s_scan.op.spatial_axis_[0]].value == 1)
        assert(fxpt[s_scan.op.spatial_axis_[1]].value == 0)

    def test_scan4_reach_other():
        s_h1 = tvm.compute((l, n, m), lambda t, j, i: s_state[t-1, j, j], name="h1")
        s_h2 = tvm.compute((l, m, n), lambda t, i, j: s_state[t-1, i, j] * 2, name="h1")
        s_update = tvm.compute((l, m, n),
                               lambda t, i, j: s_h1[t, j, i] + s_h2[t, i, j], name="update")
        s_scan = tvm.scan(s_init, s_update, s_state)
        fxpt = tvm.schedule.ScanFixPointAnalysis(s_scan.op)
        assert(fxpt[s_scan.op.spatial_axis_[0]].value == 0)
        assert(fxpt[s_scan.op.spatial_axis_[1]].value == 0)

    def test_scan5_multi_output():
        m = tvm.var("m")
        n = tvm.var("n")
        x1 = tvm.placeholder((m, n))
        s1 = tvm.placeholder((m, n))
        x2 = tvm.placeholder((m, n))
        s2 = tvm.placeholder((m, n))
        s1_init = tvm.compute((1, n), lambda _, i: x1[0, i])
        s2_init = tvm.compute((1, n), lambda _, i: x2[0, i])
        s1_update = tvm.compute((m, n), lambda t, i: s1[t-1, i] +  x1[t, i])
        s2_update = tvm.compute((m, n), lambda t, i: x2[t, i] + s2[t-1,i])
        r0, r1 = tvm.scan([s1_init, s2_init],
                          [s1_update, s2_update],
                          [s1, s2])
        body = tvm.schedule.ScanGetBody(r0.op)
        fxpt = tvm.schedule.ScanFixPointAnalysis(r0.op)
        assert(fxpt[r1.op.spatial_axis_[0]].value == 1)

    test_scan0()
    test_scan1()
    test_scan3_not_exact_reach()
    test_scan4_reach_other()
    test_scan5_multi_output()

def test_create_read_graph():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j])
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3)

    g = tvm.schedule.CreateReadGraph([A2.op])

    assert g[A2.op][0] == A1
    assert g[A1.op][0] == A
    post_order = tvm.schedule.PostDFSOrder([A2.op], g)
    assert(post_order[0] == A.op)
    assert(post_order[1] == A1.op)


if __name__ == "__main__":
    test_scan()
    test_create_read_graph()
    test_scan_fix_point()
