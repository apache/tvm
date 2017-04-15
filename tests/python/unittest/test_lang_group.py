"""Test group effect"""
import tvm

def test_scan_group():
    m = tvm.var("m")
    n = tvm.var("n")
    x = tvm.compute((m, n), lambda i, j: tvm.const(1, "float32"), name="x")
    s_state = tvm.placeholder((m, n))
    s_init = tvm.compute((1, n), lambda _, i: x[0, i])

    s_update1 = tvm.compute((m, n), lambda t, i: s_state[t-1, i] + x[t, i])
    s_update2 = tvm.compute((m, n), lambda t, i: s_update1[t, i] + 1)
    s_update3 = tvm.compute((m, n), lambda t, i: s_update2[t, i] + 1)
    res = tvm.scan(s_init, s_update3, s_state, inputs=x)

    s = tvm.create_schedule(res.op)
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
    except tvm.TVMError:
        pass

def test_compute_group():
    m = tvm.var("m")
    n = tvm.var("n")
    x = tvm.compute((m, n), lambda i, j: tvm.const(1, "float32"), name="x")
    x1 = tvm.compute(x.shape, lambda *i: x(*i) + 1, name="x1")
    x2 = tvm.compute(x.shape, lambda *i: x1(*i) + 2, name="x2")
    s = tvm.create_schedule(x2.op)
    g = s.create_group(outputs=x1, inputs=x, include_inputs=True)
    assert s[x1].group == g
    assert s[x].group == g
    g.compute_at(s[x2], x2.op.axis[1])
    assert g.attach_stage == s[x2]
    assert g.num_child_stages == 2

def test_nest_group():
    m = tvm.var("m")
    n = tvm.var("n")
    x = tvm.compute((m, n), lambda i, j: tvm.const(1, "float32"), name="x")
    x1 = tvm.compute(x.shape, lambda *i: x(*i) + 1, name="x1")
    x2 = tvm.compute(x.shape, lambda *i: x1(*i) + 2, name="x2")
    s = tvm.create_schedule(x2.op)
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
