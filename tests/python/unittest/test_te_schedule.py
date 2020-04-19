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
import pytest
import tvm
from tvm import te
import pickle as pkl

def test_schedule_create():
    m = te.size_var('m')
    n = te.size_var('n')
    l = te.size_var('l')
    A = te.placeholder((m, l), name='A')
    B = te.placeholder((n, l), name='B')
    AA = te.compute((m, l), lambda i, j: A[i, j])
    T = te.compute((m, n, l), lambda i, j, k: AA(i, k) * B(j, k))
    s = te.create_schedule(T.op)
    s[AA].set_scope("shared")
    xo, xi = s[T].split(T.op.axis[0], factor=10)
    xi1, xi2 = s[T].split(xi, factor=2)
    s[AA].compute_at(s[T], xi1)
    xo, xi = s[AA].split(AA.op.axis[0], factor=10)
    s[T].reorder(xi2, xi1)
    assert T.op.axis[1] in s[T].leaf_iter_vars

    # save load json
    json_str = tvm.ir.save_json(s)
    s_loaded = tvm.ir.load_json(json_str)
    assert isinstance(s_loaded, tvm.te.schedule.Schedule)
    assert(str(s_loaded.outputs[0].body) == str(s.outputs[0].body))

    # pickle unpickle
    dump = pkl.dumps(s)
    s_loaded = pkl.loads(dump)
    assert isinstance(s_loaded, tvm.te.schedule.Schedule)
    assert(str(s_loaded.outputs[0].body) == str(s.outputs[0].body))


def test_reorder():
    m = te.size_var('m')
    A = te.placeholder((m,), name='A')
    T = te.compute(m, lambda i: A[i+1])

    s = te.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=10)
    xi1, xi2 = s[T].split(xi, factor=2)
    order = (xi2, xi1, xo)
    assert tuple(s[T].leaf_iter_vars) != order
    s[T].reorder(*order)
    assert tuple(s[T].leaf_iter_vars) == order
    try:
        # pass duplicate IterVar
        # must raise an error
        s[T].reorder(xi2, xi1, xi2)
        assert False
    except tvm.error.TVMError:
        pass

def test_split():
    m = te.size_var('m')
    A = te.placeholder((m,), name='A')
    T = te.compute((m,), lambda i: A[i])

    s = te.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=10)
    assert tuple(s[T].leaf_iter_vars) == (xo, xi)


def test_tile():
    m = te.size_var('m')
    n = te.size_var('n')
    A = te.placeholder((m, n), name='A')
    T = te.compute((m, n), lambda i, j: A[i, j])

    s = te.create_schedule(T.op)
    xo, yo, xi, yi = s[T].tile(T.op.axis[0], T.op.axis[1], x_factor=10, y_factor=5)
    assert tuple(s[T].leaf_iter_vars) == (xo, yo, xi, yi)


def test_fuse():
    m = te.size_var('m')
    n = te.size_var('n')
    A = te.placeholder((m, n), name='A')
    T = te.compute((m, n), lambda i, j: A[i, j])

    s = te.create_schedule(T.op)
    xo, yo, xi, yi = s[T].tile(T.op.axis[0], T.op.axis[1], x_factor=10, y_factor=5)
    fused = s[T].fuse(xo, yo)
    assert any(isinstance(x, tvm.te.schedule.Fuse) for x in s[T].relations)
    assert tuple(s[T].leaf_iter_vars) == (fused, xi, yi)

def test_fuse_with_split():
    m = te.size_var('m')
    n = te.size_var('n')
    A = te.placeholder((m, n), name='A')
    T = te.compute((m, n), lambda i, j: A[i, j])

    s = te.create_schedule(T.op)
    y = T.op.axis[1]
    xo, xi = s[T].split(T.op.axis[0], factor=10)
    fused = s[T].fuse(xi, y)
    assert any(isinstance(x, tvm.te.schedule.Fuse) for x in s[T].relations)
    assert tuple(s[T].leaf_iter_vars) == (xo, fused)

@pytest.mark.xfail
def test_fuse_with_out_of_order_axis():
    m = te.size_var('m')
    n = te.size_var('n')
    A = te.placeholder((m, n), name='A')
    T = te.compute((m, n), lambda i, j: A[i, j])

    s = te.create_schedule(T.op)
    y = T.op.axis[1]
    xo, xi = s[T].split(T.op.axis[0], factor=10)
    fused = s[T].fuse(xo, y) # should throw here

@pytest.mark.xfail
def test_fuse_with_out_of_order_axis_with_reorder():
    m = te.size_var('m')
    n = te.size_var('n')
    A = te.placeholder((m, n), name='A')
    T = te.compute((m, n), lambda i, j: A[i, j])

    s = te.create_schedule(T.op)
    y = T.op.axis[1]
    xo, xi = s[T].split(T.op.axis[0], factor=10)
    s[T].reorder(y, xo, xi)
    fused = s[T].fuse(y, xo) # should be ok

    s = te.create_schedule(T.op)
    y = T.op.axis[1]
    xo, xi = s[T].split(T.op.axis[0], factor=10)
    s[T].reorder(y, xo, xi)
    fused = s[T].fuse(y, xi) # should throw here

def test_singleton():
    print("test singleton")
    A = te.placeholder((), name='A')
    T = te.compute((), lambda : A() + 1)
    s = te.create_schedule(T.op)
    print("test singleton fin1")
    fused = s[T].fuse()
    assert any(isinstance(x, tvm.te.schedule.Singleton) for x in s[T].relations)
    assert tuple(s[T].leaf_iter_vars) == (fused,)
    dump = pkl.dumps(s)
    print("test singleton fin3")
    s_loaded = pkl.loads(dump)
    print("test singleton fin2")
    assert isinstance(s_loaded, tvm.te.schedule.Schedule)
    print("test singleton fin")

def test_vectorize():
    m = te.size_var('m')
    n = te.size_var('n')
    A = te.placeholder((m, n), name='A')
    T = te.compute((m, n), lambda i, j: A[i, j])

    s = te.create_schedule(T.op)
    xo, yo, xi, yi = s[T].tile(T.op.axis[0], T.op.axis[1], x_factor=10, y_factor=5)
    s[T].vectorize(yi)
    s[T].unroll(xi)
    UNROLL = tvm.te.schedule.IterVar.Unrolled
    VECTORIZE = tvm.te.schedule.IterVar.Vectorized
    assert s[T].iter_var_attrs[xi].iter_type == UNROLL
    assert s[T].iter_var_attrs[yi].iter_type == VECTORIZE

@pytest.mark.xfail
def test_vectorize_commreduce():
    V = te.placeholder((128,), name='V')
    ax = te.reduce_axis((0, 128), name='ax')
    O = te.compute((1,), lambda _: te.sum(V[ax], axis=[ax]))
    s = te.create_schedule(O.op)
    s[O].vectorize(ax) # should throw here

def test_pragma():
    m = 100
    A = te.placeholder((m,), name='A')
    T = te.compute((m,), lambda i: A[i])

    s = te.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=10)
    s[T].pragma(xo, "pragma1")
    s[T].pragma(xi, "vectorize")
    VECTORIZE = tvm.te.schedule.IterVar.Vectorized
    assert s[T].iter_var_attrs[xo].pragma_keys[0].value == "pragma1"
    assert s[T].iter_var_attrs[xi].iter_type == VECTORIZE


def test_rfactor():
    n = te.size_var('n')
    k1 = te.reduce_axis((0, n), name="k1")
    k2 = te.reduce_axis((0, n), name="k2")
    A = te.placeholder((n, n, n), name='A')
    B = te.compute((n, ), lambda i: te.sum(A[i, k1, k2], axis=[k1, k2]))
    # normal schedule
    s = te.create_schedule(B.op)
    BF = s.rfactor(B, k1)
    assert(tuple(BF.shape) == (n, n))
    assert(set(BF.op.body[0].axis) == set([k2]))
    assert(s[B].op.body[0].axis[0].dom.extent == n)
    assert(len(s[B].all_iter_vars) == 2)
    # schedule with splot
    s = te.create_schedule(B.op)
    ko, ki = s[B].split(k1, factor=4)
    xo, xi = s[B].split(B.op.axis[0], factor=8)
    BF = s.rfactor(B, ki)
    assert(BF.shape[0].value == 4)
    assert(BF.shape[1] == n)
    assert(BF.op.body[0].axis[0] ==  k2)
    assert(BF.op.body[0].axis[1].var ==  ko.var)
    assert(s[B].op.body[0].axis[0].dom.extent.value == 4)
    # schedule with factor_axis
    s = te.create_schedule(B.op)
    ko, ki = s[B].split(k1, factor=4)
    xo, xi = s[B].split(B.op.axis[0], factor=8)
    BF = s.rfactor(B, ki, 1)
    assert(n == BF.shape[0])
    assert(BF.shape[1].value == 4)
    assert(BF.op.body[0].axis[0] ==  k2)
    assert(BF.op.body[0].axis[1].var ==  ko.var)
    assert(s[B].op.body[0].axis[0].dom.extent.value == 4)

def test_tensor_intrin():
    n = 16
    x = te.placeholder((n,), name='x')
    y = te.placeholder((n,), name='y')
    z = te.compute(x.shape, lambda i: x[i] + y[i], name='z')
    def intrin_func(ins, outs):
        assert(isinstance(ins[0], tvm.te.schedule.Buffer))
        assert(ins[0].shape[0].value == n)
        return tvm.tir.call_packed("vadd", ins[0].data, outs[0].data, ins[0].shape[0])
    intrin = te.decl_tensor_intrin(z.op, intrin_func)
    assert intrin.op == z.op
    assert intrin.reduce_init is None
    assert tuple(intrin.inputs) == tuple(z.op.input_tensors)
    assert(intrin.buffers[0].shape[0].value == n)
    m = 32
    x = te.placeholder((m,), name='x')
    y = te.placeholder((m,), name='y')
    z = te.compute(x.shape, lambda i: x[i] + y[i], name='z')
    s = te.create_schedule(z.op)
    xo, xi = s[z].split(z.op.axis[0], factor=n)
    s[z].tensorize(xi, intrin)
    assert(s[z].iter_var_attrs[xi].tensor_intrin == intrin)
    assert(s[z].iter_var_attrs[xi].iter_type == tvm.te.schedule.IterVar.Tensorized)

def test_tensor_intrin_scalar_params():
    n = te.size_var("n")
    x = te.placeholder((n,), name='x')
    v = te.size_var("v")
    w = te.size_var("w")
    z = te.compute((n,), lambda i: x[i]*v + w, name='z')

    def intrin_func(ins, outs, sp):
        assert(isinstance(ins[0], tvm.te.schedule.Buffer))
        assert(ins[0].shape[0] == n)
        assert(sp[0] == v)
        assert(sp[1] == w)
        return tvm.tir.call_packed("hw_func", ins[0].data, outs[0].data, sp[0], sp[1])

    with tvm.target.build_config(offset_factor=1):
      intrin = te.decl_tensor_intrin(z.op, intrin_func, scalar_params=[v, w])
    assert intrin.op == z.op
    assert intrin.reduce_init is None
    assert tuple(intrin.inputs) == tuple(z.op.input_tensors)
    assert(intrin.buffers[0].shape[0] == n)
    assert tuple(intrin.scalar_params) == tuple((v, w))

    A = te.placeholder((10,10), name='A')
    # Pass scalar inputs to the TensorIntrin, interleaved with tensor inputs
    C = te.compute((10,10), lambda i, j: intrin(i*i, A[i, j], i+j), name="C")
    s = te.create_schedule(C.op)
    stmt = tvm.lower(s, [A, C])["main"].body
    assert isinstance(stmt.body.body, tvm.tir.Evaluate)
    assert len(stmt.body.body.value.args) == 5
    assert str(stmt.body.body.value.args[3]) == "(i*i)"
    assert str(stmt.body.body.value.args[4]) == "(i + j)"

if __name__ == "__main__":
    test_singleton()
    test_pragma()
    test_tensor_intrin()
    test_tensor_intrin_scalar_params()
    test_rfactor()
    test_schedule_create()
    test_reorder()
    test_tile()
    test_split()
    test_fuse()
    test_fuse_with_split()
    test_fuse_with_out_of_order_axis()
    test_fuse_with_out_of_order_axis_with_reorder()
    test_vectorize()
    test_vectorize_commreduce()
