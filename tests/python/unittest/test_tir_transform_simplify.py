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

def test_stmt_simplify():
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    C = ib.pointer("float32", name="C")
    n = te.size_var("n")
    with ib.for_range(0, n, name="i") as i:
        with ib.if_scope(i < 12):
            A[i] = C[i]

    body = tvm.tir.LetStmt(n, 10, ib.get())
    mod = tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([A, C, n], body))
    body = tvm.tir.transform.Simplify()(mod)["main"].body
    assert isinstance(body.body, tvm.tir.Store)


def test_thread_extent_simplify():
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    C = ib.pointer("float32", name="C")
    n = te.size_var("n")
    tx = te.thread_axis("threadIdx.x")
    ty = te.thread_axis("threadIdx.y")
    ib.scope_attr(tx, "thread_extent", n)
    ib.scope_attr(tx, "thread_extent", n)
    ib.scope_attr(ty, "thread_extent", 1)
    with ib.if_scope(tx + ty < 12):
        A[tx] = C[tx + ty]
    body = tvm.tir.LetStmt(n, 10, ib.get())
    mod = tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([A, C, n], body))
    body = tvm.tir.transform.Simplify()(mod)["main"].body
    assert isinstance(body.body.body.body, tvm.tir.Store)


def test_basic_likely_elimination():
    n = te.size_var('n')
    X = te.placeholder(shape=(n,), name="x")
    W = te.placeholder(shape=(n + 1,), dtype="int32", name="w")

    def f(i):
        start = W[i]
        extent = W[i+1] - W[i]
        rv = te.reduce_axis((0, extent))
        return te.sum(X[rv + start], axis=rv)
    Y = te.compute(X.shape, f, name="y")
    s = te.create_schedule([Y.op])
    stmt = tvm.lower(s, [X, W, Y], simple_mode=True)
    assert('if' not in str(stmt))

def test_complex_likely_elimination():
    def cumsum(X):
        """
        Y[i] = sum(X[:i])
        """
        (m, ) = X.shape
        s_state = te.placeholder((m + 1, ), dtype="int32", name="state")
        s_init = te.compute((1, ), lambda _: tvm.tir.const(0, "int32"))
        s_update = te.compute((m + 1, ), lambda l: s_state[l - 1] + X[l - 1])
        return tvm.te.scan(s_init, s_update, s_state, inputs=[X], name="cumsum")

    def sparse_lengths_sum(data, indices, lengths):
        oshape = list(data.shape)
        oshape[0] = lengths.shape[0]
        length_offsets = cumsum(lengths)

        def sls(n, d):
            gg = te.reduce_axis((0, lengths[n]))
            indices_idx = length_offsets[n] + gg
            data_idx = indices[indices_idx]
            data_val = data[data_idx, d]
            return te.sum(data_val, axis=gg)

        return te.compute(oshape, sls)

    m, n, d, i, l = te.size_var('m'), te.size_var('n'), te.size_var('d'),\
                    te.size_var('i'), te.size_var('l')
    data_ph = te.placeholder((m, d * 32), name="data")
    indices_ph = te.placeholder((i,), name="indices", dtype="int32")
    lengths_ph = te.placeholder((n,), name="lengths", dtype="int32")
    Y = sparse_lengths_sum(data_ph, indices_ph, lengths_ph)
    s = te.create_schedule([Y.op])
    (n, d) = s[Y].op.axis
    (do, di) = s[Y].split(d, factor=32)
    (gg,) = s[Y].op.reduce_axis
    s[Y].reorder(n, do, gg, di)
    s[Y].vectorize(di)
    stmt = tvm.lower(s, [data_ph, indices_ph, lengths_ph, Y], simple_mode=True)
    assert('if' not in str(stmt))

if __name__ == "__main__":
    test_stmt_simplify()
    test_thread_extent_simplify()
    test_basic_likely_elimination()
    test_complex_likely_elimination()
