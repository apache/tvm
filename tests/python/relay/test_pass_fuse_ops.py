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
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass
import tvm.testing


def test_fuse_simple():
    """Simple testcase."""

    def before():
        x = relay.var("x", shape=(10, 20))
        y = relay.add(x, relay.const(1, "float32"))
        z = relay.exp(y)
        w = relay.squeeze(z)
        return relay.Function([x], w)

    def expected():
        x = relay.var("p", shape=(10, 20))
        y = relay.add(x, relay.const(1, "float32"))
        z = relay.exp(y)
        w = relay.squeeze(z)
        f1 = relay.Function([x], w)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        x = relay.var("x", shape=(10, 20))
        y = relay.Call(f1, [x])
        return relay.Function([x], y)

    z = before()
    zz = run_opt_pass(z, transform.FuseOps(fuse_opt_level=2))
    zz = run_opt_pass(z, transform.FuseOps())
    after = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)


def test_conv2d_fuse():
    """Test fusion case of conv2d"""

    def before(dshape):
        x = relay.var("x", shape=dshape)
        x = relay.add(x, relay.const(1, "float32"))
        y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=16)
        # this is the next dominator.
        y1 = relay.add(relay.const(1, "float32"), y)
        y = relay.add(y, y1)
        # second path
        z2 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(1, 1), padding=(0, 0), channels=16)
        z3 = relay.nn.conv2d(y, relay.var("w3"), kernel_size=(3, 3), padding=(1, 1), channels=16)
        # add can only be fused to z1
        z = relay.add(z2, z3)
        return relay.Function(relay.analysis.free_vars(z), z)

    def expected(dshape):
        # segment 0
        x = relay.var("p0", shape=dshape)
        y = relay.add(x, relay.const(1, "float32"))
        f0 = relay.Function([x], y)
        f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        # segment 1
        x = relay.var("p0", shape=dshape)
        w = relay.var("p1")
        y = relay.nn.conv2d(x, w, kernel_size=(3, 3), padding=(1, 1), channels=16)
        y1 = relay.add(relay.const(1, "float32"), y)
        y = relay.add(y, y1)
        f1 = relay.Function([x, w], y)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        # segment 2
        x = relay.var("p0", shape=dshape)
        w = relay.var("p1")
        z2 = relay.nn.conv2d(x, w, kernel_size=(3, 3), padding=(1, 1), channels=16)
        f2 = relay.Function([x, w], z2)
        f2 = f2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        # segment 3
        x = relay.var("p0", shape=dshape)
        w = relay.var("p1")
        offset = relay.var("p2", shape=dshape)
        z3 = relay.nn.conv2d(x, w, kernel_size=(1, 1), padding=(0, 0), channels=16)
        z3 = relay.add(z3, offset)
        f3 = relay.Function([x, w, offset], z3)
        f3 = f3.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        # compose
        x = relay.var("x", shape=dshape)
        y = relay.Call(f0, [x])
        y = relay.Call(f1, [y, relay.var("w1")])
        z2 = relay.Call(f2, [y, relay.var("w3")])
        z3 = relay.Call(f3, [y, relay.var("w2"), z2])
        z = z3
        return relay.Function(relay.analysis.free_vars(z), z)

    dshape = (1, 16, 64, 64)
    z = before(dshape)
    zz = run_opt_pass(z, transform.FuseOps(fuse_opt_level=2))
    after = run_opt_pass(expected(dshape), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)


def test_concatenate():
    """Test fusion case involving concat op and Tuple node"""

    def before(dshape):
        x = relay.var("x", shape=dshape)
        pooled = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
        upsampled = relay.nn.upsampling(pooled, scale_h=2, scale_w=2, layout="NCHW")
        concat = relay.concatenate((upsampled, x), axis=1)
        out = relay.add(concat, relay.const(1, "float32"))
        return relay.Function(relay.analysis.free_vars(out), out)

    def expected(dshape):
        x = relay.var("x", shape=dshape)
        pooled = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
        f0 = relay.Function([x], pooled)
        f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        p0 = relay.var("p0", shape=(dshape[0], dshape[1], dshape[2] // 2, dshape[3] // 2))
        p1 = relay.var("p1", shape=dshape)
        upsampled = relay.nn.upsampling(p0, scale_h=2, scale_w=2, layout="NCHW")
        concat = relay.concatenate((upsampled, p1), axis=1)
        out = relay.add(concat, relay.const(1, "float32"))
        f1 = relay.Function([p0, p1], out)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("x", shape=dshape)
        y = relay.Call(f0, [x])
        z = relay.Call(f1, [y, x])
        return relay.Function([x], z)

    dshape = (1, 16, 64, 64)
    z = before(dshape)
    zz = run_opt_pass(z, transform.FuseOps(fuse_opt_level=0))
    assert not relay.analysis.free_vars(zz)
    zz = run_opt_pass(z, transform.FuseOps(fuse_opt_level=2))
    assert not relay.analysis.free_vars(zz)
    after = run_opt_pass(expected(dshape), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)


def test_tuple_root():
    """Test fusion case where Tuple node is the root in its group"""

    def before(dshape):
        x = relay.var("x", shape=dshape)
        pooled = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
        upsampled = relay.nn.upsampling(pooled, scale_h=2, scale_w=2, layout="NCHW")
        out = relay.Tuple((upsampled, x))
        return relay.Function(relay.analysis.free_vars(out), out)

    def expected(dshape):
        x = relay.var("x", shape=dshape)
        pooled = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
        f0 = relay.Function([x], pooled)
        f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        p0 = relay.var("p0", shape=(dshape[0], dshape[1], dshape[2] // 2, dshape[3] // 2))
        upsampled = relay.nn.upsampling(p0, scale_h=2, scale_w=2, layout="NCHW")
        f1 = relay.Function([p0], upsampled)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("x", shape=dshape)
        y = relay.Call(f0, [x])
        z = relay.Call(f1, [y])
        tup = relay.Tuple((z, x))
        return relay.Function([x], tup)

    dshape = (1, 16, 64, 64)
    z = before(dshape)
    zz = run_opt_pass(z, transform.FuseOps(fuse_opt_level=0))
    assert not relay.analysis.free_vars(zz)
    zz = run_opt_pass(z, transform.FuseOps(fuse_opt_level=2))
    assert not relay.analysis.free_vars(zz)
    after = run_opt_pass(expected(dshape), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)


def test_stop_fusion():
    def before(dshape):
        x = relay.var("x", shape=dshape)
        y = relay.add(x, relay.const(1, "float32"))
        y = relay.annotation.stop_fusion(y)
        z = relay.exp(y)
        return relay.Function([x], z)

    def expected(dshape):
        x = relay.var("p0", shape=dshape)
        y = relay.add(x, relay.const(1, "float32"))
        f1 = relay.Function([x], y)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("p01", shape=dshape)
        y = relay.exp(x)
        f2 = relay.Function([x], y)
        f2 = f2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("x", shape=dshape)
        y = relay.Call(f1, [x])
        z = relay.Call(f2, [y])
        return relay.Function([x], z)

    dshape = (10, 20)
    z = before(dshape)
    zz = run_opt_pass(z, transform.FuseOps())
    after = run_opt_pass(expected(dshape), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)


def test_fuse_myia_regression():
    def before(dshape, dtype):
        x = relay.var("x", shape=dshape, dtype=dtype)
        y = relay.var("y", shape=dshape, dtype=dtype)
        sb = relay.ScopeBuilder()
        with sb.if_scope(relay.op.greater(x, y)):
            sb.ret(relay.Function([], x))
        with sb.else_scope():
            sb.ret(relay.Function([], y))
        return relay.Function([x, y], relay.Call(sb.get(), []))

    def expected(dshape, dtype):
        x = relay.var("x", shape=dshape, dtype=dtype)
        y = relay.var("y", shape=dshape, dtype=dtype)
        sb = relay.ScopeBuilder()
        p1 = relay.var("p1", shape=dshape, dtype=dtype)
        p2 = relay.var("p2", shape=dshape, dtype=dtype)
        fused_gt = relay.Function([p1, p2], relay.op.greater(p1, p2))
        fused_gt = fused_gt.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        with sb.if_scope(fused_gt(x, y)):
            sb.ret(relay.Function([], x))
        with sb.else_scope():
            sb.ret(relay.Function([], y))
        return relay.Function([x, y], relay.Call(sb.get(), []))

    dshape = ()
    dtype = "int64"
    f = before(dshape, dtype)
    zz = run_opt_pass(f, transform.FuseOps())
    after = run_opt_pass(expected(dshape, dtype), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)


def test_fuse_tuple_get_elemwise():
    def before(dim):
        X = relay.var("X", shape=(1, dim))
        W = relay.var("W", shape=(3 * dim, dim))
        matmul = relay.nn.dense(X, W)
        splitted = relay.split(matmul, indices_or_sections=3, axis=1)
        out = relay.sigmoid(splitted[0]) + relay.tanh(splitted[1]) * relay.exp(splitted[2])
        return relay.Function([X, W], out)

    def expected(dim):
        p0 = relay.var("p0", shape=(1, dim))
        p1 = relay.var("p1", shape=(3 * dim, dim))
        matmul = relay.nn.dense(p0, p1)
        f0 = relay.Function([p0, p1], matmul)
        f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        p01 = relay.var("p01", shape=(1, 3 * dim))
        splitted = relay.split(p01, indices_or_sections=3, axis=1)
        out = relay.sigmoid(splitted[0]) + relay.tanh(splitted[1]) * relay.exp(splitted[2])
        f1 = relay.Function([p01], out)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        X = relay.var("X", shape=(1, dim))
        W = relay.var("W", shape=(3 * dim, dim))
        y = relay.Call(f0, [X, W])
        z = relay.Call(f1, [y])
        return relay.Function([X, W], z)

    dim = 10
    z = before(dim)
    zz = run_opt_pass(z, transform.FuseOps(fuse_opt_level=0))
    assert not relay.analysis.free_vars(zz)
    zz = run_opt_pass(z, transform.FuseOps(fuse_opt_level=2))
    assert not relay.analysis.free_vars(zz)
    after = run_opt_pass(expected(dim), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)


def test_tuple_get_root():
    def before(dim):
        X = relay.var("X", shape=(1, 3 * dim))
        W = relay.var("W", shape=(dim, dim))
        splitted = relay.split(X, indices_or_sections=3, axis=1)
        out = relay.nn.dense(splitted[0], W)
        return relay.Function([X, W], out)

    def expected(dim):
        p0 = relay.var("p0", shape=(1, 3 * dim))
        splitted = relay.split(p0, indices_or_sections=3, axis=1)
        out = splitted[0]
        f0 = relay.Function([p0], out)
        f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        p01 = relay.var("p01", shape=(1, dim))
        p1 = relay.var("p1", shape=(dim, dim))
        out = relay.nn.dense(p01, p1)
        f1 = relay.Function([p01, p1], out)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        X = relay.var("X", shape=(1, 3 * dim))
        W = relay.var("W", shape=(dim, dim))
        y = relay.Call(f0, [X])
        z = relay.Call(f1, [y, W])
        return relay.Function([X, W], z)

    dim = 10
    z = before(dim)
    zz = run_opt_pass(z, transform.FuseOps(fuse_opt_level=0))
    assert not relay.analysis.free_vars(zz)
    zz = run_opt_pass(z, transform.FuseOps(fuse_opt_level=2))
    assert not relay.analysis.free_vars(zz)
    after = run_opt_pass(expected(dim), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)


fuse0 = relay.transform.FuseOps(fuse_opt_level=0)
fuse2 = relay.transform.FuseOps(fuse_opt_level=2)


def test_tuple_intermediate():
    def before(x):
        inj = relay.squeeze(x)
        y1 = relay.add(inj, relay.const(1, "float32"))
        tmp = relay.squeeze(inj)
        tmp = relay.add(tmp, relay.const(1, "float32"))
        y2 = relay.add(tmp, relay.const(1, "float32"))
        y3 = relay.add(inj, relay.const(1, "float32"))
        concat = relay.concatenate((y1, y2, y3), axis=1)
        out_inj = relay.squeeze(concat)
        out = relay.add(out_inj, relay.const(1, "float32"))
        return relay.Function(relay.analysis.free_vars(out), out)

    def expected(p0):
        f0 = before(p0)
        f1 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        x = relay.var("x", shape=dshape)
        y = relay.Call(f1, [x])
        return relay.Function([x], y)

    dshape = (1, 16, 64, 64)
    x = relay.var("x", shape=dshape)
    orig = before(x)
    fuse0(tvm.IRModule.from_expr(orig))
    m = fuse2(tvm.IRModule.from_expr(orig))
    relay.build(m, "llvm")
    after = run_opt_pass(expected(x), transform.InferType())
    assert tvm.ir.structural_equal(m["main"], after)


def test_tuple_consecutive():
    def gen_intermediate_tuple(x):
        y1 = relay.add(x, relay.const(1, "float32"))
        y2 = relay.add(x, relay.const(1, "float32"))
        y3 = relay.add(x, relay.const(1, "float32"))
        concat = relay.concatenate((y1, y2, y3), axis=1)
        out = relay.add(concat, relay.const(1, "float32"))
        return out

    def gen_consecutive_tuple(x):
        y1 = gen_intermediate_tuple(x)
        y2 = gen_intermediate_tuple(x)
        y3 = gen_intermediate_tuple(x)
        concat = relay.concatenate((y1, y2, y3), axis=1)
        return concat

    def before(x):
        concat = gen_consecutive_tuple(x)
        pooled = relay.nn.max_pool2d(concat, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
        out = relay.add(pooled, relay.const(1, "float32"))
        out2 = relay.add(out, relay.const(1, "float32"))
        out_tup = relay.Tuple((out, out2))
        return relay.Function(relay.analysis.free_vars(out_tup), out_tup)

    def expected(dshape):
        p0 = relay.var("p0", shape=dshape)
        concat = gen_consecutive_tuple(p0)
        f0 = relay.Function([p0], concat)
        f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        p01 = relay.var("p01", shape=(1, dshape[1] * 9, dshape[2], dshape[3]))
        pooled = relay.nn.max_pool2d(p01, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
        out = relay.add(pooled, relay.const(1, "float32"))
        f1 = relay.Function([p01], out)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        p02 = relay.var("p02", shape=(1, dshape[1] * 9, dshape[2] // 2, dshape[3] // 2))
        out = relay.add(p02, relay.const(1, "float32"))
        f2 = relay.Function([p02], out)
        f2 = f2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("x", shape=dshape)
        y = relay.Call(f0, [x])
        z = relay.Call(f1, [y])
        z2 = relay.Call(f2, [z])

        return relay.Function([x], relay.Tuple((z, z2)))

    dshape = (1, 16, 64, 64)
    x = relay.var("x", shape=dshape)
    orig = before(x)
    fuse0(tvm.IRModule.from_expr(orig))
    m = fuse2(tvm.IRModule.from_expr(orig))
    relay.build(m, "llvm")
    after = run_opt_pass(expected(dshape), transform.InferType())
    assert tvm.ir.structural_equal(m["main"], after)


def test_inception_like():
    def conv(data):
        y = relay.nn.conv2d(data, relay.var("w"), kernel_size=(3, 3), padding=(1, 1), channels=16)
        return relay.nn.relu(data=y)

    def inception_like(data):
        c0 = conv(data)
        c1 = conv(data)
        return relay.concatenate((c0, c1), axis=1)

    def before(dshape):
        x = relay.var("x", shape=dshape)
        in1 = inception_like(x)
        in2 = inception_like(in1)
        return relay.Function(relay.analysis.free_vars(in2), in2)

    def expected(dshape):
        p0 = relay.var("p0", shape=dshape)
        c = conv(p0)
        f0 = relay.Function(relay.analysis.free_vars(c), c)
        f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        p01 = relay.var("p01", shape=dshape)
        c = conv(p01)
        f1 = relay.Function(relay.analysis.free_vars(c), c)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        p02 = relay.var("p02", shape=dshape)
        p12 = relay.var("p12", shape=dshape)
        concat1 = relay.concatenate((p02, p12), axis=1)
        f_concat1 = relay.Function([p02, p12], concat1)
        f_concat1 = f_concat1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        dshape2 = (dshape[0], dshape[1] * 2, dshape[2], dshape[3])

        p03 = relay.var("p03", shape=dshape2)
        c = conv(p03)
        f2 = relay.Function(relay.analysis.free_vars(c), c)
        f2 = f2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        p04 = relay.var("p04", shape=dshape2)
        c = conv(p04)
        f3 = relay.Function(relay.analysis.free_vars(c), c)
        f3 = f3.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        p05 = relay.var("p05", shape=dshape)
        p15 = relay.var("p15", shape=dshape)
        concat2 = relay.concatenate((p05, p15), axis=1)
        f_concat2 = relay.Function([p05, p15], concat2)
        f_concat2 = f_concat2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("x", shape=dshape)
        c1 = relay.Call(f0, [x, relay.var("w1")])
        c2 = relay.Call(f1, [x, relay.var("w2")])
        concat = relay.Call(f_concat1, [c1, c2])
        c3 = relay.Call(f2, [concat, relay.var("w3")])
        c4 = relay.Call(f3, [concat, relay.var("w4")])
        out = relay.Call(f_concat2, [c3, c4])

        return relay.Function(relay.analysis.free_vars(out), out)

    dshape = (1, 16, 64, 64)
    orig = before(dshape)
    fuse0(tvm.IRModule.from_expr(orig))
    m = fuse2(tvm.IRModule.from_expr(orig))
    relay.build(m, "llvm")
    after = run_opt_pass(expected(dshape), transform.InferType())
    assert tvm.ir.structural_equal(m["main"], after)


def test_fuse_parallel_injective():
    """Test fusing parallel injective ops to an elemwise op."""

    def before():
        x = relay.var("x", shape=(10, 20))
        y = relay.add(x, relay.const(1, "float32"))
        z = relay.squeeze(y)
        u = relay.transpose(y, axes=[0, 1])
        w = relay.left_shift(z, u)
        return relay.Function([x], w)

    def expected():
        x = relay.var("p", shape=(10, 20))
        y = relay.add(x, relay.const(1, "float32"))
        z = relay.squeeze(y)
        u = relay.transpose(y, axes=[0, 1])
        w = relay.left_shift(z, u)
        f1 = relay.Function([x], w)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        x = relay.var("x", shape=(10, 20))
        y = relay.Call(f1, [x])
        return relay.Function([x], y)

    z = before()
    zz = run_opt_pass(z, transform.FuseOps(fuse_opt_level=0))
    assert not relay.analysis.free_vars(zz)
    zz = run_opt_pass(z, transform.FuseOps(fuse_opt_level=2))
    assert not relay.analysis.free_vars(zz)
    after = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)


def test_immutable():
    """Verify the fusion pass won't change original module."""

    def before():
        x = relay.var("x", shape=(10, 20))
        y = relay.add(x, relay.const(1, "float32"))
        z = relay.exp(y)
        w = relay.squeeze(z)
        mod = tvm.IRModule()
        mod["main"] = relay.Function([x], w)
        return mod

    def expected():
        x = relay.var("p", shape=(10, 20))
        y = relay.add(x, relay.const(1, "float32"))
        z = relay.exp(y)
        w = relay.squeeze(z)
        f1 = relay.Function([x], w)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        x = relay.var("x", shape=(10, 20))
        y = relay.Call(f1, [x])
        mod = tvm.IRModule()
        mod["main"] = relay.Function([x], y)
        return mod

    mod = before()
    new_mod = transform.FuseOps(fuse_opt_level=2)(mod)
    assert tvm.ir.structural_equal(mod, before())
    assert tvm.ir.structural_equal(new_mod, expected())


def test_split():
    """Test that the result is well formed."""
    x = relay.var("x", shape=(6, 9))
    y = relay.split(x, 3).astuple()
    a = relay.TupleGetItem(y, 0)
    b = relay.TupleGetItem(y, 1)
    c = relay.TupleGetItem(y, 2)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], a + relay.RefRead(relay.RefCreate(b)) + c)
    mod = transform.FuseOps()(mod)


def test_fuse_max():
    """Test the constraint of number of nodes in op fusion."""

    def before(n):
        x = relay.var("x", shape=(10, 20))
        y = x
        for i in range(n):
            y = relay.exp(y)
        return relay.Function([x], y)

    def expected(n, max_fused_ops):
        x = relay.var("p", shape=(10, 20))
        y = x
        for i in range(max_fused_ops):
            y = relay.exp(y)
        f1 = relay.Function([x], y)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        x = relay.var("x", shape=(10, 20))
        z = relay.Call(f1, [x])
        xx = relay.var("pp", shape=(10, 20))
        yy = xx
        # it is assumed that there are two fused functions
        for i in range(n - max_fused_ops):
            yy = relay.exp(yy)
        f2 = relay.Function([xx], yy)
        f2 = f2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        zz = relay.Call(f2, [z])
        return relay.Function([x], zz)

    max_fused_ops = 256
    n = 300
    z = before(n)
    zz = run_opt_pass(z, transform.FuseOps(fuse_opt_level=2))
    zz = run_opt_pass(z, transform.FuseOps())
    after = run_opt_pass(expected(n, max_fused_ops), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)

    max_fused_ops = 10
    n = 20
    z = before(n)
    after = run_opt_pass(expected(n, max_fused_ops), transform.InferType())

    with tvm.transform.PassContext(config={"relay.FuseOps.max_depth": max_fused_ops}):
        zz = run_opt_pass(z, transform.FuseOps())

    assert tvm.ir.structural_equal(zz, after)


def test_fuse_take():
    """Test fusion case involving concat and take"""

    def before():
        shape = (tvm.tir.const(10, "int64"), tvm.tir.const(1, "int64"))
        x = relay.var("x", shape=shape)
        concat = relay.concatenate([x, x], axis=-1)
        out = relay.op.take(concat, indices=relay.const([0], dtype="int64"))
        return relay.Function(relay.analysis.free_vars(out), out)

    def expected():
        shape1 = (tvm.tir.const(10, "int64"), tvm.tir.const(1, "int64"))
        shape2 = (tvm.tir.const(1, "int64"),)
        x = relay.var("x", shape=shape1)
        p0 = relay.var("p0", shape=shape1)
        p1 = relay.var("p1", shape=shape2, dtype="int64")
        c = relay.const([0], dtype="int64")
        concat = relay.concatenate([p0, p0], axis=-1)
        out = relay.op.take(concat, indices=p1)

        f0 = relay.Function([p0, p1], out)
        f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        y = relay.Call(f0, [x, c])
        return relay.Function([x], y)

    orig = before()
    m = fuse2(tvm.IRModule.from_expr(orig))
    relay.build(m, "llvm")
    after = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(m["main"], after)


def test_fuse_gather_nd():
    """Test fusion case involving concat and gather_nd"""

    def before():
        shape = (tvm.tir.const(10, "int64"), tvm.tir.const(1, "int64"))
        x = relay.var("x", shape=shape)
        concat = relay.concatenate([x, x], axis=-1)
        out = relay.gather_nd(concat, indices=relay.expr.const([[0, 1], [1, 0]], dtype="int64"))
        return relay.Function(relay.analysis.free_vars(out), out)

    def expected():
        shape1 = (tvm.tir.const(10, "int64"), tvm.tir.const(1, "int64"))
        shape2 = (tvm.tir.const(2, "int64"), tvm.tir.const(2, "int64"))
        x = relay.var("x", shape=shape1)
        p0 = relay.var("p0", shape=shape1)
        p1 = relay.var("p1", shape=shape2, dtype="int64")
        c = relay.const([[0, 1], [1, 0]], dtype="int64")
        concat = relay.concatenate([p0, p0], axis=-1)
        out = relay.gather_nd(concat, indices=p1)

        f0 = relay.Function([p0, p1], out)
        f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        y = relay.Call(f0, [x, c])
        return relay.Function([x], y)

    orig = before()
    m = fuse2(tvm.IRModule.from_expr(orig))
    relay.build(m, "llvm")
    after = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(m["main"], after)


@tvm.testing.uses_gpu
def test_fuse_bcast_reduce_scalar():
    """Test fusion case with broadcast and reduction involving scalar"""

    def before():
        x = relay.var("x", shape=(), dtype="int32")
        less = relay.less(x, relay.const(10, dtype="int32"))
        z = relay.min(less)
        return relay.Function([x], z)

    def expected():
        p0 = relay.var("p0", shape=(), dtype="int32")
        less = relay.less(p0, relay.const(10, dtype="int32"))
        z0 = relay.min(less)
        f0 = relay.Function([p0], z0)
        f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("x", shape=(), dtype="int32")
        f = relay.Call(f0, [x])
        return relay.Function([x], f)

    orig = before()
    m = fuse2(tvm.IRModule.from_expr(orig))
    for tgt, ctx in tvm.testing.enabled_targets():
        relay.build(m, tgt)
    after = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(m["main"], after)


def test_fuse_max_diamond():
    def create_diamond(x, branch_len):
        x1 = x
        x2 = x
        for _ in range(branch_len):
            x1 = relay.exp(x1)
            x2 = relay.exp(x2)
        return relay.add(x1, x2)

    def before(branch_len, num_diamond):
        x = relay.var("x", shape=(10, 20))
        out = x
        for _ in range(num_diamond):
            out = create_diamond(out, branch_len)
        return relay.Function([x], out)

    def after(branch_len, num_diamond):
        def create_diamond_func(inp):
            inp_var = relay.var("p", shape=(10, 20))
            d = create_diamond(inp_var, branch_len)
            f = relay.Function([inp_var], d)
            f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
            return relay.Call(f, [inp])

        inp = relay.var("x", shape=(10, 20))
        out = inp
        for _ in range(num_diamond):
            out = create_diamond_func(out)
        return relay.Function([inp], out)

    branch_len = 5
    max_fused_ops = branch_len * 2 + 1  # the number of ops in one diamond
    num_diamond = 3

    with tvm.transform.PassContext(config={"relay.FuseOps.max_depth": max_fused_ops}):
        fused = run_opt_pass(before(branch_len, num_diamond), transform.FuseOps())

    expected = run_opt_pass(after(branch_len, num_diamond), transform.InferType())
    assert tvm.ir.structural_equal(fused, expected)


if __name__ == "__main__":
    test_fuse_simple()
    test_conv2d_fuse()
    test_concatenate()
    test_tuple_root()
    test_stop_fusion()
    test_fuse_myia_regression()
    test_fuse_tuple_get_elemwise()
    test_tuple_get_root()
    test_tuple_intermediate()
    test_tuple_consecutive()
    test_inception_like()
    test_fuse_parallel_injective()
    test_immutable()
    test_split()
    test_fuse_max()
    test_fuse_take()
    test_fuse_gather_nd()
    test_fuse_bcast_reduce_scalar()
    test_fuse_max_diamond()
