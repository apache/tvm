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
from tvm import relay
from tvm.relay.transform import recast


def test_recast_simple():
    """Recast a single convolution operator."""

    def before():
        x = relay.var("x", shape=[8, 8, 8, 8])
        w = relay.var("w", shape=[8, 8, 3, 3])
        c = relay.nn.conv2d(x, w, padding=(1, 1), out_dtype="float32")
        return relay.Function([x, w], c)

    def expected():
        x = relay.var("x", shape=[8, 8, 8, 8])
        w = relay.var("w", shape=[8, 8, 3, 3])
        x_int = relay.cast(x, "int8")
        w_int = relay.cast(w, "int8")
        c = relay.nn.conv2d(x_int, w_int, padding=(1, 1), out_dtype="int32")
        c_float = relay.cast(c, "float32")
        return relay.Function([x, w], c_float)

    pre = before()
    post = recast(pre, "int8", "int32")
    expected = expected()
    assert tvm.ir.structural_equal(expected, post)


def test_recast_medium():
    """Recast a slightly larger graph."""

    def before():
        x = relay.var("x", shape=[8, 8, 8, 8])
        w = relay.var("w", shape=[8, 8, 3, 3])
        c = relay.nn.conv2d(x, w, padding=(1, 1), out_dtype="float32")
        w2 = relay.var("w2", shape=[8, 8, 3, 3])
        c2 = relay.nn.conv2d(c, w2, padding=(1, 1), out_dtype="float32")
        return relay.Function([x, w, w2], c2)

    def expected():
        x = relay.var("x", shape=[8, 8, 8, 8])
        w = relay.var("w", shape=[8, 8, 3, 3])
        x_int = relay.cast(x, "int8")
        w_int = relay.cast(w, "int8")
        c = relay.nn.conv2d(x_int, w_int, padding=(1, 1), out_dtype="int32")
        c_float = relay.cast(c, "float32")
        w2 = relay.var("w2", shape=[8, 8, 3, 3])
        w2_int = relay.cast(w2, "int8")
        c_float_int = relay.cast(c_float, "int8")
        c2 = relay.nn.conv2d(c_float_int, w2_int, padding=(1, 1), out_dtype="int32")
        c2_float = relay.cast(c2, "float32")
        return relay.Function([x, w, w2], c2_float)

    pre = before()
    post = recast(pre, "int8", "int32")
    expected = expected()
    assert tvm.ir.structural_equal(expected, post)


def test_recast_skip():
    """Recast a graph using skip layers."""

    def before():
        x = relay.var("x", shape=[8, 8, 8, 8])
        w = relay.var("w", shape=[8, 8, 3, 3])
        c = relay.nn.conv2d(x, w, padding=(1, 1), out_dtype="float32")
        w2 = relay.var("w2", shape=[8, 8, 3, 3])
        c2 = relay.nn.conv2d(c, w2, padding=(1, 1), out_dtype="float32")
        return relay.Function([x, w, w2], c2)

    def expected():
        x = relay.var("x", shape=[8, 8, 8, 8])
        w = relay.var("w", shape=[8, 8, 3, 3])
        c = relay.nn.conv2d(x, w, padding=(1, 1), out_dtype="float32")
        w2 = relay.var("w2", shape=[8, 8, 3, 3])
        w2_int = relay.cast(w2, "int8")
        c_int = relay.cast(c, "int8")
        c2 = relay.nn.conv2d(c_int, w2_int, padding=(1, 1), out_dtype="int32")
        c2_float = relay.cast(c2, "float32")
        return relay.Function([x, w, w2], c2_float)

    pre = before()
    post = recast(pre, "int8", "int32", skip_layers=[0])
    expected = expected()
    assert tvm.ir.structural_equal(expected, post)


def test_recast_concat():
    def before():
        x = relay.var("x", shape=[1, 4])
        y = relay.var("y", shape=[1, 4])
        t = relay.Tuple([x, y])
        c = relay.op.concatenate(t, axis=1)
        return relay.Function([x, y], c)

    def expected():
        xv = relay.var("x", shape=[1, 4])
        yv = relay.var("y", shape=[1, 4])
        x = relay.cast(xv, "float16")
        y = relay.cast(yv, "float16")
        t = relay.Tuple([x, y])
        c = relay.op.concatenate(t, axis=1)
        c = relay.cast(c, "float32")
        return relay.Function([xv, yv], c)

    pre = before()
    post = recast(pre, "float16", "float32", ops=["concatenate"])
    expected = expected()
    assert tvm.ir.structural_equal(expected, post)


def test_recast_relu():
    """Recast a ReLU operator which does not have attributes."""

    def before():
        x = relay.var("x", shape=[8, 8, 8, 8])
        w = relay.var("w", shape=[8, 8, 3, 3])
        c = relay.nn.conv2d(x, w, padding=(1, 1), out_dtype="float32")
        r = relay.nn.relu(c)
        return relay.Function([x, w], r)

    def expected():
        x = relay.var("x", shape=[8, 8, 8, 8])
        w = relay.var("w", shape=[8, 8, 3, 3])
        x_fp16 = relay.cast(x, "float16")
        w_fp16 = relay.cast(w, "float16")
        c = relay.nn.conv2d(x_fp16, w_fp16, padding=(1, 1), out_dtype="float16")
        c_float32 = relay.cast(c, "float32")
        c_float16 = relay.cast(c_float32, "float16")
        r = relay.nn.relu(c_float16)
        r_float32 = relay.cast(r, "float32")
        return relay.Function([x, w], r_float32)

    pre = before()
    post = recast(pre, "float16", "float16", ops=["nn.conv2d", "nn.relu"])
    expected = expected()
    assert tvm.ir.structural_equal(expected, post)


if __name__ == "__main__":
    test_recast_simple()
    test_recast_medium()
    test_recast_skip()
    test_recast_concat()
    test_recast_relu()
