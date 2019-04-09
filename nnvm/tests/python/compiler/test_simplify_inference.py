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
"""Unittest cases for simplify batch_norm"""
import nnvm
from nnvm import symbol as sym
from nnvm.compiler import graph_util, graph_attr

def test_simplify_batchnorm():
    def simple_bn(x, gamma, beta, moving_mean, moving_var,
                  axis=1, epsilon=1e-5, shape=None):
        # expect = (x - moving_mean) / sym.sqrt(moving_var + eps) * gamma + beta
        scale = sym.elemwise_mul(1 / sym.sqrt(moving_var + epsilon), gamma)
        shift = sym.elemwise_add(
            sym.elemwise_mul(sym.negative(moving_mean), scale), beta)
        # for 2D
        num_newaxis=len(shape) - axis - 1
        if num_newaxis:
            scale = sym.expand_dims(scale, axis=1, num_newaxis=num_newaxis)
            shift = sym.expand_dims(shift, axis=1, num_newaxis=num_newaxis)
        return x * scale + shift


    # Before simplify
    def check(dim, axis, nstep):
        eps = 0.01
        x = sym.Variable("x") + 1
        beta = sym.Variable("beta")
        gamma = sym.Variable("gamma")
        moving_var = sym.Variable("moving_var")
        moving_mean = sym.Variable("moving_mean")
        y1, y2 = x, sym.Variable("xx") + 1
        ishape = {"x": tuple(10 for i in range(dim))}
        for i in range(nstep):
            y1 = sym.batch_norm(
                y1 + 1, gamma, beta, moving_mean, moving_var, epsilon=eps, axis=axis)
            y1 = sym.dropout(y1)
            y2 = simple_bn(y2 + 1, gamma, beta, moving_mean, moving_var,
                           epsilon=eps, axis=axis, shape=ishape["x"])
        g = nnvm.graph.create(y1)
        g2 = nnvm.graph.create(y2)
        graph_attr.set_shape_inputs(g, ishape)
        g1 = g.apply("InferShape").apply("SimplifyInference")
        # assert graph equals as expected
        graph_util.check_graph_equal(g1, g2)

    check(2, 1, 1)
    check(4, 0, 3)
    check(4, 1, 2)

if __name__ == "__main__":
    test_simplify_batchnorm()
