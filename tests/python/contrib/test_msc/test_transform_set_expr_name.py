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

import tvm.testing
from tvm.relay import testing
from tvm.relay.expr_functor import ExprVisitor

from tvm.relax.testing import nn
from tvm.relax import PyExprVisitor

from tvm.contrib.msc.core import _ffi_api
from tvm.contrib.msc.core import transform as msc_transform


class RelayChecker(ExprVisitor):
    """Check if name as span attribute is setted."""

    def check(self, expr):
        self._missing_exprs = []
        super.visit(expr)
        assert len(self._missing_exprs) == 0, "Missing {} names".format(len(self._missing_exprs))

    def visit(self, expr):
        super().visit(expr)
        name = _ffi_api.SpanGetAttr(expr.span, "name")
        if not name:
            self._missing_exprs.append(expr)


class RelaxChecker(PyExprVisitor):
    """Check if name as span attribute is setted."""

    def check(self, expr):
        self._missing_exprs = []
        super.visit(expr)
        assert len(self._missing_exprs) == 0, "Missing {} names".format(len(self._missing_exprs))

    def visit_binding(self, binding):
        super().visit_binding(binding)
        name = _ffi_api.SpanGetAttr(binding.value.span, "name")
        if not name:
            self._missing_exprs.append(binding.value)

    def visit_constant_(self, op):
        super().visit_constant_(op)
        name = _ffi_api.SpanGetAttr(op.span, "name")
        if not name:
            self._missing_exprs.append(op)


def test_relay():
    mod, _ = testing.resnet.get_workload(num_layers=50, batch_size=1, dtype="float32")
    mod = msc_transform.SetExprName(as_relax=False)(mod)
    print("mod " + str(mod))
    RelayChecker().check(mod["main"])


def test_relax():
    builder = tvm.relax.BlockBuilder()

    # a symbolic variable to represent minibatch size
    n = tvm.tir.Var("n", "int64")
    input_size = 784
    hidden_sizes = [128, 32]
    output_size = 10

    # build a three linear-layer neural network for a classification task
    with builder.function("main"):
        model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.LogSoftmax(),
        )
        data = nn.Placeholder((n, input_size), name="data")
        output = model(data)
        params = [data] + model.parameters()
        builder.emit_func_output(output, params=params)

    # get and print the IRmodule being built
    mod = builder.get()
    mod = msc_transform.SetExprName()(mod)
    print("mod " + str(mod))
    RelaxChecker().check(mod["main"])


if __name__ == "__main__":
    # tvm.testing.main()
    test_relay()
