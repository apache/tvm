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

""" Test MSC basic Pass. """

import tvm.testing
from tvm.relax.frontend.torch import from_fx
from tvm.relax import PyExprVisitor

from tvm.relay import testing
from tvm.relay.expr_functor import ExprVisitor
from tvm.relay.build_module import bind_params_by_name

from tvm.contrib.msc.core import transform as msc_transform
from tvm.contrib.msc.core import utils as msc_utils


def test_relax_layout():
    """Test SetExprLayout for relax"""

    # pylint: disable=import-outside-toplevel
    try:
        import torch
        import torchvision
        from torch import fx
    except:  # pylint: disable=bare-except
        print("please install pytorch python package")
        return

    class RelaxLayoutChecker(PyExprVisitor):
        """Check if name as span attribute is setted."""

        def check(self, expr):
            self._missing_exprs = []
            if isinstance(expr, tvm.relax.Expr):
                self.visit_expr(expr)
            elif isinstance(expr, tvm.relax.BindingBlock):
                self.visit_binding_block(expr)
            assert len(self._missing_exprs) == 0, "Missing {} layouts".format(
                len(self._missing_exprs)
            )

        def visit_var_binding_(self, binding) -> None:
            super().visit_var_binding_(binding)
            if not msc_utils.get_expr_layout(binding.value):
                self._missing_exprs.append(binding.value)

        def visit_constant_(self, op) -> None:
            super().visit_constant_(op)
            if not msc_utils.get_expr_layout(op):
                self._missing_exprs.append(op)

    torch_model = torchvision.models.resnet50()
    graph_model = fx.symbolic_trace(torch_model)
    input_info = [([1, 3, 224, 224], "float32")]
    with torch.no_grad():
        mod = from_fx(graph_model, input_info)
    mod = msc_transform.SetExprLayout()(mod)
    RelaxLayoutChecker().check(mod)


def test_relay_name():
    """Test SetExprName for relay"""

    class RelayNameChecker(ExprVisitor):
        """Check if name as span attribute is setted."""

        def check(self, expr):
            self._missing_exprs = []
            super().visit(expr)
            assert len(self._missing_exprs) == 0, "Missing {} names".format(
                len(self._missing_exprs)
            )

        def visit_constant(self, expr):
            super().visit_constant(expr)
            if not msc_utils.get_expr_name(expr):
                self._missing_exprs.append(expr)

        def visit_call(self, expr):
            super().visit_call(expr)
            if not msc_utils.get_expr_name(expr):
                self._missing_exprs.append(expr)

    mod, params = testing.resnet.get_workload(num_layers=50, batch_size=1, dtype="float32")
    mod["main"] = bind_params_by_name(mod["main"], params)
    mod = msc_transform.SetExprName(as_relax=False)(mod)
    RelayNameChecker().check(mod["main"])


def test_relax():
    """Test SetExprName for relax"""

    # pylint: disable=import-outside-toplevel
    try:
        import torch
        import torchvision
        from torch import fx
    except:  # pylint: disable=bare-except
        print("please install pytorch python package")
        return

    class RelaxNameChecker(PyExprVisitor):
        """Check if name as span attribute is setted."""

        def check(self, expr):
            self._missing_exprs = []
            if isinstance(expr, tvm.relax.Expr):
                self.visit_expr(expr)
            elif isinstance(expr, tvm.relax.BindingBlock):
                self.visit_binding_block(expr)
            assert len(self._missing_exprs) == 0, "Missing {} names".format(
                len(self._missing_exprs)
            )

        def visit_var_binding_(self, binding) -> None:
            super().visit_var_binding_(binding)
            if not msc_utils.get_expr_name(binding.value):
                self._missing_exprs.append(binding.value)

        def visit_constant_(self, op) -> None:
            super().visit_constant_(op)
            if not msc_utils.get_expr_name(op):
                self._missing_exprs.append(op)

    torch_model = torchvision.models.resnet50()
    graph_model = fx.symbolic_trace(torch_model)
    input_info = [([1, 3, 224, 224], "float32")]
    with torch.no_grad():
        mod = from_fx(graph_model, input_info)
    mod = msc_transform.SetExprName()(mod)
    RelaxNameChecker().check(mod)


if __name__ == "__main__":
    tvm.testing.main()
