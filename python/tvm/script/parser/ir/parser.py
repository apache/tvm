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
# pylint: disable=unused-argument
"""The base parser for ir module"""

from tvm.ir import GlobalVar
from tvm.relax import ExternFunc

from ...ir_builder import ir as I
from .._core import Parser, dispatch, doc


class ModuleWithGlobalVars:
    """A Module that can add global vars during parsing, to support `Module.function` syntax."""

    def __getattr__(self, attr):
        # Customize the error message.
        # NOTE: `__getattr__` is only called when the attribute access fails with an AttributeError
        raise AttributeError(f"Cannot find the function `{attr}` in the current IRModule")


@dispatch.register(token="ir", type_name="ClassDef")
def _visit_class_def(self: Parser, node: doc.ClassDef) -> None:
    """The class definition visiting method for ir module.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.ClassDef
        The doc AST class definition node.
    """

    with self.var_table.with_frame():
        with I.ir_module():
            # Step 0. Add the class name to the var table
            fake_module = ModuleWithGlobalVars()
            self.var_table.add(node.name, fake_module)

            # Step 1: Check if this class inherits from BasePyModule
            is_base_py_module = _check_base_py_module_inheritance(node)
            if is_base_py_module:
                # Store this information in the IRModule for later use
                I.module_attrs({"base_py_module": True})
                # Set the parser context to allow Python functions
                self.set_class_context(node.name, True)
            else:
                # Set the parser context to disallow Python functions
                self.set_class_context(node.name, False)

            # Step 2. Visit non-function stmts, including but not limited to
            # 1. `I.module_attrs`
            # 2. `I.module_global_infos`
            with self.with_dispatch_token("ir"):
                for stmt in node.body:
                    if not isinstance(stmt, doc.FunctionDef):
                        self.visit(stmt)

            # Step 3. Visit function stmts to declare the global vars
            for stmt in node.body:
                if isinstance(stmt, doc.FunctionDef):
                    global_var = self.visit_tvm_declare_function(stmt)
                    fake_module.__setattr__(stmt.name, global_var)

            # Step 4. Visit and parse the functions
            with self.with_dispatch_token("ir"):
                for stmt in node.body:
                    if isinstance(stmt, doc.FunctionDef):
                        self.visit(stmt)


@dispatch.register(token="ir", type_name="Assign")
def _visit_assign(self: Parser, node: doc.Assign) -> None:
    """The assign visiting method for ir module.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.ClassDef
        The doc AST assign node.
    """
    if len(node.targets) != 1:
        self.report_error(node, "Consequential assignments like 'a = b = c' are not supported.")
    lhs = node.targets[0].id
    rhs = self.eval_expr(node.value)

    I.decl_function(lhs, rhs)
    I.def_function(lhs, rhs)


@dispatch.register(token="ir", type_name="Expr")
def _visit_expr(self: Parser, node: doc.Expr) -> None:
    """The expression visiting method for ir module.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.ClassDef
        The doc AST expression node.
    """
    self.eval_expr(node.value)


@dispatch.register(token="default", type_name="Assign")
def visit_assign(self: Parser, node: doc.Assign) -> None:
    if len(node.targets) != 1:
        self.report_error(node, "Consequential assignments like 'a = b = c' are not supported.")
    lhs = node.targets[0]
    rhs = self.eval_expr(node.value)
    self.eval_assign(
        target=lhs, source=rhs, bind_value=lambda _a, _b, _c, value: value, allow_shadowing=True
    )


@dispatch.register(token="default", type_name="pre_visit_local_function")
def pre_visit_local_function(self: Parser, node: doc.Expr) -> None:
    pass


@dispatch.register(token="default", type_name="post_visit_local_function")
def post_visit_local_function(self: Parser, node: doc.Expr) -> None:
    pass


@dispatch.register(token="pyfunc", type_name="tvm_declare_function")
def visit_tvm_declare_function(self: Parser, node: doc.FunctionDef) -> GlobalVar:
    """Declare a Python function as an ExternFunc in the IRModule."""
    # Check if Python functions are allowed in this context
    # We need to check if we're in a class that inherits from BasePyModule
    current_class = self._get_current_class_context()
    if current_class and not self._is_base_py_module_context():
        self.report_error(
            node,
            "@I.pyfunc are only allowed in classes that inherit from BasePyModule. "
            f"Class '{current_class}' does not inherit from BasePyModule.",
        )

    # Create ExternFunc with proper attributes for Python functions
    func = ExternFunc(node.name)
    func = func.with_attr("is_pyfunc", True)
    func = func.with_attr("function_type", "python")
    func = func.with_attr("python_function_name", node.name)

    # Add placeholder attributes that will be filled in later
    func = func.with_attr("python_source", f"# Source will be filled for {node.name}")
    func = func.with_attr("python_packed_func", None)  # Will be filled in entry.py

    # Store the function name for later retrieval
    return I.decl_function(node.name, func)


@dispatch.register(token="pyfunc", type_name="FunctionDef")
def visit_function_def(self: Parser, node: doc.FunctionDef) -> None:
    """Visit Python function definition - no need to parse the body."""
    # Python function body is not parsed in TVMScript


def _check_base_py_module_inheritance(node: doc.ClassDef) -> bool:
    """Check if a class inherits from BasePyModule.

    Parameters
    ----------
    node : doc.ClassDef
        The class definition node to check.

    Returns
    -------
    bool
        True if the class inherits from BasePyModule, False otherwise.
    """
    if not node.bases:
        return False

    # Check each base class
    for base in node.bases:
        if hasattr(base, "id"):
            if base.id == "BasePyModule":
                return True
        elif hasattr(base, "attr"):
            if base.attr == "BasePyModule":
                return True
        elif hasattr(base, "value") and hasattr(base.value, "id"):
            if (
                base.value.id in ["BasePyModule", "tvm", "relax"]
                and hasattr(base, "attr")
                and base.attr == "BasePyModule"
            ):
                return True

    return False
