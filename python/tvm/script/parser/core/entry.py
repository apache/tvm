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
"""The entry point of TVM parser."""
import inspect
from typing import Any, Dict, Union

import tvm
from tvm.relax import ExternFunc
from ....ir.module import IRModule
from ...ir_builder import IRBuilder
from . import doc
from .diagnostics import Source
from .error import ParserError
from .parser import Parser

WELL_FORMED_ERROR_MESSAGE = (
    "Program is not well-formed. If this is deliberate, consider "
    "setting check_well_formed in the top-level decorator to False "
    "(e.g., @I.ir_module(check_well_formed=False) or "
    "@R.function(check_well_formed=False))."
)


def _default_globals() -> Dict[str, Any]:
    from tvm.script.parser import ir  # pylint: disable=import-outside-toplevel
    from tvm.script.parser import relax  # pylint: disable=import-outside-toplevel
    from tvm.script.parser import tir  # pylint: disable=import-outside-toplevel

    extra_vars = {
        "tvm": tvm,
        "I": ir,
        "ir": ir,
        "T": tir,
        "tir": tir,
        "R": relax,
        "relax": relax,
    }
    return extra_vars


def scan_macro(program: Union[Any, str], extra_vars: Dict[str, Any] = None) -> Any:
    """Generate the AST, and the source code for __repr__."""
    # The AST will be converted into TIR at the time of expansion.
    source = Source(program)
    closure_vars = extra_vars or _default_globals()
    return source, closure_vars


def parse(
    program: Union[doc.AST, Any, str],
    extra_vars: Dict[str, Any] = None,
    check_well_formed: bool = True,
) -> Any:
    """Register a method for a operand type, AST operator node and operand index.

    Parameters
    ----------
    program : Union[doc.AST, Any, str]
        The TVMScript code to parse.

    extra_vars : Dict[str, Any]
        The extra variable table for parsing.

    check_well_formed : bool
        Whether to check well-formedness after parsing.

    Returns
    -------
    func : Any
        The parsed TVMScript program.
    """
    if extra_vars is None:
        extra_vars = _default_globals()

    ann = {}
    all_pyfuncs = {}
    if inspect.isfunction(program):
        ann = {program.__name__: program.__annotations__}
    elif inspect.isclass(program):
        for name, func in program.__dict__.items():
            if inspect.isfunction(func):
                ann[name] = func.__annotations__
                all_pyfuncs[name] = func

    source = Source(program)
    parser = Parser(source, ann)
    with IRBuilder() as builder:
        try:
            parser.parse(extra_vars=extra_vars)
        except ParserError as err:
            parser.report_error(err.node, err.args[0])
    ret = builder.get()
    # Attach pyfuncs to the IRModule
    if inspect.isclass(program) and isinstance(ret, IRModule):
        _attach_pyfuncs_to_irmodule(ret, all_pyfuncs)

    # check well-formedness in both Relax and TIR
    if check_well_formed:
        check_ret = ret
        if not isinstance(check_ret, IRModule):
            check_ret = IRModule.from_expr(ret)

        source_ast = source.as_ast()

        if isinstance(ret, (IRModule, tvm.relax.Function)) and not tvm.relax.analysis.well_formed(
            ret
        ):
            parser.report_error(source_ast, err=WELL_FORMED_ERROR_MESSAGE)

        try:
            tvm.tir.analysis.verify_well_formed(check_ret)
        except Exception as err:  # pylint: disable=broad-exception-caught
            parser.report_error(
                source_ast,
                err=f"{WELL_FORMED_ERROR_MESSAGE}\n\nTraceback: {str(err)}",
            )
    return ret


def _create_python_packed_func(pyfunc):
    """Create a PackedFunc wrapper for a Python function.

    This function creates a PackedFunc that can be called from TVM runtime
    and will execute the original Python function.

    Parameters
    ----------
    pyfunc : Callable
        The Python function to wrap.

    Returns
    -------
    PackedFunc
        A PackedFunc that wraps the Python function.
    """

    def packed_func_wrapper(*args, **kwargs):
        """Wrapper function that calls the original Python function."""
        try:
            result = pyfunc(*args, **kwargs)
            return result
        except Exception as error:
            print(f"Error calling Python function {pyfunc.__name__}: {error}")
            raise

    return packed_func_wrapper


def _attach_pyfuncs_to_irmodule(irmodule, all_pyfuncs):
    """Attach Python functions to IRModule with reduced nesting."""
    if not all_pyfuncs:
        return

    if not hasattr(irmodule, "pyfuncs"):
        irmodule.pyfuncs = {}

    for global_var, func in irmodule.functions_items():
        if not isinstance(func, ExternFunc):
            continue
        if not func.attrs.get("is_pyfunc", False):
            continue

        pyfunc_name = global_var.name_hint
        if pyfunc_name not in all_pyfuncs:
            continue

        pyfunc = all_pyfuncs[pyfunc_name]
        irmodule.pyfuncs[pyfunc_name] = pyfunc

        try:
            source_code = inspect.getsource(pyfunc)
            func = func.with_attr("python_source", source_code)
        except (OSError, TypeError):
            func = func.with_attr("python_source", f"# Source unavailable for {pyfunc_name}")

        packed_func = _create_python_packed_func(pyfunc)
        func = func.with_attr("python_packed_func", packed_func)

        irmodule[global_var] = func
