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

from ....relax.analysis import well_formed as relax_well_formed
from ....tir.analysis import verify_well_formed as tir_well_formed
from ....ir.module import IRModule
from ...ir_builder import IRBuilder
from . import doc
from .diagnostics import Source
from .error import ParserError
from .parser import Parser


def _default_globals() -> Dict[str, Any]:
    import tvm  # pylint: disable=import-outside-toplevel
    from tvm.script.parser import ir  # pylint: disable=import-outside-toplevel
    from tvm.script.parser import relax  # pylint: disable=import-outside-toplevel
    from tvm.script.parser import tir  # pylint: disable=import-outside-toplevel

    extra_vars = {"tvm": tvm, "I": ir, "ir": ir, "T": tir, "tir": tir, "R": relax, "relax": relax}
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
    if inspect.isfunction(program):
        ann = {program.__name__: program.__annotations__}
    elif inspect.isclass(program):
        for name, func in program.__dict__.items():
            if inspect.isfunction(func):
                ann[name] = func.__annotations__

    source = Source(program)
    parser = Parser(source, ann)
    with IRBuilder() as builder:
        try:
            parser.parse(extra_vars=extra_vars)
        except ParserError as err:
            parser.report_error(err.node, err.args[0])
    ret = builder.get()
    # check well-formedness in both Relax and TIR
    if (
        check_well_formed
        and isinstance(ret, IRModule)
        and not (relax_well_formed(ret) and tir_well_formed(ret, assert_mode=False))
    ):
        parser.report_error(
            source.as_ast(),
            err="Program containing Relax functions is not well-formed",
        )
    return ret
