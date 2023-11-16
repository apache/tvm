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
"""The core parser"""

import abc
import inspect
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np

from tvm._ffi.base import TVMError
from tvm.error import DiagnosticError
from tvm.ir import GlobalVar

from . import dispatch, doc
from .diagnostics import Diagnostics, Source
from .evaluator import eval_assign, eval_expr

DEFAULT_VISIT = {
    "Interactive",
    "Module",
    "Expression",
    "Pass",
}


def _deferred(exit_f: Callable[[], None]):
    """Created context with certain exit function.

    Parameters
    ----------
    exit_f : Callable[[], None]
        The function to call when exiting the context.

    Returns
    -------
    res : Any
        The created context.
    """

    @contextmanager
    def context():
        try:
            yield
        finally:
            exit_f()

    return context()


def _do_nothing(*args, **kwargs):  # pylint: disable=unused-argument
    pass


class ScriptMacro(abc.ABC):
    """Representation of a script macro.

    This is a callable object, intended to be called from the expression evaluator.
    The evaluator is expected to insert the current parser into the environment
    undef the name given by "parser_object_name".

    Once called, the ScriptMacro object will locate the current parser, and use it
    to parse the macro's body and produce the result.

    There were two major considerations for this design:
    1. Implementing hygienic and non-hygienic macros.
    2. Implementing macros that return values.

    Macro uses in TIR are only allowed at a statement-level, and they don't produce
    any values. Parsing of such macros could easily be done by intercepting doc.Call
    nodes in the TIR parser. If a macro is a value-producing expression, then there
    may not be a direct way to intercept calls to it if it's embedded in a complex
    expression. Because macros use function-call syntax, the evaluator will try to
    call the macro object, which this design relies on to parse and evaluate the macro.
    """

    parser_object_name = "__current_script_parser__"

    def __init__(
        self,
        source: Source,
        closure_vars: Dict[str, Any],
        func: Callable,
        hygienic: bool,
    ) -> None:
        self.source = source
        self.closure_vars = closure_vars
        self.func = func
        self.hygienic = hygienic

    def __repr__(self):
        return self.source.source

    @abc.abstractmethod
    def parse_macro(self, parser: "Parser") -> Any:
        """The main macro parsing function. Different scripts may have different
        ways to parse a macro, and to return a value to the evaluator.

        Parameters
        ----------
        parser : Parser
            The parser with the appropriate frame already created and populated depending
            macro's hygiene settings,

        Returns
        -------
            The return value depends on the specifics of the particular script. It can be
            "None" or any other value or any type.
        """

    def _find_parser_def(self):
        outer_frame_infos = inspect.getouterframes(inspect.currentframe())
        for finfo in outer_frame_infos:
            parser = finfo.frame.f_globals.get(ScriptMacro.parser_object_name)
            if parser is not None:
                return parser
        raise RuntimeError(f"{ScriptMacro.parser_object_name} not available")

    def get_macro_def(self):
        ast_module = self.source.as_ast()
        for decl in ast_module.body:
            if isinstance(decl, doc.FunctionDef) and decl.name == self.__name__:
                return decl
        raise RuntimeError(f"cannot find macro definition for {self.__name__}")

    def __call__(self, *args, **kwargs):
        param_binding = inspect.signature(self.func).bind(*args, **kwargs)
        param_binding.apply_defaults()
        local_vars = param_binding.arguments
        parser = self._find_parser_def()

        if self.hygienic:
            saved_var_table = parser.var_table
            parser.var_table = VarTable()

            with parser.var_table.with_frame():
                for k, v in self.closure_vars.items():
                    parser.var_table.add(k, v)
                for k, v in local_vars.items():
                    parser.var_table.add(k, v)

                parse_result = self.parse_macro(parser)

            parser.var_table = saved_var_table

        else:
            with parser.var_table.with_frame():
                for k, v in local_vars.items():
                    parser.var_table.add(k, v)

                parse_result = self.parse_macro(parser)

        return parse_result


class VarTableFrame:
    """The variable table frame.
    A frame of variable table stores the variables created in one block or scope.

    Parameters
    ----------
    vars : Set[str]
        The set of variable names in the variable table frame.
    """

    vars: Set[str]

    def __init__(self):
        self.vars = set()

    def add(self, var: str):
        """Add a new variable into variable table frame.

        Parameters
        ----------
        var : str
            The name of new variable.
        """
        if var in self.vars:
            raise ValueError(f"Variable {var} already defined in current scope")
        self.vars.add(var)

    def pop_all(self, fn_pop: Callable[[str], None]):
        """Pop out all variable in variable table frame.

        Parameters
        ----------
        fn_pop : Callable[[str], None]
            The methods to call when popping each variable.
        """
        for var in self.vars:
            fn_pop(var)
        self.vars.clear()


class VarTable:
    """The variable table.
    A variable table stores the all variables when parsing TVMScript.

    Parameters
    ----------
    frames : List[VarTableFrame]
        The list or stack of variable table frame.

    name2value : Dict[str, List[Any]]
        The dictionary for variable table name-based query.
    """

    frames: List[VarTableFrame]
    name2value: Dict[str, List[Any]]

    def __init__(self):
        self.frames = []
        self.name2value = defaultdict(list)

    def with_frame(self):
        """Create a new variable table frame as with statement.

        Returns
        -------
        res : Any
            The context with new variable table frame.
        """

        def pop_frame():
            frame = self.frames.pop()
            frame.pop_all(lambda name: self.name2value[name].pop())

        self.frames.append(VarTableFrame())
        return _deferred(pop_frame)

    def add(self, var: str, value: Any, allow_shadowing: bool = False):
        """Add a new variable to variable table.

        Parameters
        ----------
        var : str
            The name of variable.

        value : Any
            The value of variable.

        allow_shadowing : bool
            The options of whether variable shadowing allowed for this variable.
        """
        # Skip if the key and value are equal to those in the var_table
        if self.name2value[var] and isinstance(self.name2value[var][-1], type(value)):
            if isinstance(value, np.ndarray) and (self.name2value[var][-1] == value).all():
                return
            elif self.name2value[var][-1] == value:
                return
        if allow_shadowing and var in self.frames[-1].vars:
            # Shadowing
            self.name2value[var][-1] = value
        else:
            self.frames[-1].add(var)
            self.name2value[var].append(value)

    def get(self) -> Dict[str, Any]:
        """Get a variable dictionary of latest variables.

        Returns
        -------
        res : Any
            The variable dictionary copy of latest variables.
        """
        return {key: values[-1] for key, values in self.name2value.items() if values}

    def exist(self, value: Any) -> bool:
        """Check if any value exists in variable table.

        Parameters
        ----------
        value : Any
            The value of variable.

        Returns
        -------
        res : bool
            The existence of the value.
        """
        return any(
            value.same_as(known_value)
            for known_value_stack in self.name2value.values()
            for known_value in known_value_stack
        )


def _dispatch_wrapper(func: dispatch.ParseMethod) -> dispatch.ParseMethod:
    def _wrapper(self: "Parser", node: doc.AST) -> None:
        try:
            return func(self, node)
        except DiagnosticError:
            raise
        except Exception as e:  # pylint: disable=broad-except,invalid-name
            self.report_error(node, e)
            raise

    return _wrapper


def _dispatch(self: "Parser", type_name: str) -> dispatch.ParseMethod:
    for token in [self.dispatch_tokens[-1], "default"]:
        func = dispatch.get(token=token, type_name=type_name, default=None)
        if func is not None:
            return _dispatch_wrapper(func)
    return _dispatch_wrapper(lambda self, node: self.generic_visit(node))


class Parser(doc.NodeVisitor):
    """The TVMScript parser

    Parameters
    ----------
    diag : Diagnostics
        The diagnostics for error reporting.

    dispatch_tokens : List[str]
        The list of dispatching tokens to dispatching parsing method
        of different IRs and different doc AST structure.

    var_table : VarTable
        The variable table for parsing.
    """

    diag: Diagnostics
    dispatch_tokens: List[str]
    function_annotations: Optional[Dict[str, Dict[str, Any]]]
    var_table: VarTable
    inside_function: bool  # whether we are within a function

    def __init__(
        self,
        source: Source,
        function_annotations: Dict[str, Dict[str, Any]],
    ) -> None:
        self.diag = Diagnostics(source)
        self.dispatch_tokens = ["default"]
        self.function_annotations = function_annotations
        self.var_table = VarTable()
        self.inside_function = False

    def parse(self, extra_vars: Optional[Dict[str, Any]] = None) -> Any:
        """The main parse method for parser.

        Parameters
        ----------
        extra_vars : Optional[Dict[str, Any]]
            The optional global value table for parsing.

        Returns
        -------
        res : Any
            The doc AST node visiting result.
        """
        if extra_vars is None:
            extra_vars = {}
        with self.var_table.with_frame():
            for k, v in extra_vars.items():
                self.var_table.add(k, v)
            node = self.diag.source.as_ast()
            self.visit(node)

    def get_dispatch_token(self, node: doc.FunctionDef) -> str:
        if not isinstance(node, doc.FunctionDef):
            self.report_error(node, "Only can get dispatch token for function.")
        if not node.decorator_list:
            self.report_error(node, "Function must be decorated")
        # TODO: only the last decorator is parsed
        decorator = self.eval_expr(node.decorator_list[-1])
        if not hasattr(decorator, "dispatch_token"):
            self.report_error(node, "The parser does not understand the decorator")
        return decorator.dispatch_token

    def with_dispatch_token(self, token: str):
        """Add a new dispatching token as with statement.

        Parameters
        ----------
        token : str
            The dispatching token.

        Returns
        -------
        res : Any
            The context with new dispatching token.
        """

        self.dispatch_tokens.append(token)
        enter_func = dispatch.get(token=token, type_name="enter_token", default=lambda *args: None)
        context = enter_func(self)

        def pop_token():
            exit_func = dispatch.get(
                token=token, type_name="exit_token", default=lambda *args: None
            )
            exit_func(self, context)
            self.dispatch_tokens.pop()

        return _deferred(pop_token)

    def eval_expr(
        self,
        node: Union[doc.Expression, doc.expr],
        extra_vars: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Expression evaluation when parsing.

        Parameters
        ----------
        node : Union[doc.expr, doc.Expression]
            The root node of AST tree node of expression to evaluate.

        extra_vars : Optional[Dict[str, Any]]
            The optional global value table for expression evaluation.

        Returns
        -------
        res : Any
            The evaluation result.
        """
        var_values = self.var_table.get()
        if extra_vars is not None:
            for k, v in extra_vars.items():
                var_values[k] = v
        var_values[ScriptMacro.parser_object_name] = self
        return eval_expr(self, node, var_values)

    def _duplicate_lhs_check(self, target: doc.expr) -> Union[bool, Set[str]]:
        """Check whether duplicate lhs exists in assignment.

        Parameters
        ----------
        target : doc.expr
            The doc AST expr node for lhs.

        Returns
        -------
        res : Union[bool, Set[str]]
            The result of true if duplicate lhs exists,
            or the set of lhs names if no duplicate lhs exists.
        """
        if isinstance(target, (doc.Tuple, doc.List)):
            vars: Set[str] = set()  # pylint: disable=redefined-builtin
            for i in target.elts:
                res = self._duplicate_lhs_check(i)
                if isinstance(res, bool) and res:
                    return True
                assert isinstance(res, set)
                if vars & res:
                    return True
                vars = vars.union(res)
            return vars
        elif isinstance(target, doc.Name):
            return {target.id}
        elif isinstance(target, doc.Starred):
            return self._duplicate_lhs_check(target.value)
        else:
            self.report_error(target, "Invalid type in assign statement")
            raise NotImplementedError

    def eval_assign(
        self,
        target: doc.expr,
        source: Any,
        bind_value: Callable[["Parser", doc.expr, str, Any], Any],
        allow_shadowing: bool = False,
    ) -> Dict[str, Any]:
        """Expression assignment evaluation when parsing.

        Parameters
        ----------
        target : doc.expr
            The root node of AST tree node of assigned expression to evaluate.

        source : Any
            The source to be assigned with evaluated expression.

        bind_value : Callable[["Parser", doc.expr, str, Any], Any]
            The value binding method when assigning the values to variables.

        allow_shadowing : bool
            The options of whether variable shadowing allowed for assignment.

        Returns
        -------
        res : Dict[str, Any]
            The dictionary of assignment result.
        """
        if self._duplicate_lhs_check(target) is True:
            self.report_error(target, "Duplicate vars assigned.")
        var_values = eval_assign(self, target, source)
        for k, v in var_values.items():
            var = bind_value(self, target, k, v)
            self.var_table.add(k, var, allow_shadowing)
        return var_values

    def report_error(
        self, node: doc.AST, err: Union[Exception, str]
    ) -> None:  # pylint: disable=no-self-use
        """The error reporting when parsing.

        Parameters
        ----------
        node : doc.AST
            The doc AST node with errors.

        err: Union[Exception, str]
            The error to report.
        """
        # Only take the last line of the error message
        if isinstance(err, TVMError):
            msg = list(filter(None, str(err).split("\n")))[-1]
        elif isinstance(err, KeyError):
            msg = "KeyError: " + str(err)
        else:
            msg = str(err)

        try:
            self.diag.error(node, msg)
        except Exception as diag_err:
            # Calling self.diag.error is guaranteed to throw an
            # exception.  When shown to a user, this error should
            # reference the point of error within the provided
            # TVMScript.  However, when caught in pdb, the full
            # traceback should be available for debugging.
            if isinstance(err, Exception):
                diag_err = diag_err.with_traceback(err.__traceback__)
            raise diag_err

    def visit(self, node: doc.AST) -> None:
        """The general visiting method.

        Parameters
        ----------
        node : doc.AST
            The doc AST node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        if isinstance(node, (list, tuple)):
            for item in node:
                self.visit(item)
            return
        if not isinstance(node, doc.AST):
            return
        name = node.__class__.__name__.split(".")[-1]
        if name in DEFAULT_VISIT:
            func = self.generic_visit
        else:
            func = getattr(self, "visit_" + name, None)
        if func is None:
            raise NotImplementedError(f"Visitor of AST node is not implemented: {name}")
        try:
            func(node)
        except DiagnosticError:
            raise
        except Exception as e:  # pylint: disable=broad-except,invalid-name
            self.report_error(node, str(e))
            raise

    def visit_body(self, node: List[doc.stmt]) -> Any:
        """The general body visiting method.

        Parameters
        ----------
        node : List[doc.stmt]
            The list of statements in body.

        Returns
        -------
        res : Any
            The visiting result.
        """
        for stmt in node:
            self.visit(stmt)

    def visit_tvm_annotation(self, node: doc.expr) -> Any:
        """The general TVM annotation visiting method.

        Parameters
        ----------
        node : doc.expr
            The doc AST expr node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        return _dispatch(self, "tvm_annotation")(self, node)

    def visit_FunctionDef(self, node: doc.FunctionDef) -> None:  # pylint: disable=invalid-name
        """The general function definition visit method.

        Parameters
        ----------
        node : doc.FunctionDef
            The doc FunctionDef node.
        """
        token = self.get_dispatch_token(node)
        func = dispatch.get(token=token, type_name="FunctionDef", default=None)
        if func is None:
            self.report_error(node, "The parser does not understand the decorator")
        _dispatch(self, "pre_visit_local_function")(self, node)
        _dispatch_wrapper(func)(self, node)
        _dispatch(self, "post_visit_local_function")(self, node)

    def visit_tvm_declare_function(self, node: doc.FunctionDef) -> GlobalVar:
        token = self.get_dispatch_token(node)
        with self.with_dispatch_token(token):
            return _dispatch(self, "tvm_declare_function")(self, node)

    def visit_ClassDef(self, node: doc.ClassDef) -> Any:  # pylint: disable=invalid-name
        """The general class definition visiting method.

        Parameters
        ----------
        node : doc.ClassDef
            The doc AST class definition node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        func = dispatch.get(token="ir", type_name="ClassDef", default=None)
        if func is None:
            self.report_error(node, "The parser does not understand the decorator")
        _dispatch_wrapper(func)(self, node)

    def visit_arguments(self, node: doc.arguments) -> Any:
        """The general arguments visiting method.

        Parameters
        ----------
        node : doc.arguments
            The doc AST arguments node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        return _dispatch(self, "arguments")(self, node)

    def visit_For(self, node: doc.For) -> Any:  # pylint: disable=invalid-name
        """The general for visiting method.

        Parameters
        ----------
        node : doc.For
            The doc AST for node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        return _dispatch(self, "For")(self, node)

    def visit_While(self, node: doc.While) -> Any:  # pylint: disable=invalid-name
        """The general while visiting method.

        Parameters
        ----------
        node : doc.While
            The doc AST while node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        return _dispatch(self, "While")(self, node)

    def visit_With(self, node: doc.With) -> Any:  # pylint: disable=invalid-name
        """The general with visiting method.

        Parameters
        ----------
        node : doc.With
            The doc AST with node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        return _dispatch(self, "With")(self, node)

    def visit_Assign(self, node: doc.Assign) -> Any:  # pylint: disable=invalid-name
        """The general assign visiting method.

        Parameters
        ----------
        node : doc.Assign
            The doc AST assign node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        return _dispatch(self, "Assign")(self, node)

    def visit_AnnAssign(self, node: doc.AnnAssign) -> Any:  # pylint: disable=invalid-name
        """The general annotated assign visiting method.

        Parameters
        ----------
        node : doc.Assign
            The doc AST annotated assign node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        return _dispatch(self, "AnnAssign")(self, node)

    def visit_Expr(self, node: doc.Expr) -> Any:  # pylint: disable=invalid-name
        """The general expression visiting method.

        Parameters
        ----------
        node : doc.Expr
            The doc AST expression node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        return _dispatch(self, "Expr")(self, node)

    def visit_If(self, node: doc.If) -> Any:  # pylint: disable=invalid-name
        """The general if visiting method.

        Parameters
        ----------
        node : doc.If
            The doc AST if node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        return _dispatch(self, "If")(self, node)

    def visit_AugAssign(self, node: doc.AugAssign) -> Any:  # pylint: disable=invalid-name
        """The general augmented assignment visiting method.

        Parameters
        ----------
        node : doc.AugAssign
            The doc AST augmented assignment node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        return _dispatch(self, "AugAssign")(self, node)

    def visit_Assert(self, node: doc.Assert) -> Any:  # pylint: disable=invalid-name
        """The general assert visiting method.

        Parameters
        ----------
        node : doc.Assert
            The doc AST assert node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        return _dispatch(self, "Assert")(self, node)

    def visit_Return(self, node: doc.Return) -> Any:  # pylint: disable=invalid-name
        """The general return visiting method.

        Parameters
        ----------
        node : doc.Return
            The doc AST return node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        return _dispatch(self, "Return")(self, node)

    def visit_Nonlocal(self, node: doc.Nonlocal) -> Any:  # pylint: disable=invalid-name
        """The general nonlocal visiting method.

        Parameters
        ----------
        node : doc.Nonlocal
            The doc AST nonlocal node.

        Returns
        -------
        res : Any
            The visiting result.
        """
        return _dispatch(self, "Nonlocal")(self, node)
