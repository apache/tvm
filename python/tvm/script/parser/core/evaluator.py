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
"""AST Evaluation"""

import ast
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union

from . import dispatch, doc
from .error import ParserError

if TYPE_CHECKING:
    from .parser import Parser

DEFAULT_OP: Dict[Type, Callable[..., Any]] = {
    doc.Add: lambda a, b: a + b,
    doc.Sub: lambda a, b: a - b,
    doc.Mult: lambda a, b: a * b,
    doc.Div: lambda a, b: a / b,
    doc.FloorDiv: lambda a, b: a // b,
    doc.Mod: lambda a, b: a % b,
    doc.LShift: lambda a, b: a << b,
    doc.RShift: lambda a, b: a >> b,
    doc.BitOr: lambda a, b: a | b,
    doc.BitXor: lambda a, b: a ^ b,
    doc.BitAnd: lambda a, b: a & b,
    doc.MatMult: lambda a, b: a @ b,
    doc.Pow: lambda a, b: a**b,
    doc.Eq: lambda a, b: a == b,
    doc.NotEq: lambda a, b: a != b,
    doc.Lt: lambda a, b: a < b,
    doc.LtE: lambda a, b: a <= b,
    doc.Gt: lambda a, b: a > b,
    doc.GtE: lambda a, b: a >= b,
    doc.Is: lambda a, b: a is b,
    doc.IsNot: lambda a, b: a is not b,
    doc.In: lambda a, b: a in b,
    doc.NotIn: lambda a, b: a not in b,
    doc.And: lambda a, b: a and b,
    doc.Or: lambda a, b: a or b,
    doc.Invert: lambda a: ~a,
    doc.Not: lambda a: not a,
    doc.UAdd: lambda a: +a,
    doc.USub: lambda a: -a,
}


def _get_builtin_or_none(name: str):
    builtins = globals().get("__builtins__")
    if not builtins:
        return None
    if not isinstance(builtins, dict) and hasattr(builtins, "__dict__"):
        builtins = builtins.__dict__
    if isinstance(builtins, dict):
        return builtins.get(name)
    return None


class ExprEvaluator:
    """Expression evaluator for TVMScript parser.

    Parameters
    ----------
    parser : Parser
        The parser bound with the evaluator.

    value_table : Dict[str, Any]
        The value table for expression evaluation.

    new_value_count : int
        The count for intermediate result added during evaluation.
    """

    parser: "Parser"
    value_table: Dict[str, Any]
    new_value_count: int

    def __init__(self, parser: "Parser", value_table: Dict[str, Any]) -> None:
        super().__init__()
        self.parser = parser
        self.value_table = value_table
        self.new_value_count = 0

    @staticmethod
    def eval(parser: "Parser", value_table: Dict[str, Any], node: doc.AST) -> Any:
        """Expression evaluation for TVMScript parser.

        Parameters
        ----------
        parser : Parser
            The parser bound with the evaluator.

        value_table : Dict[str, Any]
            The value table for expression evaluation.

        node : doc.AST
            The root node of AST tree node of expression to evaluate.

        Returns
        -------
        res : Any
            The evaluation result.
        """
        self = ExprEvaluator(parser, value_table)
        result = self._visit(node)  # pylint: disable=protected-access
        if isinstance(result, doc.Name):
            if result.id in self.value_table:
                return self.value_table[result.id]
            else:
                builtin = _get_builtin_or_none(result.id)
                if builtin:
                    return builtin
                raise ParserError(result, f"Undefined variable: {result.id}")
        if isinstance(result, doc.Constant):
            return result.value
        raise TypeError(f"Unexpected result type: {type(result)}")

    def _add_intermediate_result(self, value: Any) -> doc.Name:
        """Add intermediate result during evaluation into value table.

        Parameters
        ----------
        value : Any
            The intermediate result.

        Returns
        -------
        name : doc.Name
            The doc AST name node with intermediate name for intermediate result.
        """
        name = f"__tvm_tmp_value_{self.new_value_count}"
        self.new_value_count += 1
        self.value_table[name] = value
        lineno = 0
        col_offset = 0
        return doc.Name(
            id=name,
            ctx=doc.Load(
                lineno=lineno,
                col_offset=col_offset,
                end_lineno=None,
                end_col_offset=None,
            ),
            lineno=lineno,
            col_offset=col_offset,
            end_lineno=None,
            end_col_offset=None,
        )

    def _visit(self, node: doc.AST) -> Any:
        """General doc AST node visiting method for expression evaluation.

        Parameters
        ----------
        node : doc.AST
            The root node of AST tree node of expression to evaluate.

        Returns
        -------
        res : Any
            The evaluation result.
        """
        args = []
        if (
            isinstance(node, doc.Call)
            and hasattr(node.func, "attr")
            and node.func.attr not in ["reads", "writes", "match_buffer", "realize"]
        ) or isinstance(node, (doc.BinOp, doc.UnaryOp, doc.Compare, doc.BoolOp)):
            if isinstance(node, doc.BinOp):
                args = [node.left, node.right]
            elif isinstance(node, doc.UnaryOp):
                args = [node.operand]
            elif isinstance(node, doc.Compare):
                args = [node.left, *node.comparators]
            else:
                if isinstance(node, doc.Call):
                    args = node.args
                elif isinstance(node, doc.BoolOp):
                    args = node.values
        for arg in args:
            if isinstance(arg, doc.Subscript) and isinstance(arg.slice, (doc.Slice, doc.Tuple)):
                if isinstance(arg.slice, doc.Slice):
                    check_slices = [arg.slice]
                else:
                    check_slices = []
                    for p in arg.slice.elts:
                        if isinstance(p, doc.Slice):
                            check_slices.append(p)
                for s in check_slices:
                    if not s.step and s.upper and s.lower:
                        s.step = doc.Constant(
                            1,
                            None,
                            1,
                            1,
                            s.upper.lineno,
                            s.upper.end_col_offset + 1,
                            s.upper.lineno,
                            s.upper.end_col_offset + 2,
                        )
        if isinstance(node, list):
            return [self._visit(n) for n in node]
        if isinstance(node, tuple):
            return tuple(self._visit(n) for n in node)
        assert isinstance(node, doc.AST)
        if isinstance(node, doc.Name):
            if node.id not in self.value_table and not _get_builtin_or_none(node.id):
                raise ParserError(node, f"Undefined variable: {node.id}")
            return node
        if isinstance(
            node,
            (
                doc.Constant,
                doc.expr_context,
                doc.operator,
                doc.boolop,
                doc.unaryop,
                doc.cmpop,
            ),
        ):
            return node
        if not isinstance(node, (doc.expr, doc.slice)):
            return node
        if isinstance(node, doc.Lambda):
            return self._eval_lambda(node)
        if isinstance(node, doc.Starred):
            value = self._visit(node.value)
            return doc.Starred(
                value=value,
                ctx=node.ctx,
                lineno=node.lineno,
                col_offset=node.col_offset,
                end_lineno=node.end_lineno,
                end_col_offset=node.end_col_offset,
            )

        fields = {}
        for field in node.__class__._FIELDS:  # pylint: disable=protected-access
            attr = getattr(node, field)
            if isinstance(attr, (doc.AST, tuple, list)):
                fields[field] = self._visit(attr)
            else:
                fields[field] = attr
        try:
            if isinstance(node, doc.BoolOp):
                value = self._eval_bool_op(fields)
            elif isinstance(node, doc.Compare):
                value = self._eval_compare(fields)
            elif isinstance(node, doc.UnaryOp):
                value = self._eval_unary_op(fields)
            elif isinstance(node, doc.BinOp):
                value = self._eval_bin_op(fields)
            elif isinstance(node, doc.Slice):
                value = self._eval_slice(fields)
            else:
                value = self._eval_expr(node.__class__(**fields))
        except Exception as e:  # pylint: disable=broad-except,invalid-name
            self.parser.report_error(node, e)
        return self._add_intermediate_result(value)

    def _eval_lambda(self, node: doc.Lambda) -> Any:
        """The doc AST lambda node evaluating method.

        Parameters
        ----------
        node : doc.Lambda
            The root node of AST tree node of expression to evaluate.

        Returns
        -------
        res : Any
            The evaluation result.
        """
        try:
            value = self._eval_expr(node)
        except Exception as e:  # pylint: disable=broad-except,invalid-name
            self.parser.report_error(node, str(e))
        return self._add_intermediate_result(value)

    def _eval_bool_op(self, fields: Dict[str, Any]) -> Any:
        """The doc AST boolean operator node evaluating method.

        Parameters
        ----------
        fields : Dict[str, Any]
            The dictionary of boolean operation information,
            e.g., operator types, operand values.

        Returns
        -------
        res : Any
            The evaluation result.
        """
        op = fields["op"]
        if not isinstance(op, (doc.And, doc.Or)):
            raise TypeError(f"Unexpected operator: {op}")
        value = self._eval_expr(fields["values"][0])
        for rhs in fields["values"][1:]:
            value = _eval_op(op, values=[value, self._eval_expr(rhs)])
        return value

    def _eval_compare(self, fields: Dict[str, Any]) -> Any:
        """The doc AST comparison operation node evaluating method.

        Parameters
        ----------
        fields : Dict[str, Any]
            The dictionary of comparison operation information,
            e.g., operator types, operand values.

        Returns
        -------
        res : Any
            The evaluation result.
        """
        value = self._eval_expr(fields["left"])
        for op, rhs in zip(fields["ops"], fields["comparators"]):
            value = _eval_op(op, values=[value, self._eval_expr(rhs)])
        return value

    def _eval_unary_op(self, fields: Dict[str, Any]) -> Any:
        """The doc AST unary operation node evaluating method.

        Parameters
        ----------
        fields : Dict[str, Any]
            The dictionary of unary operation information,
            e.g., operator types, operand values.

        Returns
        -------
        res : Any
            The evaluation result.
        """
        value = self._eval_expr(fields["operand"])
        value = _eval_op(fields["op"], values=[value])
        return value

    def _eval_bin_op(self, fields: Dict[str, Any]) -> Any:
        """The doc AST binary operation node evaluating method.

        Parameters
        ----------
        fields : Dict[str, Any]
            The dictionary of binary operation information,
            e.g., operator types, operand values.

        Returns
        -------
        res : Any
            The evaluation result.
        """
        return _eval_op(
            fields["op"],
            values=[
                self._eval_expr(fields["left"]),
                self._eval_expr(fields["right"]),
            ],
        )

    def _eval_slice(self, fields: Dict[str, Any]) -> slice:
        """The doc AST slice node evaluating method.

        Parameters
        ----------
        fields : Dict[str, Any]
            The dictionary of slice information,
            e.g., lower bound, upper bound, step.

        Returns
        -------
        res : slice
            The evaluation result.
        """
        lower, upper, step = fields["lower"], fields["upper"], fields["step"]

        lower = self._eval_expr(lower) if lower is not None else None
        upper = self._eval_expr(upper) if upper is not None else None
        step = self._eval_expr(step) if step is not None else None

        return slice(lower, upper, step)

    def _eval_expr(self, v: Any) -> Any:
        """The doc AST expression node evaluating method.

        Parameters
        ----------
        v : Any
            The root node of AST tree node of expression to evaluate.

        Returns
        -------
        res : Any
            The evaluation result.
        """
        return _eval_expr(v, self.value_table)


def eval_expr(
    parser: "Parser",
    node: Union[doc.expr, doc.Expression],
    dict_globals: Optional[Dict[str, Any]],
) -> Any:
    """Expression evaluation for TVMScript parser.

    Parameters
    ----------
    parser : Parser
        The parser bound with the evaluator.

    node : Union[doc.expr, doc.Expression]
        The root node of AST tree node of expression to evaluate.

    dict_globals : Optional[Dict[str, Any]]
        The optional global value table for expression evaluation.

    Returns
    -------
    res : Any
        The evaluation result.
    """
    value_table = {}
    if dict_globals is not None:
        value_table.update(dict_globals)
    return ExprEvaluator.eval(parser, value_table, node)


def eval_assign(
    parser: "Parser",
    target: doc.expr,
    source: Any,
) -> Dict[str, Any]:
    """Expression assignment evaluation for TVMScript parser.

    Parameters
    ----------
    parser : Parser
        The parser bound with the evaluator.

    target : doc.expr
        The root node of AST tree node of assigned expression to evaluate.

    source : Any
        The source to be assigned with evaluated expression.

    Returns
    -------
    res : Any
        The evaluation result.
    """
    try:
        return _eval_assign(target, source)
    except Exception as e:  # pylint: disable=broad-except,invalid-name
        parser.report_error(target, f"Failed to evaluate assignment: {str(e)}")
        raise


def _eval_expr(
    node: Union[doc.expr, doc.Expression],
    dict_globals: Optional[Dict[str, Any]],
) -> Any:
    """Expression evaluation implementation for TVMScript parser.

    Parameters
    ----------
    node : Union[doc.expr, doc.Expression]
        The root node of AST tree node of expression to evaluate.

    dict_globals : Optional[Dict[str, Any]]
        The optional global value table for expression evaluation.

    Returns
    -------
    res : Any
        The evaluation result.
    """
    node = doc.from_doc(node)
    if isinstance(node, ast.expr):
        node = ast.Expression(body=node)
    assert isinstance(node, ast.Expression), "Expects an ast.Expression, but gets: " + str(node)
    if dict_globals is None:
        dict_globals = {}
    node = ast.fix_missing_locations(node)
    exe = compile(node, filename="<ast>", mode="eval")
    return eval(exe, dict_globals)  # pylint: disable=eval-used


def _eval_op(
    op: doc.AST,
    values: List[Any],
):
    """Operation expression evaluation implementation for TVMScript parser.

    Parameters
    ----------
    op : doc.AST
        The root node of AST tree node of operation expression to evaluate.

    values : List[Any]
        The list of values of operands.

    Returns
    -------
    res : Any
        The evaluation result.
    """
    op_type = type(op)  # pylint: disable=protected-access
    for i, v in enumerate(values):
        v_type = getattr(type(v), "_dispatch_type", None)
        if v_type is None:
            continue
        f = dispatch.get_op(
            operand_type=v_type, op_node_type=op_type, operand_index=i, default=None
        )
        if f is not None:
            return f(*values)
    return DEFAULT_OP[op_type](*values)


def _eval_assign(
    target: doc.expr,
    source: Any,
) -> Dict[str, Any]:
    """Expression assignment evaluation implementation for TVMScript parser.

    Parameters
    ----------
    target : doc.expr
        The root node of AST tree node of assigned expression to evaluate.

    source : Any
        The source to be assigned with evaluated expression.

    Returns
    -------
    res : Any
        The evaluation result.
    """
    target = doc.from_doc(target)
    assert isinstance(target, ast.expr)
    RHS_VAR_NAME = "__tvm_rhs_var__"  # pylint: disable=invalid-name
    rhs_var_name = RHS_VAR_NAME
    dict_locals = {rhs_var_name: source}
    mod = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.Assign(
                    targets=[target],
                    value=ast.Name(
                        id=rhs_var_name,
                        ctx=ast.Load(),
                    ),
                )
            ],
            type_ignores=[],
        )
    )
    exe = compile(mod, filename="<ast>", mode="exec")
    exec(exe, {}, dict_locals)  # pylint: disable=exec-used
    del dict_locals[rhs_var_name]
    return dict_locals
