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

# pylint: disable=invalid-name, unused-argument
"""A parser for Relay's text format."""
from __future__ import absolute_import

import sys
from ast import literal_eval
from collections import deque

try:
    # no typing.Deque in Python 3.5
    # https://bugs.python.org/issue29011
    from typing import Any, Dict, List, Optional, TypeVar, Tuple, Union, MutableSequence, T, Deque
except ImportError:
    class Deque(deque, MutableSequence[T], extra=deque):

        def __new__(cls, *args, **kwds):
            if _geqv(cls, Deque):
                raise TypeError("Type Deque cannot be instantiated; "
                                "use deque() instead")
            return deque.__new__(cls, *args, **kwds)

import tvm
import tvm.ir._ffi_api
from tvm.ir import IRModule

from .base import Span, SourceName
from . import adt
from . import expr
from . import function
from . import ty
from . import op

PYTHON_VERSION = sys.version_info.major
try:
    from antlr4 import InputStream, CommonTokenStream
    from antlr4.error.ErrorListener import ErrorListener
except ImportError:
    raise Exception("Couldn't find ANTLR runtime." +
                    "Try running `pip{version} install antlr4-python{version}-runtime`."
                    .format(version=PYTHON_VERSION))

try:
    from .grammar.py3.RelayVisitor import RelayVisitor
    from .grammar.py3.RelayParser import RelayParser
    from .grammar.py3.RelayLexer import RelayLexer
except ImportError:
    raise Exception("Couldn't find ANTLR parser. Try building with USE_ANTLR=ON.")


sys.setrecursionlimit(10000)

class ParseError(Exception):
    """Exception type for parse errors."""

    def __init__(self, message: str) -> None:
        super(ParseError, self).__init__()
        self.message = message

    def __repr__(self):
        return "ParseError({})".format(self.message)

    def __str__(self):
        return repr(self)

class OpWrapper:
    """Overload the __call__ for op."""


class ExprOp(OpWrapper):
    """Call an expr. The default, but does not handle attrs well."""
    def __init__(self, operator):
        self.operator = operator

    def __call__(self, args, attrs, type_args):
        try:
            return expr.Call(self.operator, args, attrs, type_args)
        except Exception:
            raise Exception("Operator {} is not registered. It's attributes are {}"
                            .format(self.operator, attrs))

class FuncOp(OpWrapper):
    """Convert the attrs, call the python function with the attrs passed in as keyword arguments.
    Tvm should provide this in the future, as this is pretty similar to what op.get is providing.
    """
    def __init__(self, operator):
        self.operator = operator

    def convert(self, v):
        if isinstance(v, tuple):
            return tuple([self.convert(x) for x in v])
        if isinstance(v, expr.Constant):
            return v.data.asnumpy().item()
        if isinstance(v, str):
            return v
        raise Exception(v)

    def __call__(self, args, attrs, type_args):
        if attrs is None:
            attrs = {}
        x = self.operator(*args, **{k: self.convert(v) for k, v in attrs.items()})
        if isinstance(x, expr.TupleWrapper):
            x = x.astuple()
        return x

BINARY_OPS = {
    RelayParser.MUL: op.multiply,
    RelayParser.DIV: op.divide,
    RelayParser.ADD: op.add,
    RelayParser.SUB: op.subtract,
    RelayParser.LT:  op.less,
    RelayParser.GT:  op.greater,
    RelayParser.LE:  op.less_equal,
    RelayParser.GE:  op.greater_equal,
    RelayParser.EQ:  op.equal,
    RelayParser.NE:  op.not_equal,
}

FUNC_OPS = {
    "nn.conv2d": op.nn.conv2d,
    "nn.batch_norm": op.nn.batch_norm,
    "nn.dense": op.nn.dense,
    "nn.bias_add": op.nn.bias_add,
    "nn.max_pool2d": op.nn.max_pool2d,
    "nn.max_pool3d": op.nn.max_pool3d,
    "nn.global_max_pool2d": op.nn.global_max_pool2d,
    "nn.avg_pool2d": op.nn.avg_pool2d,
    "nn.avg_pool3d": op.nn.avg_pool3d,
    "nn.global_avg_pool2d": op.nn.global_avg_pool2d,
    "nn.softmax": op.nn.softmax,
    "reshape": op.reshape,
    "nn.conv2d_transpose": op.nn.conv2d_transpose,
    "nn.conv1d_transpose": op.nn.conv1d_transpose,
    "concatenate": op.concatenate,
    "nn.dropout": op.nn.dropout_raw,
    "zeros": op.zeros,
    "split": op.split,
    "cast": op.cast,
    "clip": op.clip,
    "right_shift": op.right_shift,
}

TYPE_PREFIXES = [
    "int",
    "uint",
    "float",
    "bool",
]

T = TypeVar("T")
Scope = Deque[Tuple[str, T]]
Scopes = Deque[Scope[T]]

def lookup(scopes: Scopes[T], name: str) -> Optional[T]:
    """Look up `name` in `scopes`."""

    for scope in scopes:
        for key, val in scope:
            if key == name:
                return val
    return None

def spanify(f):
    """A decorator which attaches span information
       to the value returned by calling `f`.

       Intended for use with the below AST visiting
       methods. The idea is that after we do the work
       of constructing the AST we attach Span information.
    """

    def _wrapper(*args, **kwargs):
        # Assumes 0th arg is self and gets source_name from object.
        sn = args[0].source_name
        # Assumes 1st arg is an ANTLR parser context.
        ctx = args[1]
        ast = f(*args, **kwargs)
        line, col = ctx.getSourceInterval()
        sp = Span(sn, line, col)
        if isinstance(ast, tvm.relay.expr.TupleWrapper):
            ast = ast.astuple()
        tvm.ir._ffi_api.NodeSetSpan(ast, sp)
        return ast
    return _wrapper

# TODO(@jmp): Use https://stackoverflow.com/q/13889941
# to figure out how to get ANTLR4 to be more unhappy about syntax errors
class ParseTreeToRelayIR(RelayVisitor):
    """Parse Relay text format into Relay IR."""

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name
        self.module = IRModule({})  # type: IRModule

        # Adding an empty scope allows naked lets without pain.
        self.var_scopes = deque([deque()])       # type: Scopes[expr.Var]
        self.global_vars = {}                    # type: Scope[expr.GlobalVar]
        self.type_var_scopes = deque([deque()])  # type: Scopes[ty.TypeVar]
        self.global_type_vars = {}               # type: Scope[expr.GlobalVar]
        self.graph_expr = []                     # type: List[expr.Expr]

        super(ParseTreeToRelayIR, self).__init__()


    def enter_var_scope(self) -> None:
        """Enter a new Var scope so it can be popped off later."""
        self.var_scopes.appendleft(deque())

    def exit_var_scope(self) -> Scope[expr.Var]:
        """Pop off the current Var scope and return it."""
        return self.var_scopes.popleft()

    def mk_var(self, name: str, typ: ty.Type = None):
        """Create a new Var and add it to the Var scope."""
        var = expr.Var(name, typ)
        self.var_scopes[0].appendleft((name, var))
        return var

    def mk_global_var(self, name: str) -> expr.GlobalVar:
        """Create a new GlobalVar and add it to the GlobalVar scope."""
        if name in self.global_vars:
            raise ParseError("duplicate global var \"{0}\"".format(name))
        var = expr.GlobalVar(name)
        self.global_vars[name] = var
        return var

    def enter_type_param_scope(self) -> None:
        """Enter a new TypeVar scope so it can be popped off later."""
        self.type_var_scopes.appendleft(deque())

    def exit_type_param_scope(self) -> Scope[ty.TypeVar]:
        """Pop off the current TypeVar scope and return it."""
        return self.type_var_scopes.popleft()

    def mk_typ(self, name: str, kind: ty.TypeKind) -> ty.TypeVar:
        """Create a new TypeVar and add it to the TypeVar scope."""
        typ = ty.TypeVar(name, kind)
        self.type_var_scopes[0].append((name, typ))
        return typ

    def mk_global_typ_var(self, name, kind):
        # (str, ty.Kind) -> ty.GlobalTypeVar
        """Create a new TypeVar and add it to the TypeVar scope."""
        typ = ty.GlobalTypeVar(name, kind)
        self._check_existing_typ_expr(name, typ)
        self.global_type_vars[name] = typ
        return typ

    # TODO(weberlo): rethink whether we should have type constructors mixed with type vars.
    def mk_global_typ_cons(self, name, cons):
        self._check_existing_typ_expr(name, cons)
        self.global_type_vars[name] = cons

    def _check_existing_typ_expr(self, name, new_expr):
        if name in self.global_type_vars:
            new_typ_name = self._type_expr_name(new_expr)
            existing_typ_name = self._type_expr_name(self.global_type_vars[name])
            raise ParseError(
                "{0} `{1}` conflicts with existing {2}".format(new_typ_name,\
                                                                name, existing_typ_name))

    def _type_expr_name(self, e):
        if isinstance(e, adt.Constructor):
            return "`{0}` ADT constructor".format(e.belong_to.name_hint)
        if isinstance(e, ty.GlobalTypeVar):
            if e.kind == ty.TypeKind.AdtHandle:
                return "ADT definition"
        return "function definition"

    def visitProjection(self, ctx):
        return expr.TupleGetItem(self.visit(ctx.expr()), self.visit(ctx.NAT()))

    def visitTerminal(self, node) -> Union[expr.Expr, int, float]:
        """Visit lexer tokens that aren't ignored or visited by other functions."""
        node_type = node.getSymbol().type
        node_text = node.getText()

        if node_type == RelayLexer.NAT:
            return int(node_text)
        if node_type == RelayLexer.FLOAT:
            return float(node_text[:-1])
        if node_type == RelayLexer.BOOL_LIT:
            if node_text == "True":
                return True
            if node_text == "False":
                return False
            raise ParseError("unrecognized BOOL_LIT: `{}`".format(node_text))
        if node_type == RelayLexer.QUOTED_STRING:
            return literal_eval(node_text)
        raise ParseError("unhandled terminal \"{0}\" of type `{1}`".format(node_text, node_type))

    def visitGeneralIdent(self, ctx):
        name = ctx.getText()
        # Look through all type prefixes for a match.
        for type_prefix in TYPE_PREFIXES:
            if name.startswith(type_prefix):
                return ty.scalar_type(name)
        # Next, look it up in the local then global type params.
        type_expr = lookup(self.type_var_scopes, name)
        if type_expr is None:
            type_expr = self.global_type_vars.get(name, None)
        if type_expr is not None:
            # Zero-arity constructor calls fall into the general ident case, so in that case,
            # we construct a constructor call with no args.
            if isinstance(type_expr, adt.Constructor) and not type_expr.inputs:
                type_expr = expr.Call(type_expr, [])
            return type_expr
        # Check if it's an operator.
        op_name = ".".join([name.getText() for name in ctx.CNAME()])
        if op_name in FUNC_OPS:
            return FuncOp(FUNC_OPS[op_name])
        return ExprOp(op.get(op_name))

    def visitGlobalVar(self, ctx):
        var_name = ctx.CNAME().getText()
        global_var = self.global_vars.get(var_name, None)
        if global_var is None:
            raise ParseError("unbound global var `{0}`".format(var_name))
        return global_var

    def visitLocalVar(self, ctx):
        var_name = ctx.CNAME().getText()
        local_var = lookup(self.var_scopes, var_name)
        if local_var is None:
            raise ParseError("unbound local var `{0}`".format(var_name))
        return local_var

    def visitGraphVar(self, ctx):
        return self.graph_expr[int(ctx.NAT().getText())]

    def visit_list(self, ctx_list) -> List[Any]:
        """"Visit a list of contexts."""
        assert isinstance(ctx_list, list)

        return [self.visit(ctx) for ctx in ctx_list]

    def getTypeExpr(self, ctx: Optional[RelayParser.TypeExprContext]) -> Optional[ty.Type]:
        """Return a (possibly None) Relay type."""
        if ctx is None:
            return None

        return self.visit(ctx)

    def visitProg(self, ctx: RelayParser.ProgContext) -> Union[expr.Expr, IRModule]:
        self.meta = None
        if ctx.METADATA():
            header, data = str(ctx.METADATA()).split("\n", 1)
            assert header == "METADATA:"
            self.meta = tvm.ir.load_json(data)
        if ctx.defn():
            self.visit_list(ctx.defn())
            return self.module

        if ctx.expr():
            return self.visit(ctx.expr())

        return self.module

    # Exprs
    def visitOpIdent(self, ctx) -> op.Op:
        op_name = ".".join([name.getText() for name in ctx.CNAME()])
        if op_name in FUNC_OPS:
            return FuncOp(FUNC_OPS[op_name])
        return ExprOp(op.get(op_name))

    # pass through
    def visitParen(self, ctx: RelayParser.ParenContext) -> expr.Expr:
        return self.visit(ctx.expr())

    # pass through
    def visitTypeParen(self, ctx: RelayParser.TypeParenContext) -> expr.Expr:
        return self.visit(ctx.typeExpr())

    # pass through
    def visitBody(self, ctx: RelayParser.BodyContext) -> expr.Expr:
        return self.visit(ctx.expr())

    def visitScalarFloat(self, ctx: RelayParser.ScalarFloatContext) -> expr.Constant:
        return expr.const(self.visit(ctx.FLOAT()))

    def visitScalarInt(self, ctx: RelayParser.ScalarIntContext) -> expr.Constant:
        return expr.const(self.visit(ctx.NAT()))

    def visitScalarBool(self, ctx: RelayParser.ScalarBoolContext) -> expr.Constant:
        return expr.const(self.visit(ctx.BOOL_LIT()))

    def visitNeg(self, ctx: RelayParser.NegContext) -> Union[expr.Constant, expr.Call]:
        val = self.visit(ctx.expr())
        if isinstance(val, expr.Constant) and val.data.asnumpy().ndim == 0:
            # fold Neg in for scalars
            return expr.const(-val.data.asnumpy().item())

        return op.negative(val)

    def visitTuple(self, ctx: RelayParser.TupleContext) -> expr.Tuple:
        tup = self.visit_list(ctx.expr())
        return expr.Tuple(tup)

    def visitLet(self, ctx: RelayParser.LetContext) -> expr.Let:
        """Desugar various sequence constructs to Relay Let nodes."""

        if ctx.var() is None:
            # anonymous identity
            ident = "_"
            typ = None
            var = self.mk_var(ident, typ)
        else:
            var = self.visitVar(ctx.var())

        self.enter_var_scope()
        value = self.visit(ctx.expr(0))
        self.exit_var_scope()

        body = self.visit(ctx.expr(1))

        return expr.Let(var, value, body)

    def visitBinOp(self, ctx: RelayParser.BinOpContext) -> expr.Call:
        """Desugar binary operators."""
        arg0, arg1 = self.visit_list(ctx.expr())
        relay_op = BINARY_OPS.get(ctx.op.type)

        if relay_op is None:
            raise ParseError("unimplemented binary op.")

        return relay_op(arg0, arg1)

    @spanify
    def visitVar(self, ctx: RelayParser.VarContext) -> expr.Var:
        """Visit a single variable."""
        ident = ctx.localVar()

        if ident is None:
            raise ParseError("only local ids may be used in vars.")

        typeExpr = self.getTypeExpr(ctx.typeExpr())

        return self.mk_var(ident.getText()[1:], typeExpr)

    def visitVarList(self, ctx: RelayParser.VarListContext) -> List[expr.Var]:
        return self.visit_list(ctx.var())

    # TODO: support a larger class of values than just Relay exprs
    def visitAttr(self, ctx: RelayParser.AttrContext) -> Tuple[str, expr.Expr]:
        return (ctx.CNAME().getText(), self.visit(ctx.expr()))

    def visitArgNoAttr(self, ctx: RelayParser.ArgNoAttrContext):
        return (self.visit_list(ctx.varList().var()), None)

    def visitAttrSeq(self, ctx: RelayParser.AttrSeqContext) -> Dict[str, expr.Expr]:
        return dict(self.visit_list(ctx.attr()))

    def visitArgWithAttr(self, ctx: RelayParser.AttrSeqContext) \
        -> Tuple[List[expr.Var], Dict[str, expr.Expr]]:
        return (self.visit_list(ctx.var()), self.visitAttrSeq(ctx.attrSeq()))

    def visitArgList(self, ctx: RelayParser.ArgListContext) \
            -> Tuple[Optional[List[expr.Var]], Optional[Dict[str, expr.Expr]]]:
        var_list = self.visit(ctx.varList()) if ctx.varList() else None
        attr_list = self.visit(ctx.attrList()) if ctx.attrList() else None
        return (var_list, attr_list)

    def visitMeta(self, ctx: RelayParser.MetaContext):
        type_key = str(ctx.CNAME())
        index = int(self.visit(ctx.NAT()))
        return self.meta[type_key][index]

    def mk_func(
            self,
            ctx: Union[RelayParser.FuncContext, RelayParser.DefnContext]) \
            -> function.Function:
        """Construct a function from either a Func or Defn."""
        # Enter var scope early to put params in scope.
        self.enter_var_scope()
        # Capture type params in params.
        self.enter_type_param_scope()
        type_params = ctx.typeParamList()

        if type_params is not None:
            type_params = type_params.typeExpr()
            assert type_params
            for ty_param in type_params:
                name = ty_param.getText()
                self.mk_typ(name, ty.TypeKind.Type)

        var_list, attr_list = self.visit(ctx.argList())
        if var_list is None:
            var_list = []
        ret_type = self.getTypeExpr(ctx.typeExpr())

        body = self.visit(ctx.body())
        # NB(@jroesch): you must stay in the type parameter scope until
        # after you exit the body, you can reference the type parameters
        # of your parent scopes.
        type_params = list(self.exit_type_param_scope())
        if type_params:
            _, type_params = zip(*type_params)
        self.exit_var_scope()

        attrs = tvm.ir.make_node("DictAttrs", **attr_list) if attr_list is not None else None
        return function.Function(var_list, body, ret_type, type_params, attrs)

    @spanify
    def visitFunc(self, ctx: RelayParser.FuncContext) -> function.Function:
        return self.mk_func(ctx)

    # TODO: how to set spans for definitions?
    # @spanify
    def visitFuncDefn(self, ctx: RelayParser.DefnContext) -> None:
        ident_name = ctx.globalVar().getText()[1:]
        ident = self.mk_global_var(ident_name)
        func = self.mk_func(ctx)
        self.module[ident] = func

    def handle_adt_header(
            self,
            ctx: Union[RelayParser.ExternAdtDefnContext, RelayParser.AdtDefnContext]):
        """Handles parsing of the name and type params of an ADT definition."""
        adt_name = ctx.generalIdent().getText()
        adt_var = self.mk_global_typ_var(adt_name, ty.TypeKind.AdtHandle)
        # parse type params
        type_params = ctx.typeParamList()
        if type_params is None:
            type_params = []
        else:
            type_params = [self.mk_typ(type_ident.getText(), ty.TypeKind.Type)
                           for type_ident in type_params.typeExpr()]
        return adt_var, type_params

    def visitExternAdtDefn(self, ctx: RelayParser.ExternAdtDefnContext):
        # TODO(weberlo): update this handler once extern is implemented
        self.enter_type_param_scope()
        adt_var, type_params = self.handle_adt_header(ctx)
        # update module being built
        self.module[adt_var] = adt.TypeData(adt_var, type_params, [])
        self.exit_type_param_scope()

    def visitAdtDefn(self, ctx: RelayParser.AdtDefnContext):
        self.enter_type_param_scope()
        adt_var, type_params = self.handle_adt_header(ctx)
        # parse constructors
        adt_cons_defns = ctx.adtConsDefnList()
        if adt_cons_defns is None:
            adt_cons_defns = []
        else:
            adt_cons_defns = adt_cons_defns.adtConsDefn()
        parsed_constructors = []
        for cons_defn in adt_cons_defns:
            inputs = [self.visit(inp) for inp in cons_defn.typeExpr()]
            cons_defn_name = cons_defn.constructorName().getText()
            cons_defn = adt.Constructor(cons_defn_name, inputs, adt_var)
            self.mk_global_typ_cons(cons_defn_name, cons_defn)
            parsed_constructors.append(cons_defn)
        # update module being built
        self.module[adt_var] = adt.TypeData(adt_var, type_params, parsed_constructors)
        self.exit_type_param_scope()

    def visitMatch(self, ctx: RelayParser.MatchContext):
        match_type = ctx.matchType().getText()
        if match_type == "match":
            complete_match = True
        elif match_type == "match?":
            complete_match = False
        else:
            raise RuntimeError("unknown match type {0}".format(match_type))

        match_data = self.visit(ctx.expr())
        match_clauses = ctx.matchClauseList()
        if match_clauses is None:
            match_clauses = []
        else:
            match_clauses = match_clauses.matchClause()
        parsed_clauses = []
        for clause in match_clauses:
            self.enter_var_scope()
            pattern = self.visit(clause.pattern())
            clause_body = self.visit(clause.expr())
            self.exit_var_scope()
            parsed_clauses.append(adt.Clause(pattern, clause_body))
        return adt.Match(match_data, parsed_clauses, complete=complete_match)

    def visitWildcardPattern(self, ctx: RelayParser.WildcardPatternContext):
        return adt.PatternWildcard()

    def visitVarPattern(self, ctx: RelayParser.VarPatternContext):
        text = ctx.localVar().getText()
        typ = ctx.typeExpr()
        if typ is not None:
            typ = self.visit(typ)
        var = self.mk_var(text[1:], typ=typ)
        return adt.PatternVar(var)

    def visitConstructorPattern(self, ctx: RelayParser.ConstructorPatternContext):
        constructor_name = ctx.constructorName().getText()
        constructor = self.global_type_vars[constructor_name]
        pattern_list = ctx.patternList()
        if pattern_list is None:
            patterns = []
        else:
            patterns = [self.visit(pattern) for pattern in pattern_list.pattern()]
        return adt.PatternConstructor(constructor, patterns)

    def visitTuplePattern(self, ctx: RelayParser.TuplePatternContext):
        return adt.PatternTuple([self.visit(pattern) for pattern in ctx.patternList().pattern()])

    def visitCallNoAttr(self, ctx: RelayParser.CallNoAttrContext):
        return (self.visit_list(ctx.exprList().expr()), None)

    def visitCallWithAttr(self, ctx: RelayParser.CallWithAttrContext):
        return (self.visit_list(ctx.expr()), self.visit(ctx.attrSeq()))

    def call(self, func, args, attrs, type_args):
        if isinstance(func, OpWrapper):
            return func(args, attrs, type_args)
        if isinstance(func, adt.Constructor):
            return func(*args)
        return expr.Call(func, args, attrs, type_args)

    @spanify
    def visitCall(self, ctx: RelayParser.CallContext) -> expr.Call:
        func = self.visit(ctx.expr())
        args, attrs = self.visit(ctx.callList())
        res = self.call(func, args, attrs, [])
        return res

    @spanify
    def visitIfElse(self, ctx: RelayParser.IfElseContext) -> expr.If:
        """Construct a Relay If node. Creates a new scope for each branch."""
        cond = self.visit(ctx.expr())

        self.enter_var_scope()
        true_branch = self.visit(ctx.body(0))
        self.exit_var_scope()

        self.enter_var_scope()
        false_branch = self.visit(ctx.body(1))
        self.exit_var_scope()

        return expr.If(cond, true_branch, false_branch)

    @spanify
    def visitGraph(self, ctx: RelayParser.GraphContext) -> expr.Expr:
        """Visit a graph variable assignment."""
        graph_nid = int(ctx.graphVar().getText()[1:])

        self.enter_var_scope()
        value = self.visit(ctx.expr(0))
        self.exit_var_scope()

        if graph_nid != len(self.graph_expr):
            raise ParseError(
                "expected new graph variable to be `%{}`,".format(len(self.graph_expr)) + \
                "but got `%{}`".format(graph_nid))
        self.graph_expr.append(value)

        kont = self.visit(ctx.expr(1))
        return kont

    # Types

    # pylint: disable=unused-argument
    def visitIncompleteType(self, ctx: RelayParser.IncompleteTypeContext) -> None:
        return None

    def visitTypeCallType(self, ctx: RelayParser.TypeCallTypeContext):
        func = self.visit(ctx.generalIdent())
        args = [self.visit(arg) for arg in ctx.typeParamList().typeExpr()]
        return ty.TypeCall(func, args)

    def visitParensShape(self, ctx: RelayParser.ParensShapeContext) -> int:
        return self.visit(ctx.shape())

    def visitShapeList(self, ctx: RelayParser.ShapeListContext) -> List[int]:
        return self.visit_list(ctx.shape())

    def visitTensor(self, ctx: RelayParser.TensorContext):
        return tuple(self.visit_list(ctx.expr()))

    def visitTensorType(self, ctx: RelayParser.TensorTypeContext) -> ty.TensorType:
        """Create a simple tensor type. No generics."""

        shape = self.visit(ctx.shapeList())
        dtype = self.visit(ctx.typeExpr())

        if not isinstance(dtype, ty.TensorType):
            raise ParseError("expected dtype to be a Relay base type.")

        dtype = dtype.dtype

        return ty.TensorType(shape, dtype)

    def visitTupleType(self, ctx: RelayParser.TupleTypeContext) -> ty.TupleType:
        return ty.TupleType(self.visit_list(ctx.typeExpr()))

    def visitFuncType(self, ctx: RelayParser.FuncTypeContext) -> ty.FuncType:
        types = self.visit_list(ctx.typeExpr())

        arg_types = types[:-1]
        ret_type = types[-1]

        return ty.FuncType(arg_types, ret_type, [], None)

def make_parser(data: str) -> RelayParser:
    """Construct a RelayParser a given data stream."""
    input_stream = InputStream(data)
    lexer = RelayLexer(input_stream)
    lexer.addErrorListener(StrictErrorListener(data))
    token_stream = CommonTokenStream(lexer)
    p = RelayParser(token_stream)
    p.addErrorListener(StrictErrorListener(data))
    return p

__source_name_counter__ = 0

class StrictErrorListener(ErrorListener):
    """This ErrorListener fail eagerly on all error, and report the program."""
    def __init__(self, text):
        self.text = text

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise Exception("Syntax Error in:\n" + self.text)

    def reportAmbiguity(self, recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs):
        raise Exception("Ambiguity Error in:\n" + self.text)

    def reportAttemptingFullContext(self,
                                    recognizer,
                                    dfa,
                                    startIndex,
                                    stopIndex,
                                    conflictingAlts,
                                    configs):
        raise Exception("Attempting Full Context in:\n" + self.text)

    def reportContextSensitivity(self, recognizer, dfa, startIndex, stopIndex, prediction, configs):
        raise Exception("Context Sensitivity in:\n" + self.text)

def fromtext(data: str, source_name: str = None) -> Union[expr.Expr, IRModule]:
    """Parse a Relay program."""
    if data == "":
        raise ParseError("cannot parse the empty string.")

    global __source_name_counter__

    if source_name is None:
        source_name = "source_file{0}".format(__source_name_counter__)

    if isinstance(source_name, str):
        source_name = SourceName(source_name)

    tree = make_parser(data).prog()
    return ParseTreeToRelayIR(source_name).visit(tree)
