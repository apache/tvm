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

import tvm

from . import module
from .base import Span, SourceName
from . import expr
from . import ty
from . import op

PYTHON_VERSION = sys.version_info.major
try:
    from .grammar.py3.RelayVisitor import RelayVisitor
    from .grammar.py3.RelayParser import RelayParser
    from .grammar.py3.RelayLexer import RelayLexer
except ImportError:
    raise Exeption("Couldn't find ANTLR parser. Try building with USE_ANTLR=ON.")

try:
    from antlr4 import InputStream, CommonTokenStream
    from antlr4.error.ErrorListener import ErrorListener
except ImportError:
    raise Exception("Couldn't find ANTLR runtime." +
                    "Try running `pip{version} install antlr4-python{version}-runtime`."
                    .format(version=PYTHON_VERSION))

sys.setrecursionlimit(10000)

class ParseError(Exception):
    """Exception type for parse errors."""

    def __init__(self, message):
        # type: (str) -> None
        super(ParseError, self).__init__()
        self.message = message

    def __repr__(self):
        return "ParseError({})".format(self.message)

    def __str__(self):
        return repr(self)

class OpWrapper:
    """Overload the __call__ for op."""
    pass

class ExprOp(OpWrapper):
    """Call an expr. The default, but does not handle attrs well."""
    def __init__(self, operator):
        self.operator = operator

    def __call__(self, args, attrs, type_args):
        try:
            return expr.Call(self.operator, args, attrs, type_args)
        except Exception:
            raise Exception(str(self.operator) + " " + str(attrs))

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
    "nn.global_max_pool2d": op.nn.global_max_pool2d,
    "nn.avg_pool2d": op.nn.avg_pool2d,
    "nn.global_avg_pool2d": op.nn.global_avg_pool2d,
    "nn.softmax": op.nn.softmax,
    "reshape": op.reshape,
    "nn.conv2d_transpose": op.nn.conv2d_transpose,
    "concatenate": op.concatenate,
    "nn.dropout": op.nn.dropout_raw,
    "zeros": op.zeros,
    "split": op.split,
}

TYPE_PREFIXES = [
    "int",
    "uint",
    "float",
    "bool",
]

T = ty.TypeVar("T")
# Scope = Deque[Tuple[str, T]]
# Scopes = Deque[Scope[T]]

def lookup(scopes, name):
    # type: (Scopes[T], str) -> Optional[T]
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
        ast.set_span(sp)
        return ast
    return _wrapper

# TODO(@jmp): Use https://stackoverflow.com/q/13889941
# to figure out how to get ANTLR4 to be more unhappy about syntax errors
class ParseTreeToRelayIR(RelayVisitor):
    """Parse Relay text format into Relay IR."""

    def __init__(self, source_name):
        # type: (str) -> None
        self.source_name = source_name
        self.module = module.Module({})   # type: module.Module

        # Adding an empty scope allows naked lets without pain.
        self.var_scopes = deque([deque()])          # type: Scopes[expr.Var]
        self.global_var_scope = deque()             # type: Scope[expr.GlobalVar]
        self.type_param_scopes = deque([deque()])   # type: Scopes[ty.TypeVar]
        self.graph_expr = []                        # type: List[expr.Expr]

        super(ParseTreeToRelayIR, self).__init__()


    def enter_var_scope(self):
        # type: () -> None
        """Enter a new Var scope so it can be popped off later."""

        self.var_scopes.appendleft(deque())

    def exit_var_scope(self):
        # type: () -> Scope[expr.Var]
        """Pop off the current Var scope and return it."""

        return self.var_scopes.popleft()

    def mk_var(self, name, type_):
        # type: (str, ty.Type) -> expr.Var
        """Create a new Var and add it to the Var scope."""

        var = expr.Var(name, type_)
        self.var_scopes[0].appendleft((name, var))
        return var

    def mk_global_var(self, name):
        # type: (str) -> expr.GlobalVar
        """Create a new GlobalVar and add it to the GlobalVar scope."""

        var = expr.GlobalVar(name)
        self.global_var_scope.append((name, var))
        return var

    def enter_type_param_scope(self):
        # type: () -> None
        """Enter a new TypeVar scope so it can be popped off later."""

        self.type_param_scopes.appendleft(deque())

    def exit_type_param_scope(self):
        # type: () -> Scope[ty.TypeVar]
        """Pop off the current TypeVar scope and return it."""

        return self.type_param_scopes.popleft()

    def mk_typ(self, name, kind):
        # (str, ty.Kind) -> ty.TypeVar
        """Create a new TypeVar and add it to the TypeVar scope."""

        typ = ty.TypeVar(name, kind)
        self.type_param_scopes[0].appendleft((name, typ))
        return typ

    def visitProjection(self, ctx):
        return expr.TupleGetItem(self.visit(ctx.expr()), self.visit(ctx.NAT()))

    def visitTerminal(self, node):
        # type: (TerminalNode) -> Union[expr.Expr, int, float]
        """Visit lexer tokens that aren't ignored or visited by other functions."""

        node_type = node.getSymbol().type
        node_text = node.getText()
        name = node_text[1:]

        # variables
        if node_type == RelayLexer.GLOBAL_VAR:
            return lookup(deque([self.global_var_scope]), node_text[1:])
        if node_type == RelayLexer.LOCAL_VAR:
            # Remove the leading '%' and lookup the name.
            var = lookup(self.var_scopes, name)
            if var is None:
                raise ParseError("Couldn't resolve `{}`.".format(name))
            return var
        if node_type == RelayLexer.GRAPH_VAR:
            try:
                return self.graph_expr[int(name)]
            except IndexError:
                raise ParseError("Couldn't resolve `{}`".format(name))

        # data types
        if node_type == RelayLexer.NAT:
            return int(node_text)
        if node_type == RelayLexer.FLOAT:
            return float(node_text[:-1])
        if node_type == RelayLexer.BOOL_LIT:
            if node_text == "True":
                return True
            if node_text == "False":
                return False
            raise ParseError("Unrecognized BOOL_LIT: `{}`".format(node_text))
        if node_type == RelayLexer.QUOTED_STRING:
            return literal_eval(node_text)

        raise ParseError("todo: `{}`".format(node_text))

    def visit_list(self, ctx_list):
        # type: (List[ParserRuleContext]) -> List[Any]
        """"Visit a list of contexts."""
        assert isinstance(ctx_list, list)

        return [self.visit(ctx) for ctx in ctx_list]

    def getType_(self, ctx):
        # type: (Optional[RelayParser.Type_Context]) -> Optional[ty.Type]
        """Return a (possibly None) Relay type."""

        if ctx is None:
            return None

        return self.visit(ctx)

    def visitProg(self, ctx):
        self.meta = None
        if ctx.METADATA():
            header, data = str(ctx.METADATA()).split('\n', 1)
            assert header == "METADATA:"
            self.meta = tvm.load_json(data)
        # type: (RelayParser.ProgContext) -> Union[expr.Expr, module.Module]
        if ctx.defn():
            self.visit_list(ctx.defn())
            return self.module

        if ctx.expr():
            return self.visit(ctx.expr())

        return self.module

    # Exprs
    def visitOpIdent(self, ctx):
        # type: (RelayParser.OpIdentContext) -> op.Op
        op_name = ctx.CNAME().getText()
        if op_name in FUNC_OPS:
            return FuncOp(FUNC_OPS[op_name])
        return ExprOp(op.get(op_name))

    # pass through
    def visitParen(self, ctx):
        # type: (RelayParser.ParenContext) -> expr.Expr
        return self.visit(ctx.expr())

    # pass through
    def visitBody(self, ctx):
        # type: (RelayParser.BodyContext) -> expr.Expr
        return self.visit(ctx.expr())

    def visitScalarFloat(self, ctx):
        # type: (RelayParser.ScalarFloatContext) -> expr.Constant
        return expr.const(self.visit(ctx.FLOAT()))

    def visitScalarInt(self, ctx):
        # type: (RelayParser.ScalarIntContext) -> expr.Constant
        return expr.const(self.visit(ctx.NAT()))

    def visitScalarBool(self, ctx):
        # type: (RelayParser.ScalarBoolContext) -> expr.Constant
        return expr.const(self.visit(ctx.BOOL_LIT()))

    def visitNeg(self, ctx):
        # type: (RelayParser.NegContext) -> Union[expr.Constant, expr.Call]
        val = self.visit(ctx.expr())
        if isinstance(val, expr.Constant) and val.data.asnumpy().ndim == 0:
            # fold Neg in for scalars
            return expr.const(-val.data.asnumpy().item())

        return op.negative(val)

    def visitTuple(self, ctx):
        # type: (RelayParser.TupleContext) -> expr.Tuple
        tup = self.visit_list(ctx.expr())
        return expr.Tuple(tup)

    def visitLet(self, ctx):
        # type: (RelayParser.SeqContext) -> expr.Let
        """Desugar various sequence constructs to Relay Let nodes."""

        if ctx.var() is None:
            # anonymous identity
            ident = "_"
            type_ = None
            var = self.mk_var(ident, type_)
        else:
            var = self.visitVar(ctx.var())

        self.enter_var_scope()
        value = self.visit(ctx.expr(0))
        self.exit_var_scope()

        body = self.visit(ctx.expr(1))

        return expr.Let(var, value, body)

    def visitBinOp(self, ctx):
        # type: (RelayParser.BinOpContext) -> expr.Call
        """Desugar binary operators."""
        arg0, arg1 = self.visit_list(ctx.expr())
        relay_op = BINARY_OPS.get(ctx.op.type)

        if relay_op is None:
            raise ParseError("Unimplemented binary op.")

        return relay_op(arg0, arg1)

    @spanify
    def visitVar(self, ctx):
        # type: (RelayParser.VarContext) -> expr.Var
        """Visit a single variable."""
        ident = ctx.LOCAL_VAR()

        if ident is None:
            raise ParseError("Only local ids may be used in vars.")

        type_ = self.getType_(ctx.type_())

        return self.mk_var(ident.getText()[1:], type_)

    def visitVarList(self, ctx):
        # type: (RelayParser.VarListContext) -> List[expr.Var]
        return self.visit_list(ctx.var())

    # TODO: support a larger class of values than just Relay exprs
    def visitAttr(self, ctx):
        # type: (RelayParser.AttrContext) -> Tuple[str, expr.Expr]
        return (ctx.CNAME().getText(), self.visit(ctx.expr()))

    def visitArgNoAttr(self, ctx):
        return (self.visit_list(ctx.varList().var()), None)

    def visitAttrSeq(self, ctx):
        # type: (RelayParser.AttrListContext) -> Dict[str, expr.Expr]
        return dict(self.visit_list(ctx.attr()))

    def visitArgWithAttr(self, ctx):
        return (self.visit_list(ctx.var()), self.visitAttrSeq(ctx.attrSeq()))

    def visitArgList(self,
                     ctx    # type: RelayParser.ArgListContext
                    ):
        # type: (...) -> Tuple[Optional[List[expr.Var]], Optional[Dict[str, expr.Expr]]]
        var_list = self.visit(ctx.varList()) if ctx.varList() else None
        attr_list = self.visit(ctx.attrList()) if ctx.attrList() else None
        return (var_list, attr_list)

    def visitMeta(self, ctx):
        type_key = str(ctx.CNAME())
        index = int(self.visit(ctx.NAT()))
        return self.meta[type_key][index]

    def mk_func(self, ctx):
        # type: (Union[RelayParser.FuncContext, RelayParser.DefnContext]) -> expr.Function
        """Construct a function from either a Func or Defn."""

        # Enter var scope early to put params in scope.
        self.enter_var_scope()
        # Capture type params in params.
        self.enter_type_param_scope()
        type_params = ctx.typeParamList()

        if type_params is not None:
            type_params = type_params.ident()
            assert type_params
            for ty_param in type_params:
                name = ty_param.getText()
                self.mk_typ(name, ty.Kind.Type)

        var_list, attr_list = self.visit(ctx.argList())
        if var_list is None:
            var_list = []
        ret_type = self.getType_(ctx.type_())

        body = self.visit(ctx.body())
        # NB(@jroesch): you must stay in the type parameter scope until
        # after you exit the body, you can reference the type parameters
        # of your parent scopes.
        type_params = list(self.exit_type_param_scope())
        if type_params:
            _, type_params = zip(*type_params)
        self.exit_var_scope()

        attrs = tvm.make.node("DictAttrs", **attr_list) if attr_list is not None else None
        return expr.Function(var_list, body, ret_type, type_params, attrs)

    @spanify
    def visitFunc(self, ctx):
        # type: (RelayParser.FuncContext) -> expr.Function
        return self.mk_func(ctx)

    # TODO: how to set spans for definitions?
    # @spanify
    def visitDefn(self, ctx):
        # type: (RelayParser.DefnContext) -> None
        ident = ctx.ident().GLOBAL_VAR()
        if ident is None:
            raise ParseError("Only global ids may be used in `def`s.")
        ident_name = ident.getText()[1:]
        ident = self.mk_global_var(ident_name)
        self.module[ident] = self.mk_func(ctx)

    def visitCallNoAttr(self, ctx):
        return (self.visit_list(ctx.exprList().expr()), None)

    def visitCallWithAttr(self, ctx):
        return (self.visit_list(ctx.expr()), self.visit(ctx.attrSeq()))

    def call(self, func, args, attrs, type_args):
        if isinstance(func, OpWrapper):
            return func(args, attrs, type_args)
        return expr.Call(func, args, attrs, type_args)

    @spanify
    def visitCall(self, ctx):
        # type: (RelayParser.CallContext) -> expr.Call
        func = self.visit(ctx.expr())
        args, attrs = self.visit(ctx.callList())
        return self.call(func, args, attrs, [])

    @spanify
    def visitIfElse(self, ctx):
        # type: (RelayParser.IfElseContext) -> expr.If
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
    def visitGraph(self, ctx):
        # type: (RelayParser.GraphContext) -> expr.Expr
        """Visit a graph variable assignment."""
        graph_nid = int(ctx.GRAPH_VAR().getText()[1:])

        self.enter_var_scope()
        value = self.visit(ctx.expr(0))
        self.exit_var_scope()

        if graph_nid != len(self.graph_expr):
            raise ParseError(
                "Expected new graph variable to be `%{}`,".format(len(self.graph_expr)) + \
                "but got `%{}`".format(graph_nid))
        self.graph_expr.append(value)

        kont = self.visit(ctx.expr(1))
        return kont

    # Types

    # pylint: disable=unused-argument
    def visitIncompleteType(self, ctx):
        # type (RelayParser.IncompleteTypeContext) -> None:
        return None

    def visitTypeIdent(self, ctx):
        # type: (RelayParser.TypeIdentContext) -> Union[ty.TensorType, str]
        '''
        Handle type identifier.
        '''
        type_ident = ctx.CNAME().getText()

        # Look through all type prefixes for a match
        for type_prefix in TYPE_PREFIXES:
            if type_ident.startswith(type_prefix):
                return ty.scalar_type(type_ident)

        type_param = lookup(self.type_param_scopes, type_ident)
        if type_param is not None:
            return type_param

        raise ParseError("Unknown builtin type: {}".format(type_ident))

    # def visitCallType(self, ctx):
    #     # type: (RelayParser.CallTypeContext) -> Union[expr.Expr, ty.TensorType]
    #     ident_type = ctx.identType().CNAME().getText()

    #     args = self.visit_list(ctx.type_())

    #     if not args:
    #         raise ParseError("Type-level functions must have arguments!")

    #     func_type = TYPE_FUNCS.get(ident_type)(args)

    #     if func_type is None:
    #         raise ParseError("Unknown type-level function: `{}`".format(ident_type))
    #     else:
    #         return func_type

    def visitParensShape(self, ctx):
        # type: (RelayParser.ParensShapeContext) -> int
        return self.visit(ctx.shape())

    def visitShapeList(self, ctx):
        # type: (RelayParser.ShapeListContext) -> List[int]
        return self.visit_list(ctx.shape())

    def visitTensor(self, ctx):
        return tuple(self.visit_list(ctx.expr()))

    def visitTensorType(self, ctx):
        # type: (RelayParser.TensorTypeContext) -> ty.TensorType
        """Create a simple tensor type. No generics."""

        shape = self.visit(ctx.shapeList())
        dtype = self.visit(ctx.type_())

        if not isinstance(dtype, ty.TensorType):
            raise ParseError("Expected dtype to be a Relay base type.")

        dtype = dtype.dtype

        return ty.TensorType(shape, dtype)

    def visitTupleType(self, ctx):
        # type: (RelayParser.TupleTypeContext) -> ty.TupleType
        return ty.TupleType(self.visit_list(ctx.type_()))

    def visitFuncType(self, ctx):
        # type: (RelayParser.FuncTypeContext) -> ty.FuncType
        types = self.visit_list(ctx.type_())

        arg_types = types[:-1]
        ret_type = types[-1]

        return ty.FuncType(arg_types, ret_type, [], None)

def make_parser(data):
    # type: (str) -> RelayParser
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

def fromtext(data, source_name=None):
    # type: (str, str) -> Union[expr.Expr, module.Module]
    """Parse a Relay program."""
    if data == "":
        raise ParseError("Cannot parse the empty string.")

    global __source_name_counter__

    if source_name is None:
        source_name = "source_file{0}".format(__source_name_counter__)

    if isinstance(source_name, str):
        source_name = SourceName(source_name)

    tree = make_parser(data).prog()
    return ParseTreeToRelayIR(source_name).visit(tree)
