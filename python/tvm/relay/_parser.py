
# pylint: disable=invalid-name, unused-import
"""A parser for Relay's text format."""
from __future__ import absolute_import

import sys

from collections import deque
from typing import TypeVar, Deque, Tuple, Optional, Union, NamedTuple, List, Callable, Any

from . import module
from . import expr
from . import ty
from . import op

class ParseError(Exception):
    """Exception type for parse errors."""

    def __init__(self, message):
        # type: (str) -> None
        super(ParseError, self).__init__()
        self.message = message

PYTHON_VERSION = sys.version_info.major
try:
    if PYTHON_VERSION == 2:
        from .grammar.py2.RelayVisitor import RelayVisitor
        from .grammar.py2.RelayParser import RelayParser
        from .grammar.py2.RelayLexer import RelayLexer
    else:
        from .grammar.py3.RelayVisitor import RelayVisitor
        from .grammar.py3.RelayParser import RelayParser
        from .grammar.py3.RelayLexer import RelayLexer
except ImportError:
    raise ParseError("Couldn't find ANTLR parser. Try building with USE_ANTLR=ON.")

try:
    from antlr4 import ParserRuleContext, InputStream, CommonTokenStream
    from antlr4.tree.Tree import TerminalNode
except ImportError:
    raise ParseError("Couldn't find ANTLR runtime." +
                     "Try running `pip{} install antlr4-python{}-runtime`."
                     .format(PYTHON_VERSION, PYTHON_VERSION))

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

TYPE_PREFIXES = [
    "int",
    "uint",
    "float",
    "bool",
]

T = TypeVar("T")
Scope = Deque[Tuple[str, T]]
Scopes = Deque[Scope[T]]

def lookup(scopes, name):
    # type: (Scopes[T], str) -> Optional[T]
    """Look up `name` in `scopes`."""

    for scope in scopes:
        for key, val in scope:
            if key == name:
                return val
    return None

# TODO(@jmp): Use https://stackoverflow.com/q/13889941
# to figure out how to get ANTLR4 to be more unhappy about syntax errors
class ParseTreeToRelayIR(RelayVisitor):
    """Parse Relay text format into Relay IR."""

    def __init__(self):
        # type: () -> None
        self.module = module.Module({})   # type: module.Module

        # Adding an empty scope allows naked lets without pain.
        self.var_scopes = deque([deque()]) # type: Scopes[expr.Var]
        self.type_param_scopes = deque([deque()]) # type: Scopes[ty.TypeVar]

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

    def visitTerminal(self, node):
        # type: (TerminalNode) -> Union[expr.Expr, int, float]
        """Visit lexer tokens that aren't ignored or visited by other functions."""

        node_type = node.getSymbol().type
        node_text = node.getText()

        # variables
        if node_type == RelayLexer.GLOBAL_VAR:
            return expr.GlobalVar(node_text[1:])
        elif node_type == RelayLexer.LOCAL_VAR:
            name = node_text[1:]
            var = lookup(self.var_scopes, name)
            if var is None:
                raise ParseError("Couldn't resolve `{}`.".format(name))

            return var

        # data types
        elif node_type == RelayLexer.INT:
            return int(node_text)
        elif node_type == RelayLexer.FLOAT:
            return float(node_text)
        elif node_type == RelayLexer.BOOL_LIT:
            if node_text == "True":
                return True
            elif node_text == "False":
                return False
            else:
                raise ParseError("Unrecognized BOOL_LIT: `{}`".format(node_text))

        else:
            raise ParseError("todo: {}".format(node_text))

    def visit_list(self, ctx_list):
        # type: (List[ParserRuleContext]) -> List[Any]
        """"Visit a list of contexts."""

        return [self.visit(ctx) for ctx in ctx_list]

    def getType_(self, ctx):
        # type: (Optional[RelayParser.Type_Context]) -> Optional[ty.Type]
        """Return a (possibly None) Relay type."""

        if ctx is None:
            return None

        return self.visit(ctx)

    def visitProg(self, ctx):
        # type: (RelayParser.ProgContext) -> Union[expr.Expr, env.Environment]
        if ctx.defn():
            self.visit_list(ctx.defn())
            return self.module

        return self.visit(ctx.expr())

    # Exprs

    def visitOpIdent(self, ctx):
        # type: (RelayParser.OpIdentContext) -> op.Op
        return op.get(ctx.CNAME().getText())

    # pass through
    def visitParens(self, ctx):
        # type: (RelayParser.ParensContext) -> expr.Expr
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
        return expr.const(self.visit(ctx.INT()))

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

    # Currently doesn't support mutable sequencing.
    def visitSeq(self, ctx):
        # type: (RelayParser.SeqContext) -> expr.Let
        """Desugar various sequence constructs to Relay Let nodes."""
        if ctx.MUT() is not None:
            raise ParseError("Mutation is currently unsupported.")

        if ctx.var() is None or ctx.var().ident() is None:
            # anonymous identity
            ident = "_"
            type_ = None
        else:
            local_var = ctx.var().ident().LOCAL_VAR()
            if local_var is None:
                raise ParseError('Only local ids may be used in `let`s.')
            ident = local_var.getText()[1:]
            type_ = self.getType_(ctx.var().type_())

        var = self.mk_var(ident, type_)

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

    def visitVar(self, ctx):
        # type: (RelayParser.VarContext) -> expr.Var
        ident = ctx.ident().LOCAL_VAR()

        if ident is None:
            raise ParseError('Only local ids may be used in params.')

        type_ = self.getType_(ctx.type_())

        return self.mk_var(ident.getText()[1:], type_)

    def visitVarList(self, ctx):
        # type: (RelayParser.VarListContext) -> List[expr.Var]
        return self.visit_list(ctx.var())

    def mk_func(self, ctx):
        # type: (Union[RelayParser.FuncContext, RelayParser.DefnContext]) -> Function
        """Construct a function from either a Func or Defn."""

        # Enter var scope early to put params in scope.
        self.enter_var_scope()
        # Capture type params in params.
        self.enter_type_param_scope()
        var_list = self.visit(ctx.varList())
        ret_type = self.getType_(ctx.type_())

        type_params = list(self.exit_type_param_scope())
        if type_params:
            _, type_params = zip(*type_params)

        body = self.visit(ctx.body())
        self.exit_var_scope()

        return expr.Function(var_list, body, ret_type, type_params) # type: ignore

    def visitFunc(self, ctx):
        # type: (RelayParser.FuncContext) -> expr.Function
        return self.mk_func(ctx)

    def visitDefn(self, ctx):
        # type: (RelayParser.DefnContext) -> None
        ident = ctx.ident().GLOBAL_VAR()
        if ident is None:
            raise ParseError('Only global ids may be used in `def`s.')
        ident = expr.GlobalVar(ident.getText()[1:])

        self.module[ident] = self.mk_func(ctx)

    def visitCall(self, ctx):
        # type: (RelayParser.CallContext) -> expr.Call
        visited_exprs = self.visit_list(ctx.expr())

        func = visited_exprs[0]
        args = visited_exprs[1:]

        return expr.Call(func, args, None, None)

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

    # Types

    # pylint: disable=unused-argument
    def visitIncompleteType(self, ctx):
        # type (RelayParser.IncompleteTypeContext) -> None:
        return None

    def visitIdentType(self, ctx):
        # type: (RelayParser.IdentTypeContext) -> Union[ty.TensorType, str]
        ident_type = ctx.CNAME().getText()

        # look through all type prefixes for a match
        for type_prefix in TYPE_PREFIXES:
            if ident_type.startswith(type_prefix):
                return ty.scalar_type(ident_type)

        raise ParseError("Unknown builtin type: {}".format(ident_type))

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

    def visitShapeSeq(self, ctx):
        # type: (RelayParser.ShapeSeqContext) -> List[int]
        return self.visit_list(ctx.shape())

    def visitTensorType(self, ctx):
        # type: (RelayParser.TensorTypeContext) -> ty.TensorType
        """Create a simple tensor type. No generics."""

        shape = self.visit(ctx.shapeSeq())
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
    token_stream = CommonTokenStream(lexer)
    return RelayParser(token_stream)

def fromtext(data):
    # type: (str) -> Union[expr.Expr, env.Environment]
    """Parse a Relay program."""
    tree = make_parser(data).prog()
    return ParseTreeToRelayIR().visit(tree)
