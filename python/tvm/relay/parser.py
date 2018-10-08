"""A parser for Relay's text format."""
from antlr4 import ParserRuleContext, InputStream, CommonTokenStream
from antlr4.tree.Tree import TerminalNode
from collections import deque
from typing import TypeVar, Deque, Tuple, Optional, Union, NamedTuple, List
import tvm
from tvm import relay
import sys
if sys.version_info.major < 3:
    from .grammar.py2.RelayVisitor import RelayVisitor
    from .grammar.py2.RelayParser import RelayParser
    from .grammar.py2.RelayLexer import RelayLexer
else:
    from .grammar.py3.RelayVisitor import RelayVisitor
    from .grammar.py3.RelayParser import RelayParser
    from .grammar.py3.RelayLexer import RelayLexer

BINARY_OPS = {
    RelayParser.MUL: relay.multiply,
    RelayParser.DIV: relay.divide,
    RelayParser.ADD: relay.add,
    RelayParser.SUB: relay.subtract,
    RelayParser.LT: relay.less,
    RelayParser.GT: relay.greater,
    RelayParser.LE: relay.less_equal,
    RelayParser.GE: relay.greater_equal,
    RelayParser.EQ: relay.equal,
    RelayParser.NE: relay.not_equal,
}

Program = NamedTuple("Program", [("ast", relay.Expr), ("env", relay.Environment)])

class ParseError(Exception):
    def __init__(self, message):
        # type: (str) -> None
        super(ParseError, self).__init__()
        self.message = message

T = TypeVar("T")
Scope = Deque[Tuple[str, T]]
Scopes = Deque[Scope[T]]

def lookup(scopes, name):
    # type: (Scopes[T], str) -> Optional[T]
    for scope in scopes:
        for n, val in scope:
            if n == name:
                return val
    return None

# TODO(@jmp): Use https://stackoverflow.com/q/13889941
# to figure out how to get ANTLR4 to be more unhappy about syntax errors
class ParseTreeToRelayIR(RelayVisitor):
    """Parse Relay text format into Relay IR."""

    def __init__(self):
        # type: () -> None
        self.env = relay.Environment({})   # type: relay.Environment

        # Adding an empty scope allows naked lets without pain.
        self.var_scopes = deque([deque()]) # type: Scopes[relay.Var]
        self.type_param_scopes = deque([deque()]) # type: Scopes[relay.TypeParam]

        super(ParseTreeToRelayIR, self).__init__()

    def enter_var_scope(self):
        # type: () -> None
        self.var_scopes.appendleft(deque())

    def exit_var_scope(self):
        # type: () -> Scope[relay.Var]
        return self.var_scopes.popleft()

    def mk_var(self, name):
        # type: (str) -> relay.Var
        var = relay.Var(name)
        self.var_scopes[0].appendleft((name, var))
        return var

    def enter_type_param_scope(self):
        # type: () -> None
        self.type_param_scopes.appendleft(deque())

    def exit_type_param_scope(self):
        # type: () -> Scope[relay.TypeParam]
        return self.type_param_scopes.popleft()

    def mk_typ(self, name, kind):
        # (str, relay.Kind) -> relay.TypeParam
        typ = relay.TypeParam(name, kind)
        self.type_param_scopes[0].appendleft((name, typ))
        return typ

    def visitTerminal(self, node):
        # type: (TerminalNode) -> Union[relay.Expr, int, float]
        """Visit lexer tokens that aren't ignored or visited by other functions."""

        node_type = node.getSymbol().type

        # variables
        if node_type == RelayLexer.GLOBAL_VAR:
            return relay.GlobalVar(node.getText()[1:])
        elif node_type == RelayLexer.VAR:
            name = node.getText()[1:]
            var = lookup(self.var_scopes, name)
            if var is None:
                raise ParseError("Couldn't resolve `{}`.".format(name))
            else:
                return var

        # data types
        elif node_type == RelayLexer.INT:
            return int(node.getText())
        elif node_type == RelayLexer.FLOAT:
            return float(node.getText())

        else:
            raise ParseError("todo: {}".format(node.getText()))

    def visit_list(self, ctx_list):
        # type: (List[ParserRuleContext]) -> List[relay.Expr]
        return [self.visit(ctx) for ctx in ctx_list]

    # TODO(@jmp): Include kind environment to set IncompleteType appropriately.
    def getType_(self, ctx):
        # type: (Optional[RelayParser.Type_Context]) -> relay.Type
        if ctx is None:
            return relay.IncompleteType()
        else:
            return self.visit(ctx)

    # Exprs

    # pass through
    def visitBody(self, ctx):
        # type: (RelayParser.BodyContext) -> relay.Expr
        return self.visit(ctx.expr())

    def visitScalarFloat(self, ctx):
        # type: (RelayParser.ScalarFloatContext) -> relay.Constant
        return relay.Constant(tvm.nd.array(self.visit(ctx.FLOAT())))

    def visitScalarInt(self, ctx):
        # type: (RelayParser.ScalarIntContext) -> relay.Constant
        return relay.Constant(tvm.nd.array(self.visit(ctx.INT())))

    def visitScalarBool(self, ctx):
        # type: (RelayParser.ScalarBoolContext) -> relay.Constant
        # return relay.Constant(tvm.nd.array(self.visit(ctx.BOOL_LIST())))
        raise ParseError("Unimplemented")

    def visitNeg(self, ctx):
        # type: (RelayParser.NegContext) -> Union[relay.Constant, relay.Call]
        val = self.visit(ctx.expr())
        if isinstance(val, relay.Constant) and val.data.asnumpy().ndim == 0:
            # fold Neg in for scalars
            return relay.Constant(tvm.nd.array(-val.data.asnumpy().item()))
        else:
            return relay.negative(val)

    def visitTuple(self, ctx):
        # type: (RelayParser.TupleContext) -> relay.Tuple
        tup = self.visit_list(ctx.expr())
        return relay.Tuple(tup)

    # Currently doesn't support mutable sequencing.
    def visitSeq(self, ctx):
        # type: (RelayParser.SeqContext) -> relay.Let
        if ctx.MUT() is not None:
            raise ParseError("Mutation is currently unsupported.")

        if ctx.ident() is None:
            # anonymous identity
            ident = self.mk_var("_")
        else:
            ident = ctx.ident().VAR()
            if ident is None:
                raise ParseError('Only local ids may be used in lets.')
            ident = self.mk_var(ident.getText()[1:])

        type_ = self.getType_(ctx.type_())

        self.enter_var_scope()
        value = self.visit(ctx.expr(0))
        self.exit_var_scope()
        
        body = self.visit(ctx.expr(1))

        return relay.Let(ident, value, body, type_)

    def visitBinOp(self, ctx):
        # type: (RelayParser.BinOpContext) -> relay.Call
        """Desugar binary operators."""
        arg0, arg1 = self.visit_list(ctx.expr())
        relay_op = BINARY_OPS.get(ctx.op.type)

        if relay_op is None:
            raise ParseError("Unimplemented binary op.")

        return relay_op(arg0, arg1)

    def visitParam(self, ctx):
        # type: (RelayParser.ParamContext) -> relay.Param
        ident = ctx.ident().VAR()

        if ident is None:
            raise ParseError('Only local ids may be used in params.')

        ident = self.mk_var(ident.getText()[1:])
        type_ = self.getType_(ctx.type_())

        return relay.Param(ident, type_)

    def visitParamList(self, ctx):
        # type: (RelayParser.ParamListContext) -> List[relay.Param]
        return self.visit_list(ctx.param())

    def visitFunc(self, ctx):
        # type: (RelayParser.FuncContext) -> relay.Function
        # Enter var scope early to put params in scope.
        self.enter_var_scope()
        # Capture type params in params.
        self.enter_type_param_scope()
        param_list = self.visit(ctx.paramList())
        ret_type = self.getType_(ctx.type_())

        type_params = list(self.exit_type_param_scope())
        if type_params:
            _, type_params = zip(*type_params)

        body = self.visit(ctx.body())
        self.exit_var_scope()

        return relay.Function(param_list, ret_type, body, type_params)

    # Types

def parse_expr(data):
    # type: (str) -> relay.Expr
    """Parse a Relay expression."""

    # try:
    # TODO add error handling here
    input_stream = InputStream(data)
    lexer = RelayLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = RelayParser(token_stream)
    tree = parser.expr()
    visitor = ParseTreeToRelayIR()
    return visitor.visit(tree)
    # except Exception as exn:
    #     raise ParseError("parser error: {}".format(exn))

def parse_prog(data):
    # type: (str) -> Program
    """Parse a Relay program."""

    # try:
    # TODO add error handling here
    input_stream = InputStream(data)
    lexer = RelayLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = RelayParser(token_stream)
    tree = parser.prog()
    visitor = ParseTreeToRelayIR()
    relay_ast = visitor.visit(tree)
    return Program(ast=relay_ast, env=visitor.env)
    # except Exception as exn:
    #     raise ParseError("parser error: {}".format(exn))

def parse_file(path):
    # type: (str) -> Program
    with open(path, 'r') as f:
        return parse_prog(f.read())
