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
# pylint: disable=redefined-outer-name,missing-docstring,invalid-name
# pylint: disable=useless-super-delegation,redefined-builtin
# pylint: disable=too-few-public-methods,too-many-arguments
class AST:
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__()
        self.lineno = lineno
        self.col_offset = col_offset
        self.end_lineno = end_lineno
        self.end_col_offset = end_col_offset


class mod(AST):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Module(mod):
    _FIELDS = ["body", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, body, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.body = body


class Interactive(mod):
    _FIELDS = ["body", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, body, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.body = body


class Expression(mod):
    _FIELDS = ["body", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, body, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.body = body


class stmt(AST):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class FunctionDef(stmt):
    _FIELDS = [
        "name",
        "args",
        "body",
        "decorator_list",
        "returns",
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
    ]

    def __init__(
        self,
        name,
        args,
        body,
        decorator_list,
        returns,
        lineno,
        col_offset,
        end_lineno,
        end_col_offset,
    ):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.name = name
        self.args = args
        self.body = body
        self.decorator_list = decorator_list
        self.returns = returns


class ClassDef(stmt):
    _FIELDS = [
        "name",
        "bases",
        "keywords",
        "body",
        "decorator_list",
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
    ]

    def __init__(
        self,
        name,
        bases,
        keywords,
        body,
        decorator_list,
        lineno,
        col_offset,
        end_lineno,
        end_col_offset,
    ):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.name = name
        self.bases = bases
        self.keywords = keywords
        self.body = body
        self.decorator_list = decorator_list


class Return(stmt):
    _FIELDS = ["value", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, value, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value


class Delete(stmt):
    _FIELDS = ["targets", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, targets, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.targets = targets


class Assign(stmt):
    _FIELDS = ["targets", "value", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, targets, value, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.targets = targets
        self.value = value


class AugAssign(stmt):
    _FIELDS = ["target", "op", "value", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, target, op, value, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.target = target
        self.op = op
        self.value = value


class AnnAssign(stmt):
    _FIELDS = [
        "target",
        "annotation",
        "value",
        "simple",
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
    ]

    def __init__(
        self, target, annotation, value, simple, lineno, col_offset, end_lineno, end_col_offset
    ):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.target = target
        self.annotation = annotation
        self.value = value
        self.simple = simple


class For(stmt):
    _FIELDS = [
        "target",
        "iter",
        "body",
        "orelse",
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
    ]

    def __init__(self, target, iter, body, orelse, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.target = target
        self.iter = iter
        self.body = body
        self.orelse = orelse


class While(stmt):
    _FIELDS = ["test", "body", "orelse", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, test, body, orelse, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.test = test
        self.body = body
        self.orelse = orelse


class If(stmt):
    _FIELDS = ["test", "body", "orelse", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, test, body, orelse, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.test = test
        self.body = body
        self.orelse = orelse


class With(stmt):
    _FIELDS = ["items", "body", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, items, body, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.items = items
        self.body = body


class Raise(stmt):
    _FIELDS = ["exc", "cause", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, exc, cause, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.exc = exc
        self.cause = cause


class Try(stmt):
    _FIELDS = [
        "body",
        "handlers",
        "orelse",
        "finalbody",
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
    ]

    def __init__(
        self, body, handlers, orelse, finalbody, lineno, col_offset, end_lineno, end_col_offset
    ):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.body = body
        self.handlers = handlers
        self.orelse = orelse
        self.finalbody = finalbody


class Assert(stmt):
    _FIELDS = ["test", "msg", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, test, msg, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.test = test
        self.msg = msg


class Import(stmt):
    _FIELDS = ["names", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, names, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.names = names


class ImportFrom(stmt):
    _FIELDS = ["module", "names", "level", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, module, names, level, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.module = module
        self.names = names
        self.level = level


class Global(stmt):
    _FIELDS = ["names", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, names, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.names = names


class Nonlocal(stmt):
    _FIELDS = ["names", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, names, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.names = names


class Expr(stmt):
    _FIELDS = ["value", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, value, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value


class Pass(stmt):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Break(stmt):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Continue(stmt):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class expr(AST):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class BoolOp(expr):
    _FIELDS = ["op", "values", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, op, values, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.op = op
        self.values = values


class BinOp(expr):
    _FIELDS = ["left", "op", "right", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, left, op, right, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.left = left
        self.op = op
        self.right = right


class UnaryOp(expr):
    _FIELDS = ["op", "operand", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, op, operand, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.op = op
        self.operand = operand


class Lambda(expr):
    _FIELDS = ["args", "body", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, args, body, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.args = args
        self.body = body


class IfExp(expr):
    _FIELDS = ["test", "body", "orelse", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, test, body, orelse, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.test = test
        self.body = body
        self.orelse = orelse


class Dict(expr):
    _FIELDS = ["keys", "values", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, keys, values, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.keys = keys
        self.values = values


class Set(expr):
    _FIELDS = ["elts", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, elts, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.elts = elts


class ListComp(expr):
    _FIELDS = ["elt", "generators", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, elt, generators, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.elt = elt
        self.generators = generators


class SetComp(expr):
    _FIELDS = ["elt", "generators", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, elt, generators, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.elt = elt
        self.generators = generators


class DictComp(expr):
    _FIELDS = ["key", "value", "generators", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, key, value, generators, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.key = key
        self.value = value
        self.generators = generators


class GeneratorExp(expr):
    _FIELDS = ["elt", "generators", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, elt, generators, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.elt = elt
        self.generators = generators


class Yield(expr):
    _FIELDS = ["value", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, value, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value


class YieldFrom(expr):
    _FIELDS = ["value", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, value, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value


class Compare(expr):
    _FIELDS = ["left", "ops", "comparators", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, left, ops, comparators, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.left = left
        self.ops = ops
        self.comparators = comparators


class Call(expr):
    _FIELDS = ["func", "args", "keywords", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, func, args, keywords, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.func = func
        self.args = args
        self.keywords = keywords


class FormattedValue(expr):
    _FIELDS = [
        "value",
        "conversion",
        "format_spec",
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
    ]

    def __init__(
        self, value, conversion, format_spec, lineno, col_offset, end_lineno, end_col_offset
    ):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value
        self.conversion = conversion
        self.format_spec = format_spec


class JoinedStr(expr):
    _FIELDS = ["values", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, values, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.values = values


class Constant(expr):
    _FIELDS = ["value", "kind", "s", "n", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, value, kind, s, n, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value
        self.kind = kind
        self.s = s
        self.n = n


class NamedExpr(expr):
    _FIELDS = ["target", "value", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, target, value, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.target = target
        self.value = value


class Attribute(expr):
    _FIELDS = ["value", "attr", "ctx", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, value, attr, ctx, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value
        self.attr = attr
        self.ctx = ctx


class slice(AST):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Slice(slice):
    _FIELDS = ["lower", "upper", "step", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lower, upper, step, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.lower = lower
        self.upper = upper
        self.step = step


class ExtSlice(slice):
    _FIELDS = ["dims", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, dims, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.dims = dims


class Index(slice):
    _FIELDS = ["value", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, value, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value


class Subscript(expr):
    _FIELDS = ["value", "slice", "ctx", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, value, slice, ctx, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value
        self.slice = slice
        self.ctx = ctx


class Starred(expr):
    _FIELDS = ["value", "ctx", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, value, ctx, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.value = value
        self.ctx = ctx


class Name(expr):
    _FIELDS = ["id", "ctx", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, id, ctx, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.id = id
        self.ctx = ctx


class List(expr):
    _FIELDS = ["elts", "ctx", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, elts, ctx, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.elts = elts
        self.ctx = ctx


class Tuple(expr):
    _FIELDS = ["elts", "ctx", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, elts, ctx, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.elts = elts
        self.ctx = ctx


class expr_context(AST):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class AugLoad(expr_context):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class AugStore(expr_context):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Param(expr_context):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Suite(mod):
    _FIELDS = ["body", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, body, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.body = body


class Del(expr_context):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Load(expr_context):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Store(expr_context):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class boolop(AST):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class And(boolop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Or(boolop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class operator(AST):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Add(operator):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class BitAnd(operator):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class BitOr(operator):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class BitXor(operator):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Div(operator):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class FloorDiv(operator):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class LShift(operator):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Mod(operator):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Mult(operator):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class MatMult(operator):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Pow(operator):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class RShift(operator):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Sub(operator):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class unaryop(AST):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Invert(unaryop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Not(unaryop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class UAdd(unaryop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class USub(unaryop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class cmpop(AST):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Eq(cmpop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Gt(cmpop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class GtE(cmpop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class In(cmpop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Is(cmpop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class IsNot(cmpop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class Lt(cmpop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class LtE(cmpop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class NotEq(cmpop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class NotIn(cmpop):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class comprehension(AST):
    _FIELDS = [
        "target",
        "iter",
        "ifs",
        "is_async",
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
    ]

    def __init__(self, target, iter, ifs, is_async, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.target = target
        self.iter = iter
        self.ifs = ifs
        self.is_async = is_async


class excepthandler(AST):
    _FIELDS = ["lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)


class ExceptHandler(excepthandler):
    _FIELDS = ["type", "name", "body", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, type, name, body, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.type = type
        self.name = name
        self.body = body


class arguments(AST):
    _FIELDS = [
        "args",
        "vararg",
        "kwonlyargs",
        "kw_defaults",
        "kwarg",
        "defaults",
        "posonlyargs",
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
    ]

    def __init__(
        self,
        args,
        vararg,
        kwonlyargs,
        kw_defaults,
        kwarg,
        defaults,
        posonlyargs,
        lineno,
        col_offset,
        end_lineno,
        end_col_offset,
    ):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.args = args
        self.vararg = vararg
        self.kwonlyargs = kwonlyargs
        self.kw_defaults = kw_defaults
        self.kwarg = kwarg
        self.defaults = defaults
        self.posonlyargs = posonlyargs


class arg(AST):
    _FIELDS = ["arg", "annotation", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, arg, annotation, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.arg = arg
        self.annotation = annotation


class keyword(AST):
    _FIELDS = ["arg", "value", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, arg, value, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.arg = arg
        self.value = value


class alias(AST):
    _FIELDS = ["name", "asname", "lineno", "col_offset", "end_lineno", "end_col_offset"]

    def __init__(self, name, asname, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.name = name
        self.asname = asname


class withitem(AST):
    _FIELDS = [
        "context_expr",
        "optional_vars",
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
    ]

    def __init__(self, context_expr, optional_vars, lineno, col_offset, end_lineno, end_col_offset):
        super().__init__(lineno, col_offset, end_lineno, end_col_offset)
        self.context_expr = context_expr
        self.optional_vars = optional_vars


__all__ = [
    "AST",
    "mod",
    "Module",
    "Interactive",
    "Expression",
    "stmt",
    "FunctionDef",
    "ClassDef",
    "Return",
    "Delete",
    "Assign",
    "AugAssign",
    "AnnAssign",
    "For",
    "While",
    "If",
    "With",
    "Raise",
    "Try",
    "Assert",
    "Import",
    "ImportFrom",
    "Global",
    "Nonlocal",
    "Expr",
    "Pass",
    "Break",
    "Continue",
    "expr",
    "BoolOp",
    "BinOp",
    "UnaryOp",
    "Lambda",
    "IfExp",
    "Dict",
    "Set",
    "ListComp",
    "SetComp",
    "DictComp",
    "GeneratorExp",
    "Yield",
    "YieldFrom",
    "Compare",
    "Call",
    "FormattedValue",
    "JoinedStr",
    "Constant",
    "NamedExpr",
    "Attribute",
    "slice",
    "Slice",
    "ExtSlice",
    "Index",
    "Subscript",
    "Starred",
    "Name",
    "List",
    "Tuple",
    "expr_context",
    "AugLoad",
    "AugStore",
    "Param",
    "Suite",
    "Del",
    "Load",
    "Store",
    "boolop",
    "And",
    "Or",
    "operator",
    "Add",
    "BitAnd",
    "BitOr",
    "BitXor",
    "Div",
    "FloorDiv",
    "LShift",
    "Mod",
    "Mult",
    "MatMult",
    "Pow",
    "RShift",
    "Sub",
    "unaryop",
    "Invert",
    "Not",
    "UAdd",
    "USub",
    "cmpop",
    "Eq",
    "Gt",
    "GtE",
    "In",
    "Is",
    "IsNot",
    "Lt",
    "LtE",
    "NotEq",
    "NotIn",
    "comprehension",
    "excepthandler",
    "ExceptHandler",
    "arguments",
    "arg",
    "keyword",
    "alias",
    "withitem",
]
