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
"""Pretty Printers for lldb debugger.
Install the pretty printers by loading this file from .lldbinit:
  command script import ~/bin/lldb/tvm.py
Update the list of nodes for which debug information is displayed
by adding to the list below.
"""


import lldb
import lldb.formatters

g_indent = 0


def __lldb_init_module(debugger, _):
    # Only types that are supported by PrettyPrint() will be printed.
    for node in [
        "tvm::Array",
        "tvm::AttrFieldInfo",
        "tvm::Attrs",
        "tvm::BijectiveLayout",
        "tvm::Buffer",
        "tvm::Channel",
        "tvm::EnvFunc",
        "tvm::Expr",
        "tvm::GenericFunc",
        "tvm::Integer",
        "tvm::IterVar",
        "tvm::IterVarAttr",
        "tvm::IterVarRelation",
        "tvm::Layout",
        "tvm::Map",
        "tvm::Map",
        "tvm::MemoryInfo",
        "tvm::Operation",
        "tvm::Range",
        "tvm::Schedule",
        "tvm::Stage",
        "tvm::Stmt",
        "tvm::Target",
        "tvm::Tensor",
        "tvm::TensorIntrin",
        "tvm::TensorIntrinCall",
        "tvm::TypedEnvFunc",
        "tvm::tir::Var",
        "tvm::ir::CommReducer",
        "tvm::ir::FunctionRef",
        "tvm::relay::BaseTensorType",
        "tvm::relay::CCacheKey",
        "tvm::relay::CCacheValue",
        "tvm::relay::CachedFunc",
        "tvm::relay::Call",
        "tvm::relay::Clause",
        "tvm::relay::Closure",
        "tvm::relay::CompileEngine",
        "tvm::relay::Constant",
        "tvm::relay::Constructor",
        "tvm::relay::ConstructorValue",
        "tvm::relay::Expr",
        "tvm::relay::FuncType",
        "tvm::relay::Function",
        "tvm::relay::GlobalTypeVar",
        "tvm::relay::GlobalVar",
        "tvm::relay::Id",
        "tvm::relay::If",
        "tvm::relay::IncompleteType",
        "tvm::relay::InterpreterState",
        "tvm::relay::Let",
        "tvm::relay::Match",
        "tvm::relay::Module",
        "tvm::relay::NamedNDArray",
        "tvm::relay::Op",
        "tvm::relay::Pattern",
        "tvm::relay::PatternConstructor",
        "tvm::relay::PatternTuple",
        "tvm::relay::PatternVar",
        "tvm::relay::PatternWildcard",
        "tvm::relay::RecClosure",
        "tvm::relay::RefCreate",
        "tvm::relay::RefRead",
        "tvm::relay::RefType",
        "tvm::relay::RefValue",
        "tvm::relay::RefWrite",
        "tvm::relay::SourceName",
        "tvm::relay::Span",
        "tvm::relay::TempExpr",
        "tvm::relay::TensorType",
        "tvm::relay::Tuple",
        "tvm::relay::TupleGetItem",
        "tvm::relay::TupleType",
        "tvm::relay::Type",
        "tvm::relay::TypeCall",
        "tvm::relay::TypeConstraint",
        "tvm::relay::TypeData",
        "tvm::relay::TypeRelation",
        "tvm::relay::TypeReporter",
        "tvm::relay::TypeVar",
        "tvm::relay::Value",
        "tvm::relay::Var",
        "tvm::relay::alter_op_layout::LayoutAlternatedExpr",
        "tvm::relay::alter_op_layout::TransformMemorizer",
        "tvm::relay::fold_scale_axis::Message",
        "tvm::relay::fold_scale_axis:BackwardTransformer",
    ]:
        debugger.HandleCommand(
            "type summary add -F tvm.NodeRef_SummaryProvider {node} -w tvm".format(node=node)
        )
    debugger.HandleCommand("command script add -f tvm.PrettyPrint pp")
    debugger.HandleCommand("type category enable tvm")


def _log(logger, fmt, *args, **kwargs):
    global g_indent
    logger >> " " * g_indent + fmt.format(*args, **kwargs)


def _GetContext(debugger):
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    thread = process.GetThreadAtIndex(0)
    return thread.GetSelectedFrame()


def PrettyPrint(debugger, command, result, internal_dict):
    ctx = _GetContext(debugger)
    rc = ctx.EvaluateExpression("tvm::PrettyPrint({command})".format(command=command))
    result.AppendMessage(str(rc))


class EvaluateError(Exception):
    def __init__(self, error):
        super(Exception, self).__init__(str(error))


def _EvalExpression(logger, ctx, expr, value_name):
    _log(logger, "Evaluating {expr}".format(expr=expr))

    rc = ctx.EvaluateExpression(expr)
    err = rc.GetError()
    if err.Fail():
        _log(logger, "_EvalExpression failed: {err}".format(err=err))
        raise EvaluateError(err)
    _log(logger, "_EvalExpression success: {typename}".format(typename=rc.GetTypeName()))
    return rc


def _EvalExpressionAsString(logger, ctx, expr):
    result = _EvalExpression(logger, ctx, expr, None)
    return result.GetSummary() or result.GetValue() or "--"


def _EvalAsNodeRef(logger, ctx, value):
    return _EvalExpressionAsString(logger, ctx, "tvm::PrettyPrint({name})".format(name=value.name))


def NodeRef_SummaryProvider(value, _):
    global g_indent
    g_indent += 2

    try:
        if not value or not value.IsValid():
            return "<invalid>"

        lldb.formatters.Logger._lldb_formatters_debug_level = 0
        logger = lldb.formatters.Logger.Logger()

        ctx = _GetContext(lldb.debugger)
        return _EvalAsNodeRef(logger, ctx, value)

    except EvaluateError as e:
        return str(e)
    finally:
        g_indent -= 2
