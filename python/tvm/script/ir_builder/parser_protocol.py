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
"""Shared structural contract for source-to-builder rewriting.

The source namespace selects language variant ``X``; shared module support uses ``I``.
For example, ``target()[index()] = value()`` lowers to
``X.setitem_(value=value(), target=target(), key=index())``. Python evaluates the
keywords in that order, preserving the assignment's RHS-first semantics.

Native frames own construction state. Language variant implementations live in
``builder.parser_protocol``; shared contract stubs here raise NotImplementedError.
Special syntax markers are documented in the special protocol section below.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tvm import ir as _ir

from .base import MISSING, AlreadyEmitted, IRBuilderFrame, SpanEntry

_Span = SpanEntry | _ir.Span | None


# --------------------------------------
# Section: control flow
# --------------------------------------
#
# ``with X.if_(cond):`` opens a conditional frame.
# ``with X.then_():`` enters its true branch.
# ``with X.else_():`` enters its false branch.
# ``with X.for_(loop) as i:`` binds a loop iterator.
# ``with X.while_(cond):`` builds a while loop.
# ``X.range_(start, stop)`` constructs a loop descriptor.
# ``X.break_()`` emits a loop exit.
# ``X.continue_()`` emits a loop continuation.


def if_(condition: Any, *, span: _Span = None) -> IRBuilderFrame:
    """Create the native conditional frame for a source if statement.

    Parameters
    ----------
    condition : Expr or bool
        Already-evaluated predicate; both branch bodies construct IR without testing it
        in Python.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    IRBuilderFrame
        A native conditional context manager. Relax exposes its same-named merged output
        as frame.var after exit.

    Notes
    -----
    Requires an active function region. Enter ``then_``/``else_`` frames inside it; each branch's
    lexical helper is defined and called inside that branch frame. TIRx permits a missing
    else; value-producing Relax branches must agree on their final binding name/type.
    Invalid context/condition/outputs produce builder diagnostics. Explicit constexpr
    conditions remain ordinary Python if statements.

    .. code:: python

        # Source
        if condition:
            T.evaluate(1)
        # Generated builder
        with X.if_(condition):
            with X.then_():
                def branch():
                    X.emit_(X.evaluate(1))
                branch()
    """
    raise NotImplementedError


def then_(*, span: _Span = None) -> IRBuilderFrame:
    """Create the true branch of the active conditional.

    Parameters
    ----------
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    IRBuilderFrame
        The native branch context manager; its exit finalizes its region.

    Notes
    -----
    Requires an active ``if_`` frame and records the true branch. Invalid ordering or repeated
    branches raise native builder errors. The frame stores its source location; statements
    keep their individual locations. Lexical helpers take zero explicit arguments and run
    inside the entered frame.

    .. code:: python

        # Source
        if condition:
            pass
        else:
            pass
        # Generated builder, inside X.if_(condition)
        with X.then_():
            def branch():
                pass
            branch()
    """
    raise NotImplementedError


def else_(*, span: _Span = None) -> IRBuilderFrame:
    """Create the false branch of the active conditional.

    Parameters
    ----------
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    IRBuilderFrame
        The native branch context manager; its exit finalizes its region.

    Notes
    -----
    Requires an active ``if_`` frame and records the false branch. Invalid ordering or repeated
    branches raise native builder errors. The frame stores its source location; statements
    keep their individual locations. Lexical helpers take zero explicit arguments and run
    inside the entered frame.

    .. code:: python

        # Source
        if condition:
            pass
        else:
            pass
        # Generated builder, inside X.if_(condition)
        with X.else_():
            def branch():
                pass
            branch()
    """
    raise NotImplementedError


def for_(
    iterable: Any, *, names: str | Sequence[str] | None = None, span: _Span = None
) -> IRBuilderFrame:
    """Configure and return the native iteration frame for a source for loop.

    Parameters
    ----------
    iterable : ForFrame or range
        Already-evaluated iteration specification. TIRx converts a Python range to its
        native serial frame.
    names : str or sequence of str, optional
        Source target names, including at most one ``*starred`` group. None (the default)
        retains constructor defaults. Native configuration expands/validates them before
        entry; names never control the entry return shape.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    IRBuilderFrame
        The same configured ForFrame. Entry returns the native variable for one
        dimension, or the original variable sequence otherwise.

    Notes
    -----
    TIRx requires an active primitive function before entry. Variables already have final
    names at entry. Simple scalar targets use the entry result; generated tuple,
    list or starred targets use the stable frame.vars sequence for unpacking.
    Invalid iterable/names raise TypeError, ValueError or native errors. Relax rejects imperative
    loops. A frame stores its location before deferred body finalization.

    .. code:: python

        # Source
        for i in range(n):
            T.evaluate(i)
        # Generated builder
        with X.for_(X.range_(n), names=("i",)) as i:
            X.emit_(X.evaluate(i))
    """
    raise NotImplementedError


def while_(condition: Any, *, span: _Span = None) -> IRBuilderFrame:
    """Create a native while-loop frame.

    Parameters
    ----------
    condition : Expr or bool
        Loop predicate expression, constructed once and evaluated by the IR at runtime.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    IRBuilderFrame
        A context manager whose exit completes the loop body.

    Notes
    -----
    TIRx requires an active function region. Invalid predicate/context produces native
    errors; Relax raises TypeError. This does not repeatedly execute the Python body. The
    frame owns its stored source span.

    .. code:: python

        # Source
        while condition:
            T.evaluate(1)
        # Generated builder
        with X.while_(condition):
            X.emit_(X.evaluate(1))
    """
    raise NotImplementedError


def range_(*args: Any, annotations: dict[str, Any] | None = None) -> IRBuilderFrame:
    """Normalize builtin range syntax to a native serial loop frame.

    Parameters
    ----------
    args : Expr or int
        One to three already-evaluated bounds: stop; start, stop; or start, stop, step.
        Omitted start is zero and omitted step uses the native default.
    annotations : dict, optional
        Native loop annotations. None (the default) adds none.

    Returns
    -------
    IRBuilderFrame
        An unentered serial loop frame; ``for_`` configures names and source span.

    Notes
    -----
    TIRx validates arity and rejects a literal zero step (TypeError/ValueError). Native
    bound/type diagnostics propagate. Relax raises TypeError. Only the resolved builtin
    range is normalized; unrelated host callables named range retain ordinary call behavior.

    .. code:: python

        # Source
        for i in range(2, n, 2):
            pass
        # Generated builder iteration specification
        X.range_(2, n, 2)
    """
    raise NotImplementedError


def break_(*, span: _Span = None) -> AlreadyEmitted[Any]:
    """Emit break for the enclosing language variant loop.

    Parameters
    ----------
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    AlreadyEmitted[Any]
        Receipt retaining the exact emitted statement; consuming it emits nothing again.

    Notes
    -----
    TIRx requires an enclosing native ForFrame or WhileFrame inside the current primitive
    function, emits break for that loop and raises ValueError otherwise. Relax raises
    TypeError. Explicit constexpr loops retain ordinary Python control flow.

    .. code:: python

        # Source
        break
        # Generated builder
        X.break_()
    """
    raise NotImplementedError


def continue_(*, span: _Span = None) -> AlreadyEmitted[Any]:
    """Emit continue for the enclosing language variant loop.

    Parameters
    ----------
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    AlreadyEmitted[Any]
        Receipt retaining the exact emitted statement; consuming it emits nothing again.

    Notes
    -----
    TIRx requires an enclosing native ForFrame or WhileFrame inside the current primitive
    function, emits continue for that loop and raises ValueError otherwise. Relax raises
    TypeError. Explicit constexpr loops retain ordinary Python control flow.

    .. code:: python

        # Source
        continue
        # Generated builder
        X.continue_()
    """
    raise NotImplementedError


# --------------------------------------
# Section: operator overloading
# --------------------------------------
#
# ``X.if_then_else_(c, a, b)`` selects an expression.
# ``X.and_(a, b)`` constructs conjunction.
# ``X.or_(a, b)`` constructs disjunction.
# ``X.not_(a)`` negates a condition.
# ``X.lt_(a, b)`` lowers ``a < b``.
# ``X.le_(a, b)`` lowers ``a <= b``.
# ``X.gt_(a, b)`` lowers ``a > b``.
# ``X.ge_(a, b)`` lowers ``a >= b``.
# ``X.eq_(a, b)`` lowers ``a == b``.
# ``X.ne_(a, b)`` lowers ``a != b``.


def if_then_else_(condition: Any, true_value: Any, false_value: Any) -> Any:
    """Construct conditional evaluation from eagerly constructed operands.

    Parameters
    ----------
    condition : Any
        Host boolean or scalar IR predicate.
    true_value : Any
        Already-constructed true arm.
    false_value : Any
        Already-constructed false arm.

    Returns
    -------
    Any
        The selected host value or native conditional expression.

    Notes
    -----
    No statement emission or frame entry occurs. Host predicates select an existing value;
    scalar IR uses conditional evaluation. Relax also accepts its expression arms. Operand
    construction remains eager; runtime evaluation follows language variant conditional semantics.
    Invalid combinations raise native type errors. Source-result handling supplies spans.

    .. code:: python

        # Source
        yes if condition else no
        # Generated builder
        X.if_then_else_(condition, yes, no)
    """
    raise NotImplementedError


def and_(*values: Any) -> Any:
    """Construct a conjunction from ordinary boolean or comparison operands.

    Parameters
    ----------
    *values : Any
        Eagerly constructed host or IR operands in source order. TIRx constructs
        primitive conjunctions; Relax selects primitive or tensor operations.

    Returns
    -------
    Any
        The host boolean or native conjunction expression.

    Notes
    -----
    Empty operands raise TypeError. Simple comparison chains accept only names
    and numeric literals, including signed literals. They lower directly to
    adjacent comparisons with this conjunction; complex chain operands are a
    source error before evaluation. There are no temporary operand bindings or
    substitution. Explicit constexpr code retains Python short-circuit behavior.

    .. code:: python

        # Source
        0 < i < 10
        # Generated builder
        X.and_(X.lt_(0, i), X.lt_(i, 10))
    """
    raise NotImplementedError


def or_(*values: Any) -> Any:
    """Construct a disjunction from already-evaluated operands.

    Parameters
    ----------
    values : Any
        One or more host or language variant boolean operands in source order.

    Returns
    -------
    Any
        Host result or language variant disjunction expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx supports scalar/vector predicates; Relax
    also supports tensor predicates. Empty operands raise TypeError; native type checks
    propagate. Explicit constexpr operands retain host short-circuit syntax in the generated
    program. Result spans are attached by shared source handling.

    .. code:: python

        # Source
        a or b
        # Generated builder
        X.or_(a, b)
    """
    raise NotImplementedError


def not_(value: Any) -> Any:
    """Negate a host or language variant boolean without coercing IR truth in Python.

    Parameters
    ----------
    value : Any
        Once-evaluated host boolean or primitive/tensor boolean expression.

    Returns
    -------
    Any
        Host boolean or native logical negation expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx constructs primitive negation; Relax also
    supports tensor negation. Invalid operand types propagate native errors. Shared source
    handling attaches the result span.

    .. code:: python

        # Source
        not value
        # Generated builder
        X.not_(value)
    """
    raise NotImplementedError


def lt_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Construct the < comparison in written operand order.

    Parameters
    ----------
    lhs : Expr or scalar
        Already-evaluated left operand.
    rhs : Expr or scalar
        Already-evaluated right operand.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Expr
        A native comparison expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx compares primitive operands;
    Relax selects tensor comparison if either operand is a nonprimitive IR
    expression, otherwise primitive comparison. Native dtype/shape errors propagate. Do not
    use Python object equality to compare these operands.

    .. code:: python

        # Source
        lhs < rhs
        # Generated builder
        X.lt_(lhs, rhs)
    """
    raise NotImplementedError


def le_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Construct the <= comparison in written operand order.

    Parameters
    ----------
    lhs : Expr or scalar
        Already-evaluated left operand.
    rhs : Expr or scalar
        Already-evaluated right operand.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Expr
        A native comparison expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx compares primitive operands;
    Relax selects tensor comparison if either operand is a nonprimitive IR
    expression, otherwise primitive comparison. Native dtype/shape errors propagate. Do not
    use Python object equality to compare these operands.

    .. code:: python

        # Source
        lhs <= rhs
        # Generated builder
        X.le_(lhs, rhs)
    """
    raise NotImplementedError


def gt_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Construct the > comparison in written operand order.

    Parameters
    ----------
    lhs : Expr or scalar
        Already-evaluated left operand.
    rhs : Expr or scalar
        Already-evaluated right operand.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Expr
        A native comparison expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx compares primitive operands;
    Relax selects tensor comparison if either operand is a nonprimitive IR
    expression, otherwise primitive comparison. Native dtype/shape errors propagate. Do not
    use Python object equality to compare these operands.

    .. code:: python

        # Source
        lhs > rhs
        # Generated builder
        X.gt_(lhs, rhs)
    """
    raise NotImplementedError


def ge_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Construct the >= comparison in written operand order.

    Parameters
    ----------
    lhs : Expr or scalar
        Already-evaluated left operand.
    rhs : Expr or scalar
        Already-evaluated right operand.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Expr
        A native comparison expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx compares primitive operands;
    Relax selects tensor comparison if either operand is a nonprimitive IR
    expression, otherwise primitive comparison. Native dtype/shape errors propagate. Do not
    use Python object equality to compare these operands.

    .. code:: python

        # Source
        lhs >= rhs
        # Generated builder
        X.ge_(lhs, rhs)
    """
    raise NotImplementedError


def eq_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Construct the == comparison in written operand order.

    Parameters
    ----------
    lhs : Expr or scalar
        Already-evaluated left operand.
    rhs : Expr or scalar
        Already-evaluated right operand.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Expr
        A native comparison expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx compares primitive operands;
    Relax selects tensor comparison if either operand is a nonprimitive IR
    expression, otherwise primitive comparison. Native dtype/shape errors propagate. Do not
    use Python object equality to compare these operands.

    .. code:: python

        # Source
        lhs == rhs
        # Generated builder
        X.eq_(lhs, rhs)
    """
    raise NotImplementedError


def ne_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Construct the != comparison in written operand order.

    Parameters
    ----------
    lhs : Expr or scalar
        Already-evaluated left operand.
    rhs : Expr or scalar
        Already-evaluated right operand.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Expr
        A native comparison expression.

    Notes
    -----
    No statement is emitted or frame entered. TIRx compares primitive operands;
    Relax selects tensor comparison if either operand is a nonprimitive IR
    expression, otherwise primitive comparison. Native dtype/shape errors propagate. Do not
    use Python object equality to compare these operands.

    .. code:: python

        # Source
        lhs != rhs
        # Generated builder
        X.ne_(lhs, rhs)
    """
    raise NotImplementedError


# --------------------------------------
# Section: context lookup and resolution
# --------------------------------------
#
# ``X.resolve_global_info_(key)`` resolves module metadata.
# ``X.resolve_type_var_("n")`` resolves a symbolic dimension.
# ``X.call_global_var_(f, args)`` calls a module function.
# ``I.module_member_("f", f)`` installs a module member.


def resolve_global_info_(content: Any) -> Any:
    """Look up language-owned global metadata selected by a source argument.

    Parameters
    ----------
    content : str or Any
        A selector or concrete value interpreted by the active language.

    Returns
    -------
    Any
        The language's metadata value. Selector syntax, lookup context and
        unsupported-value errors belong to the language variant implementation.

    Notes
    -----
    A builder may supply this hook to
    :func:`tvm.script.ir_builder.resolve_global_info_args`. The decorator resolves
    selected string arguments after ordinary Python argument evaluation, without
    attaching a new span to the existing metadata object. The parser does not rewrite
    selector arguments.

    .. code:: python

        @resolve_global_info_args("metadata", resolver=resolve_global_info_)
        def CustomType(metadata):
            return make_type(metadata)

        CustomType(metadata="mesh[0]")
    """
    raise NotImplementedError


def resolve_type_var_(
    name: str,
    dtype: str | _ir.Type | _ir.Var | None = None,
    *,
    value: _ir.Var | None = None,
    span: _Span = None,
) -> _ir.Var:
    """Resolve or declare a symbolic variable in the nearest native function.

    Parameters
    ----------
    name : str
        Function-local lookup key for explicit header parameters and captured symbols.
    dtype : str, Type or Var, optional
        Explicit primitive type or supplied variable. None defaults new symbols to
        int64; existing symbols must agree with an explicit dtype.
    value : Var, optional
        Existing primitive variable to register without replacement. None creates a
        variable only if the name is not already registered.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Var
        The exact existing or newly registered variable.

    Notes
    -----
    Requires an active native function frame. The signature and resumed body use the same
    symbol map; nested functions use separate maps. Conflicting dtype/variable declarations
    or missing context raise a builder error. No statement is emitted.

    .. code:: python

        # Source: def f[n: T.int32](...)
        # Generated builder
        n = X.resolve_type_var_("n", dtype="int32")
    """
    raise NotImplementedError


def call_global_var_(function: _ir.GlobalVar, args: Sequence[Any]) -> _ir.Expr:
    """Construct a call to a declared module function.

    Parameters
    ----------
    function : GlobalVar
        Native callee reference reserved by the module declaration phase.
    args : sequence of Any
        Positional operands evaluated once, in source order. Generated global calls do
        not accept keyword arguments.

    Returns
    -------
    _ir.Expr
        The caller language variant's call expression; constructing it does not emit a statement.

    Notes
    -----
    The callee declaration must be available in the active module context. TIRx preserves
    its exact declared return type; Relax converts operands using its expression conversion.
    Invalid types or an unavailable declaration produce native builder errors. Source-call
    handling attaches locations to the returned expression.

    .. code:: python

        # Source
        Module.callee(x)
        # Generated builder
        X.call_global_var_(callee_reference, [x])
    """
    raise NotImplementedError


def module_member_(name: str, value: Any) -> Any:
    """Register a concrete function member or retain ordinary class setup.

    Parameters
    ----------
    name : str
        Source class member identifier.
    value : Any
        Already-evaluated class member value; a BaseFunc is declared and defined as a
        module function.

    Returns
    -------
    Any
        The reserved GlobalVar for a concrete BaseFunc, otherwise the exact original
        value.

    Notes
    -----
    Concrete function registration requires an active module frame and follows native
    duplicate/type checks. Other values need no frame and cause no IR emission or renaming.
    Shared source class setup runs before function signatures so module attributes/global
    info are available.

    .. code:: python

        # Source
        class Module:
            helper = existing_function
        # Generated builder, inside I.ir_module
        helper = I.module_member_("helper", existing_function)
    """
    from tvm.ir import BaseFunc

    from .ir import decl_function, def_function

    if isinstance(value, BaseFunc):
        reference = decl_function(name, value)
        def_function(name, value)
        return reference
    return value


# --------------------------------------
# Section: special protocol
# --------------------------------------
#
# Syntax markers live in tvm.script.parser.protocol_registry.
# ``constexpr(value)`` selects host evaluation in marked control flow.
# ``register_scalar_annotation(path, constructor, dtype=...)`` describes scalar annotations.
# ``mutable_cell_decl(path)`` marks mutable storage declarations.
# ``result_span(path)`` permits attaching a call's result span without a call context.
# ``module_decorator(path)`` marks module declaration decorators.
# ``declaration_kind(path, kind)`` marks function and helper declarations.
# ``with X.function_(...) as fn:`` builds a function frame.
# ``a = X.arg("a", ty)`` declares a parameter.
# ``X.func_name("main")`` sets the function name.
# ``X.func_ret_type(ty)`` sets the result annotation.
# ``X.check_well_formed_(result)`` validates completed IR.


def function_(*, decl: bool = False, span: _Span = None, **options: Any) -> IRBuilderFrame:
    """Create the native function frame used for signature and body construction.

    Parameters
    ----------
    decl : bool, optional
        False (default) constructs a complete function on one entry. True collects a
        signature on the first entry and retains this frame for body re-entry.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.
    options : Any
        Language variant options: primitive function ``private`` and ``persistent``
        default to False. Relax ``pure`` defaults to True and ``private``/``local``
        to False; ``local=True`` requires a declared reference when building its
        body. See :func:`tvm.tirx.script.ir_builder.function_` and
        :func:`tvm.relax.script.ir_builder.function_` for their concrete options.

    Returns
    -------
    IRBuilderFrame
        The native context manager. It owns params, result type, symbol map, reference
        and completed result.

    Notes
    -----
    Requires an active IRBuilder; module definitions also require its module frame. Declare
    every sibling signature before any body for forward references. Re-enter the same frame,
    define and invoke a zero-argument lexical helper inside it, then exit before validation.
    Invalid options/context raise builder errors. The frame stores its source location
    independently of exit-time ambient context.

    .. code:: python

        # Source
        @T.prim_func
        def f(a: T.int32):
            T.evaluate(a)
        # Generated builder
        with X.function_() as fn:
            X.func_name("f")
            a = X.arg("a", X.int32)
            X.emit_(X.evaluate(a))
    """
    raise NotImplementedError


def arg(name: str, annotation: Any, *, span: _Span = None) -> _ir.Var:
    """Add a parameter to the active native function signature.

    Parameters
    ----------
    name : str
        Source parameter name.
    annotation : Type, Var, Buffer or callable
        Concrete rewritten annotation or existing native parameter. A callable
        annotation is evaluated; an existing variable retains identity.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    _ir.Var
        The created or retained parameter variable (buffers are native variables).

    Notes
    -----
    Requires an active function signature frame. Primitive symbols use that frame's
    resolver, preserving annotation/body identity. TIRx handles buffer layout according to
    its function policy; Relax converts its type annotation. Invalid annotations or context
    raise TypeError/native builder errors.

    .. code:: python

        # Source
        def f(a: T.int32):
            pass
        # Generated builder, inside the signature frame
        a = X.arg("a", X.int32)
    """
    raise NotImplementedError


def func_name(name: str) -> None:
    """Set the active native function's source name.

    Parameters
    ----------
    name : str
        Function identifier; used for the module reference and public symbol according
        to language variant privacy options.

    Returns
    -------
    None
        No value.

    Notes
    -----
    Requires a function frame. Mutates its signature metadata and emits no statement;
    invalid or repeated naming follows native diagnostics. No source span argument is needed
    because the frame owns its location.

    .. code:: python

        # Source
        def f():
            pass
        # Generated builder, inside the function frame
        X.func_name("f")
    """
    raise NotImplementedError


def func_ret_type(annotation: Any, *, span: _Span = None) -> None:
    """Set the active native function's return annotation.

    Parameters
    ----------
    annotation : Type, Expr or callable
        Rewritten return annotation; expression annotations supply their type. None
        denotes a void/empty tuple return as supported by the language variant.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    None
        No value.

    Notes
    -----
    Requires a function signature frame. Resolves callable annotations and records the type;
    no body is emitted. Unsupported or conflicting types raise native builder errors. The
    completed function retains its frame span.

    .. code:: python

        # Source
        def f() -> T.int32:
            return 1
        # Generated builder, inside the signature frame
        X.func_ret_type(X.int32)
    """
    raise NotImplementedError


def check_well_formed_(module: _ir.IRModule) -> None:
    """Validate a completed module after every body has been finalized.

    Parameters
    ----------
    module : IRModule
        Completed native module, including all resolved forward references and mixed-
        language variant members.

    Returns
    -------
    None
        No value or mutation; success means the existing native validators accepted the
        program.

    Notes
    -----
    Requires completed IR, with no active construction frame. Root coordination
    invokes opaque whole-module hooks supplied by language variant initialization through
    tvm.script.register_module_validator. Each hook owns concrete types, eligibility
    and cross-function validation, including captured/preexisting members. No source
    decorator inventory selects the hooks. Validator exceptions propagate unchanged;
    an empty hook list raises RuntimeError instead of establishing validity.
    check_well_formed=False omits the generated call entirely.

    .. code:: python

        # Source
        @I.ir_module
        class Module:
            pass
        # Generated builder, after module frame exit
        I.check_well_formed_(module)
    """
    from tvm.script import _MODULE_VALIDATORS

    validators = tuple(_MODULE_VALIDATORS)
    if not validators:
        raise RuntimeError("No completed-module validators are registered")
    for validator in validators:
        validator(module)


# --------------------------------------
# Section: binding
# --------------------------------------
#
# ``a = X.bind_(value, name="a")`` binds a source name.
# ``X.decl_mutable_cell_(value, ty=ty)`` declares mutable storage.
# ``X.set_mutable_cell_(cell, value)`` updates mutable storage.
# ``a, b = X.unpack(value)`` destructures a binding.


def bind_(
    value: Any = MISSING,
    *,
    ty: Any = None,
    name: str | None = None,
    span: _Span = None,
    value_span: _Span = None,
    name_span: _Span = None,
    frame_value: bool = False,
) -> Any:
    """Apply the language variant's ordinary assignment policy.

    Parameters
    ----------
    value : Any, optional
        Once-evaluated RHS. MISSING (the default) denotes an omitted initializer and is
        rejected unless the language variant supports the annotation-only form.
    ty : Type or annotation callable, optional
        Already-rewritten source annotation. None (the default) lets the language variant infer
        the binding type.
    name : str, optional
        Source target name. None (the default) requests no source-derived name.
    span : SpanEntry, Span or None, optional
        Binding-target location for a newly constructed binding. None (the default)
        leaves it unspecified; this is separate from the RHS location.
    value_span : SpanEntry, Span or None, optional
        RHS source location, passed separately without first stamping the returned value.
        None (the default) leaves explicit RHS attribution unspecified. The language variant
        applies it only when binding/conversion requires value attribution; TIRx
        variable and metadata passthrough retain producer names and spans.
    name_span : SpanEntry, Span or None, optional
        Location of the target identifier. None (the default) uses span; it can differ
        from the emitted statement location.
    frame_value : bool, optional
        False by default. True names an already-entered with-target without constructing
        another binding or entering its frame.

    Returns
    -------
    Any
        The value bound to the Python target; usually a native variable, or an unchanged
        host/frame-owned value.

    Notes
    -----
    Requires the language variant construction context when producing IR. TIRx emits immutable Bind
    statements; Relax emits normalized bindings and match-casts. Unsupported
    values/annotations raise TypeError or ValueError. Ordinary TIRx Vars (including
    BufferVars) pass through before general Expr binding, without naming, stamping
    or another binding. Other Expr values retain ordinary language variant binding. Non-Expr
    metadata such as Layout and ordinary meta_class instances passes through
    unchanged without inspecting or naming its resources. Explicit typed declarations,
    mutable storage and frame targets retain their separate contracts. AlreadyEmitted
    receipts retain RHS attribution without another emission. A DSL may opt into
    concise scope entry: register the returned child
    frame's exit callback on the active parent before child entry, then return its entered
    value. Later statements enter that child, and parent exit closes it. This is language variant
    policy; it adds no parser-owned scope state. Direct-call results and source module
    aliases bypass this operation.

    .. code:: python

        # Source
        x = value
        # Generated builder
        x = X.bind_(value, name="x", span=_S[0], value_span=_S[1])

        # TIRx concise scope entry
        tid = T.launch_thread("threadIdx.x", 128)
        # Generated builder
        tid = X.bind_(X.launch_thread("threadIdx.x", 128), name="tid")
    """
    raise NotImplementedError


def decl_mutable_cell_(
    value: Any = MISSING,
    *,
    ty: Any = None,
    name: str | None = None,
    span: _Span = None,
    name_span: _Span = None,
) -> Any:
    """Introduce explicitly declared mutable storage.

    Parameters
    ----------
    value : Any, optional
        Once-evaluated storage handle for a call declaration, or initializer for an
        annotation declaration. MISSING (the default) means no initializer.
    ty : Type or annotation callable, optional
        Scalar or vector storage annotation. None (the default) identifies an already-
        created storage handle.
    name : str, optional
        Source target name. None (the default) requests no source-derived name.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.
    name_span : SpanEntry, Span or None, optional
        Location of the target identifier. None (the default) uses span; it can differ
        from the emitted statement location.

    Returns
    -------
    Any
        The same declared storage handle, or the handle allocated for the annotation.

    Notes
    -----
    TIRx requires an active primitive function; scalar primitive annotations allocate local
    storage and may initialize it. Vector annotations allocate storage but reject an
    initializer. Invalid handle/type combinations raise TypeError or ValueError. Relax
    always rejects mutable storage with TypeError. Declaration syntax takes precedence over
    any outer mutable target name.

    .. code:: python

        # Source
        x: T.int32 = 1
        # Generated builder
        x = X.decl_mutable_cell_(1, ty=X.int32, name="x")
    """
    raise NotImplementedError


def set_mutable_cell_(target: Any, value: Any, *, span: _Span = None) -> AlreadyEmitted[Any]:
    """Emit an update through an existing mutable handle without rebinding it.

    Parameters
    ----------
    target : TensorLoad, scalar wrapper or one-element buffer Var
        Storage handle returned by an explicit mutable declaration. Its identity is
        retained; this argument is not a source name or a new declaration.
    value : Expr or scalar convertible to Expr
        Once-evaluated value to store. Native store checking validates its type and
        indices against the target.
    span : SpanEntry, Span or None, optional
        Location of the emitted store. None (the default) leaves explicit location
        unspecified; existing source-call provenance is retained.

    Returns
    -------
    AlreadyEmitted[Any]
        Receipt retaining the exact emitted statement; consuming it emits nothing again.

    Notes
    -----
    TIRx requires an active primitive function/statement region and appends a native store
    to that region. Scalar wrappers are unwrapped, TensorLoad indices are preserved, and
    one-element buffer targets use index zero. Unsupported targets raise TypeError; native
    type/index checks propagate. Relax always raises TypeError because its bindings are
    immutable. No new allocation or immutable binding is created.

    .. code:: python

        # Source
        x: T.int32 = 0
        x = value
        # Generated builder
        x = X.decl_mutable_cell_(0, ty=X.int32, name="x")
        X.set_mutable_cell_(x, value)
    """
    raise NotImplementedError


def unpack(value: Any) -> Any:
    """Expose elements for ordinary Python target unpacking.

    Parameters
    ----------
    value : Any
        Concrete IR tuple, typed tuple expression, or ordinary host iterable.

    Returns
    -------
    Any
        A Python tuple of IR fields/projections for an IR tuple; otherwise the original
        value.

    Notes
    -----
    No statement is emitted and no builder frame is entered. Known IR tuple types supply
    arity; ordinary Python performs target-count/starred-unpacking checks. Field identities
    are retained for concrete tuples. Direct-call results keep ordinary Python unpacking
    without this hook.

    .. code:: python

        # Source
        a, b = value
        # Generated builder
        left, right = X.unpack(value)
        a = X.bind_(left, name="a")
        b = X.bind_(right, name="b")
    """
    raise NotImplementedError


# --------------------------------------
# Section: statement
# --------------------------------------
#
# ``X.emit_(value)`` emits an expression statement.
# ``X.return_(value)`` emits a function return.
# ``X.setitem_(a, i, value)`` emits an indexed store.
# ``X.setattr_(a, "field", value)`` emits an attribute store.
# ``X.assert_(condition, message)`` emits an assertion.


def emit_(value: Any, *, span: _Span = None) -> None:
    """Consume a source expression statement.

    Parameters
    ----------
    value : Any
        Once-evaluated expression result. AlreadyEmitted receipts and None produce no
        additional emission.
    span : SpanEntry, Span or None, optional
        Location of the emitted statement and expression, or the existing node in an
        AlreadyEmitted receipt. None leaves explicit attribution unspecified. Active
        caller provenance is composed without adding a construction context; native
        frames retain the location for finalization.

    Returns
    -------
    None
        No source-visible value.

    Notes
    -----
    Requires an active language variant function/region when emitting IR. TIRx adds statements,
    evaluates expressions, and can enter concise frames. Variables, text, layouts,
    and meta_class instances are inert; sequences are consumed
    elementwise. Relax accepts only void expressions (or
    None/AlreadyEmitted); unsupported values raise TypeError and non-void expressions raise
    ValueError. An explicit span annotates the exact previously emitted node in a receipt
    without emitting it again. Known builder results need no separate result wrapper;
    opaque source calls keep their scoped provenance before this hook.

    .. code:: python

        # Source
        T.evaluate(1)
        # Generated builder
        X.emit_(X.evaluate(1), span=_S[0])
    """
    raise NotImplementedError


def return_(value: Any = None, *, span: _Span = None) -> AlreadyEmitted[Any] | None:
    """Record a language variant function return while continuing Python construction.

    Parameters
    ----------
    value : Any, optional
        Return operand. None (the default) means an empty tuple in Relax; TIRx requires
        an expression.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    AlreadyEmitted[Any] | None
        Receipt retaining the exact emitted statement; consuming it emits nothing again.
        Relax records a function result and returns None.

    Notes
    -----
    Requires an active function/region. TIRx emits its native return operation; Relax
    records the function result. Unsupported missing/type/context combinations raise
    TypeError or native errors. A source macro retaining ordinary Python return does not
    call this hook.

    .. code:: python

        # Source
        return value
        # Generated builder
        X.return_(value)
    """
    raise NotImplementedError


def setitem_(target: Any, key: Any, value: Any, *, span: _Span = None) -> AlreadyEmitted[Any]:
    """Apply an indexed assignment using already-evaluated operands.

    Parameters
    ----------
    target : buffer Var
        Destination buffer.
    key : Expr, int, slice or sequence
        Indices in written order; native buffer-store rules validate supported forms.
    value : Expr or scalar
        Once-evaluated stored value.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    AlreadyEmitted[Any]
        Receipt retaining the exact emitted statement; consuming it emits nothing again.

    Notes
    -----
    TIRx requires an active statement region and preserves target/index evaluation order.
    Native shape/type/index errors propagate. Relax raises TypeError because indexed
    mutation is unsupported.

    .. code:: python

        # Source
        A[i] = value
        # Generated builder
        X.setitem_(A, i, value)
    """
    raise NotImplementedError


def setattr_(
    target: Any, name: str, value: Any, *, span: _Span = None
) -> AlreadyEmitted[Any] | None:
    """Apply an attribute assignment using already-evaluated operands.

    Parameters
    ----------
    target : Any
        Object containing a scalar storage attribute or ordinary mutable Python
        metadata.
    name : str
        Attribute identifier, evaluated by source syntax before this hook.
    value : Any
        Once-evaluated replacement or stored value.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    AlreadyEmitted[Any] | None
        Receipt retaining the exact emitted statement; consuming it emits nothing again.
        Ordinary host attribute updates return None because they emit no IR statement.

    Notes
    -----
    TIRx stores through scalar storage attributes without replacing their handles; other
    attributes use ordinary Python setattr. Native store checks and Python attribute errors
    propagate. Relax raises TypeError. A native store needs an active function; ordinary
    host metadata updates do not.

    .. code:: python

        # Source
        state.count = value
        # Generated builder
        X.setattr_(state, "count", value)
    """
    raise NotImplementedError


def assert_(
    condition: Any,
    message: str | tuple[str, Sequence[Any]] | Sequence[Any] = "",
    *,
    span: _Span = None,
) -> None:
    """Emit a runtime assertion.

    Parameters
    ----------
    condition : Expr or bool
        Already-evaluated predicate.
    message : str or assertion metadata, optional
        Empty text by default. Relax requires construction-time text. TIRx also accepts
        message parts or an (error_kind, parts) pair.
    span : SpanEntry, Span or None, optional
        Location for the constructed result. None (the default) leaves explicit location
        unspecified; active source-call provenance is composed by the builder. Frames
        retain their location until finalization.

    Returns
    -------
    None
        No value; emits a native assertion.

    Notes
    -----
    Requires an active function/statement region. Malformed diagnostic metadata raises
    TypeError; native predicate checking propagates. Construction does not test an IR
    predicate as a host boolean.

    .. code:: python

        # Source
        assert condition, "failed"
        # Generated builder
        X.assert_(condition, "failed")
    """
    raise NotImplementedError
