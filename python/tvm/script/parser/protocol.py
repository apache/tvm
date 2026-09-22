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
"""Parser-owned syntax policies and function-decorator registration.

This module stores only host callable identities and flat syntax metadata.
Concrete eager annotation behavior is delegated to a builder-owned adapter;
no IR definition, concrete annotation result, symbol or frame is stored here.
"""

from collections.abc import Mapping
from inspect import signature
from types import MappingProxyType
from typing import Any, NamedTuple


def constexpr(value):
    """Mark a host control value in source syntax; consumed by the transpiler.

    The same callable may annotate a JIT specialization parameter. Generated
    programs never call this marker: they evaluate its operand as Python.
    """
    raise TypeError("constexpr is a parser syntax marker, not a runtime operation")


def is_constexpr_marker(value):
    """Recognize the shared marker by identity without inspecting host values."""
    return value is constexpr


class ExprStrPolicy(NamedTuple):
    """Immutable syntax policy for registered constructor arguments.

    Parameters
    ----------
    fields : tuple of str
        Parameter names whose string values represent expressions.
    introduce : bool, optional
        Legacy symbol-introduction metadata. Default is False.
    dtype : object, optional
        Dtype spelling passed to builder symbol resolution. Default is None.
    scalar_strings : bool, optional
        Interpret bare strings as expressions. Default is True. Nested
        strings in marked fields are always treated as expressions.
    compound_declarations : bool, optional
        Legacy compound-expression metadata. Default is False.

    Notes
    -----
    This tuple record lives with its registered callable across transpilation
    passes. It stores no symbols, eager results, or function-local state.
    Dtype spellings are forwarded as builder arguments; the transpiler never
    constructs or validates symbols. Builders retain already-declared symbol
    identity and type. Construction enters no frame.
    """

    fields: tuple[str, ...]
    introduce: bool = False
    dtype: Any = None
    scalar_strings: bool = True
    compound_declarations: bool = False


# Process-wide registry: callable identity -> immutable syntax policy. Dialect
# imports register once; aliases share identities. No per-function entries or
# evaluation results are cached, and transpilers only read this table.
_ARGS_POLICIES = {}


class ArgsPolicy(NamedTuple):
    """Immutable per-parameter syntax policies and eager expression metadata."""

    fields: Mapping[str, str]
    expression: ExprStrPolicy


def get_args_policy(constructor):
    """Return registered argument policies without evaluating the constructor.

    Legacy attached expression metadata is read without changing its owner.
    Unregistered and unhashable host values have no argument policy.
    """
    try:
        policy = _ARGS_POLICIES.get(constructor)
    except TypeError:
        return None
    if policy is not None:
        return policy
    expression = getattr(constructor, "__tvm_expression_args__", None)
    if expression is None:
        return None
    return ArgsPolicy(MappingProxyType(dict.fromkeys(expression.fields, "expr_str")), expression)


def args_policy(fields, *, scalar_strings=True, as_type=False):
    """Register ``expr_str`` and ``global_info`` policies by parameter name.

    Expression strings are rewritten as expressions; global-info arguments
    become shared builder lookup calls with their original content. Unmarked
    parameters, including dtype and placement, keep literal strings. Registered
    mappings are copied and immutable. Both positional and keyword arguments
    use the same parameter policy.

    ``scalar_strings=False`` preserves a bare string in an expression field
    (such as ``Tensor("float32")``), while nested shape strings are expressions.
    Builders own the eager expression adapter: unresolved annotations outside
    builder scope return MissingType, and concrete arguments construct normally.
    ``as_type=True`` preserves the Python annotation-class surface.

    Examples
    --------
    Register shape expressions and module-owned device references::

        @args_policy({"shape": "expr_str", "vdevice": "global_info"},
                     scalar_strings=False)
        def tensor(shape=None, dtype=None, vdevice=None):
            ...
    """
    fields = dict(fields)
    unsupported = set(fields.values()).difference(("expr_str", "global_info"))
    if unsupported:
        raise ValueError(f"Unknown argument policies: {sorted(unsupported)}")

    def decorate(constructor):
        unknown = set(fields).difference(signature(constructor).parameters)
        if unknown:
            raise ValueError(f"Unknown argument policy fields: {sorted(unknown)}")
        expression_fields = tuple(name for name, kind in fields.items() if kind == "expr_str")
        if expression_fields or as_type:
            result = expr_str_args(
                *expression_fields, scalar_strings=scalar_strings, as_type=as_type
            )(constructor)
            expression = expr_str_policy(result)
        else:
            result = constructor
            expression = ExprStrPolicy((), scalar_strings=scalar_strings)
        policy = ArgsPolicy(MappingProxyType(fields.copy()), expression)
        _ARGS_POLICIES[constructor] = policy
        _ARGS_POLICIES[result] = policy
        return result

    return decorate


def expr_str_policy(constructor):
    """Look up the expression-string policy for a host value.

    Parameters
    ----------
    constructor : object
        Python value whose registration or attached policy is queried.

    Returns
    -------
    ExprStrPolicy or None
        Registered or attached policy, or None if unregistered or unhashable.
        Legacy attached policies may use the equivalent legacy record type.

    Notes
    -----
    This read-only lookup neither imports dialects nor calls the constructor.
    Returned policies are shared registration state; no builder frame is
    entered. Exceptions from custom attribute access propagate.
    """
    policy = get_args_policy(constructor)
    return policy.expression if policy is not None else None


def expr_str_args(
    *fields,
    introduce=False,
    dtype=None,
    scalar_strings=True,
    compound_declarations=False,
    as_type=False,
):
    """Register expression-string fields and wrap eager constructor calls.

    Parameters
    ----------
    *fields : str
        Parameter names whose string values denote source expressions.
    introduce : bool, optional
        Legacy syntax metadata. Default is False; builders own introduction.
    dtype : object, optional
        Dtype spelling passed to builder symbol resolution. Default is None.
    scalar_strings : bool, optional
        Interpret bare strings as expressions. Default is True. False
        preserves literal shorthand such as ``Tensor("float32")``; nested
        strings in tuples and lists remain expressions.
    compound_declarations : bool, optional
        Legacy compound-expression metadata. Default is False.
    as_type : bool, optional
        Preserve use as an annotation class, including Python type unions.
        Default is False. True returns a class invoking the wrapped callable;
        construction still returns that callable's result.

    Returns
    -------
    decorator : callable
        Registers the constructor's policy and returns its eager-call adapter.

    Raises
    ------
    ValueError
        When applying the decorator to a constructor whose signature cannot
        be inspected or does not contain a requested field.
    TypeError
        When the constructor cannot be inspected, or a wrapped call cannot
        bind its signature or has unresolved fields inside an active builder.

    Notes
    -----
    Both original and wrapped callable identities retain the policy in the
    process-wide registry. Builders own eager annotation behavior: unresolved
    strings or typing.TypeVar values in marked fields produce MissingType
    outside active builders. Concrete arguments call the original constructor.
    Handwritten builders require concrete expressions. Calls are never cached;
    annotations must be safe to re-evaluate.

    Examples
    --------
    Register a shape parameter while leaving the dtype string literal intact::

        @expr_str_args("shape", scalar_strings=False)
        def tensor_type(shape, dtype="float32"):
            return make_tensor_type(shape, dtype)
    """

    def decorate(constructor):
        call_signature = signature(constructor)
        unknown = set(fields).difference(call_signature.parameters)
        if unknown:
            raise ValueError(f"Unknown expression argument fields: {sorted(unknown)}")
        policy = ExprStrPolicy(
            tuple(fields),
            bool(introduce),
            dtype,
            bool(scalar_strings),
            bool(compound_declarations),
        )

        # Registration is syntax-only. Builders own eager construction and the
        # active-frame/MissingType decisions behind this generic wrapper factory.
        from tvm.script.ir_builder.type_var_frame import wrap_expression_constructor

        result = wrap_expression_constructor(constructor, call_signature, policy, as_type=as_type)
        result.__tvm_expression_args__ = policy
        arguments = ArgsPolicy(MappingProxyType(dict.fromkeys(fields, "expr_str")), policy)
        _ARGS_POLICIES[result] = arguments
        _ARGS_POLICIES[constructor] = arguments
        return result

    return decorate


class DeclarationArguments(NamedTuple):
    """Immutable metadata for scalar-constructor predeclarations.

    Parameters
    ----------
    value_parameter : str
        Optional value parameter whose absence denotes a declaration.
    dtype : object, optional
        Explicit primitive dtype metadata. Default is None.

    Notes
    -----
    `register_type_var_decl` attaches this record to its callable. Aliases
    and transpilation passes share it for the callable's lifetime; it stores
    no symbols or construction results and enters no builder frame.
    """

    value_parameter: str
    dtype: Any = None


def register_type_var_decl(constructor, *, value_parameter="expr", dtype=None):
    """Register a constructor that can declare a type variable.

    Parameters
    ----------
    constructor : callable
        Scalar constructor supporting attribute assignment.
    value_parameter : str, optional
        Optional value parameter. Default is ``"expr"``; omission of this
        parameter identifies declaration syntax.
    dtype : object, optional
        Explicit primitive dtype metadata. Default is None. A string dtype
        also permits predeclaration of direct zero-argument body calls.

    Returns
    -------
    callable
        The original constructor with attached `DeclarationArguments`.

    Raises
    ------
    AttributeError
        If the constructor cannot store the registration attribute.

    Notes
    -----
    The signature prepass reserves explicitly declared parameter types before
    resolving earlier shape strings. The shared name prescan also recognizes
    direct unconditional zero-argument body calls when dtype is a string,
    emitting the spelling as builder predeclaration data. It excludes nested
    control scopes and argument-bearing or effectful expressions. Original
    calls and bindings retain their body order.

    Registration persists with the callable. The constructor is neither
    wrapped nor evaluated, and no builder frame is entered.
    """
    constructor.__tvm_type_var_decl__ = DeclarationArguments(value_parameter, dtype)
    return constructor


class FunctionDecoratorInfo(NamedTuple):
    """Flat syntax registration for a source function decorator.

    Parameters
    ----------
    builder : object or None
        Opaque construction namespace, or None for ordinary Python functions.
    option_map : dict of str to str, optional
        Public option names mapped to builder option names. Default is None,
        interpreted as empty. `register_function` copies supplied mappings.
    defaults : dict of str to object, optional
        Default builder keyword values. Default is None, interpreted as
        empty. `register_function` copies supplied mappings.
    python : bool, optional
        Preserve the source body as ordinary Python. Default is False.

    Notes
    -----
    These are the complete supported fields; no generic metadata dictionary
    is retained. Records live with decorators across compilations. Consumers
    must treat mappings as read-only. Records retain no frame, function
    result, annotation result, or symbol state and enter no builder frames.
    """

    builder: Any
    option_map: dict | None = None
    defaults: dict | None = None
    python: bool = False


def register_function(decorator, builder, *, option_map=None, defaults=None, python=False):
    """Register a decorator with explicit supported syntax options.

    Parameters
    ----------
    decorator : callable
        Function decorator supporting attribute assignment.
    builder : object or None
        Opaque construction namespace, or None when ``python=True``.
    option_map : mapping of str to str, optional
        Public option names mapped to builder keyword names. Default is None,
        interpreted as an empty mapping. Entries are copied.
    defaults : mapping of str to object, optional
        Default builder keyword values. Default is None, interpreted as an
        empty mapping. Entries are copied.
    python : bool, optional
        Preserve the source body as ordinary Python. Default is False.

    Returns
    -------
    callable
        The original decorator with attached `FunctionDecoratorInfo`.

    Raises
    ------
    AttributeError
        If the decorator cannot store the registration attribute.
    TypeError or ValueError
        If supplied option mappings cannot be converted to dictionaries.

    Notes
    -----
    Registration replaces ``__tvm_function_info__`` for the callable's
    lifetime. It executes no constructor or annotation and enters no frame.
    Unknown keyword arguments are rejected by the explicit signature.
    """
    decorator.__tvm_function_info__ = FunctionDecoratorInfo(
        builder, dict(option_map or {}), dict(defaults or {}), bool(python)
    )
    return decorator


def function_info(decorator):
    """Read registered construction metadata without calling the decorator.

    The returned option mappings are shared registration state. This lookup
    enters no builder frame and propagates custom attribute-access errors.
    """
    return getattr(decorator, "__tvm_function_info__", None)
