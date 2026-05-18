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
# pylint: disable=invalid-name
"""Thin device-helper registration for TIRx intrinsic codegens.

Exposes one entry point: :func:`device_intrinsic`. Given an op name plus
``(helper_name, c_signature, body)`` (each a string or a ``(*args) -> str``
callable), it:

* wraps the body in
  ``__forceinline__ __device__ <ret> <name><sig> { <body> }``,
* registers a codegen function under the op name so
  ``call_intrin("", "tirx.<op_name>", *args)`` resolves to a call to that
  helper, and
* registers the op with TVM's Op registry (``TCallEffectKind=Opaque``) so
  it doesn't need a C++ ``TIR_DEFINE_BUILTIN_FUNC`` entry.

Args passed to the codegen are split into ``(forward_args, attr_args)``:
the trailing ``n_attrs`` are attrs (consumed by the ``helper_name`` /
``c_signature`` / ``body`` callables but **not** forwarded to the helper),
and the rest are operand args (forwarded). The default ``n_attrs=0`` means
every arg is forwarded — appropriate for fixed-arity ops with literal
``c_signature`` and ``helper_name``.

Coerce / validate attrs explicitly inside the callables — there is no
``Choice`` / ``Bool`` / ``IntAttr`` machinery; just call ``parse_str`` /
``int`` / ``bool`` on the raw arg as needed.
"""

from __future__ import annotations

from collections.abc import Callable

from tvm.tirx.op import cuda_func_call
from tvm.tirx.operator.intrinsics.cuda.registry import register_codegen

# C primitive type → TVM dtype string. Used when the caller specifies a
# non-void ``return_type`` but no explicit ``tvm_return_type`` — the helper
# knows the TVM-side dtype from the C return type.
_C_TO_TVM_DTYPE = {
    "float": "float32",
    "double": "float64",
    "uint32_t": "uint32",
    "int32_t": "int32",
    "uint64_t": "uint64",
    "int64_t": "int64",
    "uint16_t": "uint16",
    "int16_t": "int16",
    "unsigned long long": "uint64",
    "long long": "int64",
    "unsigned short": "uint16",
    "bool": "bool",
    "unsigned int": "uint32",
    "int": "int32",
}


def device_intrinsic(
    op_name: str,
    *,
    helper_name: str | Callable | None = None,
    c_signature: str | Callable = "()",
    body: str | Callable,
    n_attrs: int = 0,
    return_type: str | Callable = "void",
    tvm_return_type: str | Callable | None = None,
    templated: bool = False,
    extra_deps: tuple = (),
) -> None:
    """Register a CUDA device-helper intrinsic.

    Parameters
    ----------
    op_name :
        Registry key — ``call_intrin("", "tirx.<op_name>", ...)`` resolves
        here. Also used as the default helper name (``tvm_builtin_<op_name>``)
        when ``helper_name`` is not provided.
    helper_name :
        Literal C function name, OR ``(*args) -> str`` to compute it from
        attr values. Defaults to ``f"tvm_builtin_{op_name}"``.
    c_signature :
        Literal C parameter list including outer parens (``"(int x, int y)"``),
        OR ``(*args) -> str`` to compute it from attr values. Defaults to
        ``"()"``.
    body :
        Literal C body string (already indented), OR ``(*args) -> str``.
    n_attrs :
        Number of trailing args that are attrs (consumed by ``helper_name``
        / ``c_signature`` / ``body`` callables, NOT forwarded to the helper
        as call arguments). The first ``len(args) - n_attrs`` args are the
        operand args forwarded to the helper.
    return_type :
        C return type. Default ``"void"``. Either a literal string or
        ``(*args) -> str`` when the helper return type depends on attrs.
    tvm_return_type :
        TVM dtype for the call result, when the helper has a non-void
        return. Either a literal string (``"int32"``) or ``(*args) -> str``.
        If omitted and ``return_type`` is non-void, it is auto-derived from
        the ``_C_TO_TVM_DTYPE`` table.
    templated :
        Prefix the helper with ``template <typename T>``.
    extra_deps :
        Helper-tag list (e.g. ``("get_tmem_addr",)``) forwarded as the second
        element of the codegen result so the header generator emits the
        prerequisite snippets.
    """
    if helper_name is None:
        helper_name = f"tvm_builtin_{op_name}"
    extra_deps = tuple(extra_deps)

    def codegen(*args):
        forward = args if n_attrs == 0 else args[:-n_attrs]
        name = helper_name(*args) if callable(helper_name) else helper_name
        sig = c_signature(*args) if callable(c_signature) else c_signature
        body_str = body(*args) if callable(body) else body
        ret_type = return_type(*args) if callable(return_type) else return_type
        prefix = "template <typename T>\n" if templated else ""
        source_code = (
            f"\n{prefix}__forceinline__ __device__ {ret_type} {name}{sig} {{\n{body_str}\n}}\n"
        )
        kwargs = {"source_code": source_code}
        if tvm_return_type is not None:
            kwargs["return_type"] = (
                tvm_return_type(*args) if callable(tvm_return_type) else tvm_return_type
            )
        elif ret_type != "void":
            kwargs["return_type"] = _C_TO_TVM_DTYPE.get(ret_type, ret_type)
        result = cuda_func_call(name, *forward, **kwargs)
        return (result, list(extra_deps)) if extra_deps else result

    codegen.__name__ = f"codegen_{op_name}"
    register_codegen(op_name)(codegen)
    _ensure_op_registered(f"tirx.{op_name}")


# ---------------------------------------------------------------------------
# Dynamic Op registration — ensures op_name has a TVM Op (with default
# TCallEffectKind=Opaque) so call_intrin can resolve it without requiring a
# C++ TIR_DEFINE_BUILTIN_FUNC entry.
# ---------------------------------------------------------------------------

import tvm_ffi  # noqa: E402

_ir_register_op = tvm_ffi.get_global_func("ir.RegisterOp")
_ir_register_op_attr = tvm_ffi.get_global_func("ir.RegisterOpAttr")
# CallEffectKind enum (include/tvm/tir/op_attr_types.h): Opaque = 4.
_CALL_EFFECT_KIND_OPAQUE = 4
_registered_attrs: set = set()


def _ensure_op_registered(op_name: str) -> None:
    """Register ``op_name`` if not already in TVM's Op registry, plus a
    default ``TCallEffectKind=Opaque`` attribute. Both calls are no-ops when
    the op / attribute is already registered (the C++-side registrations win
    by plevel)."""
    try:
        _ir_register_op(op_name, "")
    except Exception:
        pass
    if op_name in _registered_attrs:
        return
    try:
        _ir_register_op_attr(op_name, "TCallEffectKind", _CALL_EFFECT_KIND_OPAQUE, 10)
        _registered_attrs.add(op_name)
    except Exception:
        pass
