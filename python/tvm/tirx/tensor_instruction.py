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
"""Named tensor instruction contracts over ordinary :class:`tvm.ir.Call`.

The Python views below are transient lowering helpers, not IR nodes. Every
expression, including optional workspace, remains in the Call's argument list.
"""

from dataclasses import dataclass, field
from inspect import Parameter, Signature
from types import SimpleNamespace
from typing import ClassVar

import tvm_ffi

from tvm.ir import (
    Attrs,
    Call,
    Expr,
    Op,
    PointerType,
    PrimType,
    StringImm,
    TensorRegion,
    Tuple,
    const,
    make_node,
)
from tvm.ir.location import UNKNOWN_LOC
from tvm.ir.op import register_op_attr
from tvm.tirx.exec_scope import ExecScope
from tvm.tirx.expr import IntImm

_REQUIRED = object()
SPECS = {}


@dataclass(frozen=True)
class Operand:
    name: str
    role: str = "expr"
    default: object = _REQUIRED


@dataclass
class Instruction:
    name: str
    kind: str
    operands: tuple[Operand, ...]
    schema: str = "ScopeAttrs"
    options: dict = field(default_factory=dict)
    workspaces: tuple[str, ...] = ()
    project: object = None
    lower: object = None

    @property
    def backend(self):
        return self.name.split(".")[1]

    def register(self):
        if self.name in SPECS:
            return SPECS[self.name]._factory
        register_op_attr(self.name, "TCallEffectKind", 3)
        register_op_attr(self.name, "TFixedReturnType", PrimType("void"))
        register_op_attr(
            self.name,
            "TIRxOpCategory",
            "tile_composite" if ".compose." in self.name else "tile_primitive",
        )
        register_op_attr(self.name, "TScriptPrinterName", self.name)
        op = Op.get(self.name)
        op.set_signature(args=[arg.name for arg in self.operands])
        op.set_attrs_type_key("tirx.tensor." + self.schema)
        tvm_ffi.get_global_func("tirx.ConfigureTensorInstruction")(op, self.validate)
        register_op_attr(
            self.name,
            "__tvm_doc_translate_op_call__",
            tvm_ffi.get_global_func("script.printer.TensorCallDocTranslate")(),
        )
        SPECS[self.name] = self

        def construct(*args, **qualifiers):
            return self.make(*args, **qualifiers)

        construct.__name__ = self.name.rsplit(".", 1)[-1]
        construct.__qualname__ = self.name.removeprefix("tirx.")
        construct.__doc__ = f"Construct the opaque void Call for {self.name}."
        parameters = [
            Parameter(
                arg.name,
                Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty
                if arg.default is _REQUIRED
                else None
                if arg.default == "@dst"
                else arg.default,
            )
            for arg in self.operands
        ]
        parameters += [
            Parameter(name, Parameter.KEYWORD_ONLY, default=default)
            for name, default in {"scope": "thread", **self.options}.items()
        ]
        construct.__signature__ = Signature(parameters, return_annotation=Call)
        self._factory = construct
        return construct

    def make(self, *values, **kw):
        if len(values) > len(self.operands):
            raise TypeError(f"{self.name} expects at most {len(self.operands)} operands")
        loc = kw.pop("loc", UNKNOWN_LOC)
        explicit_attrs = kw.pop("attrs", None)
        result_ty = kw.pop("ty", "void")
        if kw.pop("ty_args", ()):
            raise TypeError(f"{self.name} does not take type arguments")
        args = []
        for i, arg in enumerate(self.operands):
            if i < len(values):
                if arg.name in kw:
                    raise TypeError(f"{self.name}: duplicate operand {arg.name}")
                value = values[i]
            elif arg.name in kw:
                value = kw.pop(arg.name)
            elif arg.default is not _REQUIRED:
                value = arg.default
                if isinstance(value, str) and value == "@dst":
                    value = args[0]
            else:
                raise TypeError(f"{self.name}: missing operand {arg.name}")
            if value is None and isinstance(arg.default, str) and arg.default == "@dst":
                value = args[0]
            args.append(_operand(value, arg.role))
        allowed = {"scope": "thread", **self.options}
        unknown = set(kw) - set(allowed)
        if unknown:
            raise TypeError(f"{self.name}: unknown qualifier(s) {sorted(unknown)}")
        attrs_values = {name: _static(value) for name, value in kw.items()}
        if isinstance(attrs_values.get("axes"), int):
            attrs_values["axes"] = [attrs_values["axes"]]
        if self.schema == "TcMmaAttrs":
            for name in ("mma_m", "mma_n"):
                value = attrs_values.get(name)
                if value is not None and (type(value) is not int or value <= 0):
                    raise ValueError(f"{name} must be a positive integer")
        if self.schema == "TMAAttrs":
            if isinstance(attrs_values.get("oob"), str):
                oob = attrs_values["oob"]
                if oob not in {"zero", "nan"}:
                    raise ValueError(f"unsupported TensorMap oob={oob!r}")
                attrs_values["oob"] = {"zero": 0, "nan": 1}[oob]
            if isinstance(attrs_values.get("tensormap_l2_promotion"), str):
                promotion = attrs_values["tensormap_l2_promotion"]
                choices = {"none": 0, "L2::none": 0, "L2::64B": 1, "L2::128B": 2, "L2::256B": 3}
                if promotion not in choices:
                    raise ValueError(f"invalid TensorMap L2 promotion {promotion!r}")
                attrs_values["tensormap_l2_promotion"] = choices[promotion]
        for name, default in allowed.items():
            attrs_values.setdefault(name, default)
        if explicit_attrs is not None:
            if kw:
                raise TypeError("attrs cannot be combined with qualifier keywords")
            attrs = explicit_attrs
        else:
            attrs = make_node("tirx.tensor." + self.schema, **attrs_values)
        call = Call(self.name, args, attrs=attrs, ty=result_ty, loc=loc)
        call.validate()
        return call

    def validate(self, call):
        if len(call.args) != len(self.operands):
            raise TypeError(f"{self.name}: invalid operand count")
        if str(call.attrs.scope) not in {"thread", "warp", "warpgroup", "cta", "cluster"}:
            raise ValueError(f"{self.name}: invalid scope {call.attrs.scope!r}")
        for arg, value in zip(self.operands, call.args):
            if _is_absent(value):
                if arg.default is not None:
                    raise TypeError(f"{self.name}: {arg.name} is required")
                continue
            if arg.role == "region":
                _region(value)
            elif arg.role == "tensor":
                from tvm.tirx import is_tensor_var

                if not is_tensor_var(value):
                    raise TypeError(f"{arg.name} must be a tensor variable")
            elif arg.role == "address":
                from tvm.ir import is_prim_expr

                if not isinstance(value.ty, PointerType) and not is_prim_expr(value):
                    raise TypeError(f"{arg.name} must be a pointer or shared address")
            elif arg.role == "scalar":
                from tvm.ir import is_prim_expr

                if not is_prim_expr(value):
                    raise TypeError(f"{arg.name} must be a scalar expression")
            elif arg.role == "coordinates":
                if not isinstance(value, Tuple) or len(value.fields) != 4:
                    raise ValueError("gather4 must contain exactly four row coordinates")
            elif arg.role == "selectors":
                if not isinstance(value, Tuple):
                    raise TypeError("src_selector must contain (condition, region) pairs")
                for pair in value.fields:
                    if not isinstance(pair, Tuple) or len(pair.fields) != 2:
                        raise TypeError("src_selector must contain (condition, region) pairs")
                    candidate = _region(pair.fields[1])
                    from tvm.sym.analyzer import Analyzer

                    analyzer = Analyzer()
                    if not all(
                        analyzer.can_prove_equal(r.min, 0)
                        and analyzer.can_prove_equal(r.extent, dim)
                        for r, dim in zip(candidate.region, candidate.source.ty.shape)
                    ):
                        raise ValueError(
                            "src_selector candidates must cover their full tensor view"
                        )

        # Validate nested regions too, including selector candidates.
        def check_region(region):
            _region(region)

        tvm_ffi.structural_walk(call.args, (TensorRegion, check_region))


def _expr(value):
    if isinstance(value, Expr):
        return value
    if isinstance(value, list | tuple | tvm_ffi.Array):
        return Tuple([_expr(v) for v in value])
    if isinstance(value, str):
        return StringImm(value)
    if hasattr(value, "asobject"):
        return value.asobject()
    return const(value)


def _static(value):
    if isinstance(value, StringImm):
        return value.value
    if isinstance(value, IntImm):
        return bool(value) if str(value.dtype) == "bool" else int(value)
    if isinstance(value, ExecScope):
        return value.name
    if isinstance(value, Expr):
        raise TypeError("Instruction qualifiers must be static; expressions belong in operands")
    return value


def _region(value):
    from tvm.tirx import is_tensor_var

    if is_tensor_var(value):
        return value[tuple(slice(None) for _ in value.ty.shape)]
    if not isinstance(value, TensorRegion) or not is_tensor_var(value.source):
        raise TypeError("Tensor operands require a TensorRegion with a TensorVar source")
    if len(value.region) != len(value.source.ty.shape):
        raise ValueError("TensorRegion rank must match its tensor rank")
    return value


def _is_absent(value):
    return isinstance(value, Tuple) and not value.fields


def _operand(value, role="expr"):
    from tvm.tirx import is_tensor_var

    if value is None:
        return Tuple([])
    if role == "region":
        return _region(value)
    if role == "selectors":
        return Tuple([Tuple([_expr(cond), _region(candidate)]) for cond, candidate in value])
    if role == "tensor":
        return value
    if is_tensor_var(value):
        return _region(value)
    if isinstance(value, list | tuple):
        return Tuple([_operand(v) for v in value])
    return _expr(value)


def namespace(backend, instructions):
    result = SimpleNamespace()
    for instruction in instructions:
        path = instruction.name.removeprefix(f"tirx.{backend}.tile.").split(".")
        container = result
        for part in path[:-1]:
            if not hasattr(container, part):
                setattr(container, part, SimpleNamespace())
            container = getattr(container, part)
        fn = instruction.register()
        setattr(container, path[-1], fn)
    return result


class TensorCall:
    """Read-only semantic view used by tensor instruction lowering code."""

    _registry: ClassVar[dict] = {}

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if "kind" in cls.__dict__:
            cls._registry[cls.kind] = cls

    @classmethod
    def decode(cls, call):
        if isinstance(call, TensorCall):
            return call
        spec = SPECS[call.op.name]
        values = dict(zip((a.name for a in spec.operands), call.args))
        values = {k: None if _is_absent(v) else v for k, v in values.items()}
        options = {k: getattr(call.attrs, k) for k in spec.options}
        if spec.project:
            kind, args, extras = spec.project(values, call.attrs)
        else:
            kind = spec.kind
            args = [values[a.name] for a in spec.operands if a.name not in spec.workspaces]
            extras = {}
        from tvm.tirx.op import tile as _views  # noqa: F401 - register semantic views

        view = cls._registry.get(kind, cls).__new__(cls._registry.get(kind, cls))
        view.call, view.spec, view.op, view.kind = call, spec, call.op, kind
        view.args = args
        # Temporary scheduler options are decoded from individually named
        # operands and typed static fields; they are never serialized as a bag.
        view.options = {k: _expr(v) for k, v in options.items() if v is not None}
        view.options.update({k: v for k, v in extras.items() if v is not None})
        view.workspaces = {k: values[k] for k in spec.workspaces if values[k] is not None}
        view.scope = ExecScope(str(call.attrs.scope))
        return view

    def get_private_buffers(self, buffer_dict, sctx):
        if sctx.target.kind.name == "trn":
            return self.get_private_buffers_trn(buffer_dict, sctx)
        return {}

    def get_private_buffers_trn(self, buffer_dict, sctx):
        return {}

    def with_workspaces(self, values):
        args = list(self.call.args)
        for name, value in values.items():
            index = next(i for i, arg in enumerate(self.spec.operands) if arg.name == name)
            args[index] = value
        return Call(self.op, args, attrs=self.call.attrs, ty=self.call.ty, loc=self.call.loc)


@tvm_ffi.register_global_func("tirx.TensorCallScope")
def _scope(call):
    return ExecScope(str(call.attrs.scope))


for _name in (
    "ScopeAttrs",
    "MemoryAttrs",
    "AsyncAttrs",
    "TMAAttrs",
    "TcCopyAttrs",
    "TcMmaAttrs",
    "MathAttrs",
    "TrnAttrs",
):
    tvm_ffi.register_object("tirx.tensor." + _name)(type(_name, (Attrs,), {}))
