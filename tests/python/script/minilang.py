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
"""A recording mini-language for executing production-generated builder programs.

Values record which protocol hook the generated program called and its operands.
Their addition records only operand order; no type inference, folding or runtime
evaluation is implemented. Tests of expression semantics and span composition use
real common IR/Prim nodes. Frames record entry, parameters, outputs and identity;
they deliberately implement no TVM type system or parser.
Each test creates a fresh language so recorded effects cannot leak across cases.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace

from tvm.script.ir_builder import IRBuilder, resolve_global_info_args
from tvm.script.ir_builder.base import MISSING, AlreadyEmitted
from tvm.script.ir_builder.ir import constexpr
from tvm.script.parser import entry, protocol_registry


@dataclass(eq=False)
class Value:
    """An opaque protocol-call record, with operands and instrumentation locations."""

    op: str
    args: tuple = ()
    name: str = ""
    span: object = None

    def __add__(self, other):
        return Value("add", (self, other))

    def __radd__(self, other):
        return Value("add", (other, self))

    def __bool__(self):
        raise TypeError("an IR expression cannot select Python control flow")


@dataclass(eq=False)
class Function:
    name: str = ""
    params: list = field(default_factory=list)
    body: list = field(default_factory=list)
    ret_type: object = None


class Module(dict):
    """Named dummy module; normal dict lookup and function object identity."""


class Frame:
    """Native-style frame: declarations and body reuse the same params/result."""

    def __init__(self, language, kind, *, decl=False, values=(), span=None, local=False, **options):
        self.language, self.kind, self.decl = language, kind, decl
        self.values = values
        self.vars = (
            [Value("loop", (bound,), f"i{index}") for index, bound in enumerate(values)]
            if kind == "for"
            else []
        )
        self.function = Function() if kind == "function" else None
        self.params = []
        self.type_var_map = {}
        self.global_var = Value("global", (self,))
        self.local_var = Value("local", (self,))
        self.result = None
        self.branches = []
        self.names = None
        self.span = None

    def __enter__(self):
        self.language.stack.append(self)
        if self.kind == "for":
            return self.vars[0] if len(self.vars) == 1 else self.vars
        return self

    def __exit__(self, error_type, error, traceback):
        assert self.language.stack.pop() is self
        if self.kind == "function":
            if self.decl:
                self.decl = False
            else:
                self.language.functions[self.function.name] = self.function
                self.language.result = self.function
        elif self.kind in ("then", "else"):
            self.language.stack[-1].branches.append(self.result)
        elif self.kind == "if":
            self.var = Value("if", (*self.values, *self.branches))
        elif self.kind == "module":
            self.language.result = Module(self.language.functions)
        return False

    def resolve_type_var(self, name, dtype=None, *, value=None, **kwargs):
        if name not in self.type_var_map:
            self.type_var_map[name] = (
                value
                if value is not None
                else Value("symbol", ("int64" if dtype is None else dtype,), name)
            )
        return self.type_var_map[name]

    def __getitem__(self, name):
        return self.language.functions[name]

    def __getattr__(self, name):
        if self.kind == "module" and name in self.language.references:
            return self.language.references[name]
        raise AttributeError(name)


class RecordingSpanEntry:
    """Use the recording hooks for opaque dummy values, with a real fixed span."""

    def __init__(self, language, span):
        self.language, self.span = language, span

    @property
    def location(self):
        span = self.span
        return (span.source_name, span.line, span.end_line, span.column, span.end_column)

    def __call__(self, value):
        return self.language.I.at_(self.location, value)

    def ctx(self, thunk, *, attach_result=True):
        return self.language.I.with_at_group_(self.location, thunk, attach_result=attach_result)


class Language:
    """One recording source namespace M and shared infrastructure namespace I."""

    def __init__(self):
        self.stack, self.functions = [], Module()
        self.references = {}
        self.missing = MISSING
        self.result = None
        self.source_stack = []
        self.global_infos = {}
        self.I = SimpleNamespace(
            IRBuilder=self.context,
            ir_module=lambda: Frame(self, "module"),
            at_=self.at,
            with_at_group_=self.with_at_group,
            module_member_=lambda name, value: value,
            MISSING=self.missing,
            check_well_formed_=lambda result: None,
            constexpr=constexpr,
        )
        self.M = SimpleNamespace(
            supports_mutable_declarations=True,
            function_=lambda **kwargs: Frame(self, "function", **kwargs),
            func_name_=self.func_name_,
            arg_=self.arg_,
            func_ret_type_=self.func_ret_type_,
            func_ret_value=self.func_ret_value,
            return_=lambda value=None, **kwargs: self.func_ret_value(value),
            setitem_=self.setitem_,
            setattr_=self.setattr_,
            unpack_=lambda value: value,
            assert_=lambda condition, message="", **kwargs: self.statement(
                "assert", condition, message
            ),
            break_=lambda **kwargs: self.statement("break"),
            continue_=lambda **kwargs: self.statement("continue"),
            resolve_type_var_=self.resolve_type_var,
            resolve_global_info_=self.resolve_global_info,
            bind_=self.bind,
            check_well_formed_=lambda result: None,
            emit_=self.emit,
            decl_mutable_cell_=self.decl_mutable,
            set_mutable_cell_=self.set_mutable,
            call_global_var_=lambda function, args: Value("call", (function, *args)),
            range_=lambda *bounds, **kwargs: Frame(self, "for", values=(bounds,)),
            grid=lambda *bounds: Frame(self, "for", values=bounds),
            for_=self.for_frame,
            If=lambda condition, **kwargs: Frame(self, "if", values=(condition,)),
            if_=lambda condition, **kwargs: Frame(self, "if", values=(condition,)),
            then_=lambda **kwargs: Frame(self, "then"),
            else_=lambda **kwargs: Frame(self, "else"),
            while_=lambda condition, **kwargs: Frame(self, "while", values=(condition,)),
            if_then_else_=lambda *args: Value("select", args),
            and_=lambda *args, **kwargs: Value("and", args),
            or_=lambda *args: Value("or", args),
            not_=lambda value: Value("not", (value,)),
            constexpr=constexpr,
            value=lambda *args: Value("value", args),
            record=self.record,
        )
        for name in ("eq", "ne", "lt", "le", "gt", "ge"):

            def operation(*args, name=name):
                return Value(name, args)

            setattr(self.M, name + "_", operation)
        entry.register_namespace("M", self.M)
        self.M.function = entry.make_decorator(self.M, namespace_path="M.function")

        @resolve_global_info_args("device", resolver=self.resolve_global_info)
        def Tensor(shape=None, dtype="float32", device=None, placement="S[0]"):
            return Value("tensor", (shape, dtype, device, placement))

        def dynamic(name, dtype="int64"):
            return Value("symbol", (dtype,), name)

        def cell(value=None):
            return Value("cell", (value,))

        self.M.Tensor = Tensor
        self.M.dynamic = dynamic
        self.M.int32 = protocol_registry.register_scalar_annotation(
            "M.int32", lambda: None, dtype="int32"
        )
        self.M.cell = protocol_registry.register_mutable_decl("M.cell")(cell)

    @contextmanager
    def context(self):
        with IRBuilder():
            yield self

    def get(self):
        return self.result

    def frame(self):
        return next(frame for frame in reversed(self.stack) if frame.kind == "function")

    def func_name_(self, name):
        self.frame().function.name = name
        self.frame().global_var = self.references.setdefault(name, Value("global", (name,)))

    def arg_(self, name, annotation, *, span=None, **kwargs):
        value = Value("arg", (annotation,), name)
        if span is not None:
            value = span(value)
        frame = self.frame()
        frame.params.append(value)
        frame.function.params.append(value)
        return value

    def func_ret_type_(self, annotation):
        self.frame().function.ret_type = annotation() if callable(annotation) else annotation

    def func_ret_value(self, value):
        statement = ("return", value)
        self.frame().function.body.append(statement)
        return AlreadyEmitted(statement)

    def resolve_type_var(self, name, dtype=None, **kwargs):
        value = self.frame().resolve_type_var(name, dtype, **kwargs)
        return value

    def resolve_global_info(self, name):
        return self.global_infos[name]

    def bind(self, value, *, name=None, **kwargs):
        if self.stack[-1].kind in ("then", "else"):
            self.stack[-1].result = value
        return value

    def emit(self, value, *, span=None):
        if isinstance(value, AlreadyEmitted):
            return
        if span is not None:
            value = span(value)
        self.frame().function.body.append(("emit", value))

    def record(self, value):
        return value

    def decl_mutable(self, value=None, *, name=None, **kwargs):
        value = Value("cell", (value,), name)
        self.statement("declare", name, value)
        return value

    def set_mutable(self, variable, value, **kwargs):
        return self.statement("set", variable, value)

    def for_frame(self, frame, *, names=None, span=None, **kwargs):
        frame.names = names
        if names is not None:
            if isinstance(names, str):
                names = (
                    (names,)
                    if len(frame.vars) == 1
                    else [f"{names}_{index}" for index in range(len(frame.vars))]
                )
            expanded = []
            for name in names:
                if name.startswith("*"):
                    expanded.extend(
                        f"{name[1:]}_{i}" for i in range(len(frame.vars) - len(names) + 1)
                    )
                else:
                    expanded.append(name)
            for variable, name in zip(frame.vars, expanded):
                variable.name = name
        if span is not None:
            span(frame)
        return frame

    def at(self, location, value):
        if isinstance(value, Value | Frame):
            value.span = tuple([*self.source_stack, location])
        return value

    def with_at_group(self, location, thunk, *, attach_result=True):
        self.source_stack.append(location)
        try:
            value = thunk()
            return self.at(location, value) if attach_result else value
        finally:
            self.source_stack.pop()

    def statement(self, kind, *operands):
        statement = (kind, operands)
        self.frame().function.body.append(statement)
        return AlreadyEmitted(statement)

    def setitem_(self, target, key, value, **kwargs):
        target[key] = value
        return self.statement("setitem", target, key, value)

    def setattr_(self, target, name, value, **kwargs):
        setattr(target, name, value)
        return self.statement("setattr", target, name, value)
