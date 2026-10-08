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
"""Primitive operators in the TVM IR."""

import keyword
import sys
from collections.abc import Sequence
from types import ModuleType, SimpleNamespace

import tvm_ffi

from . import _ffi_api
from .expr import Expr

_RESULT_TYPE_UNSET = object()


def _make_op_api(op, module_name):
    """Build a callable whose operands and result are governed by its Op."""
    from .attrs import make_node  # pylint: disable=import-outside-toplevel
    from .expr import Call, reinfer_type  # pylint: disable=import-outside-toplevel

    def call(
        *args,
        attrs=None,
        ty_args=None,
        span=None,
        ty=_RESULT_TYPE_UNSET,
        ret_ty=_RESULT_TYPE_UNSET,
        **kwargs,
    ):
        # Bind named operands using the registered signature, without inventing
        # defaults or interpreting arbitrary Python wrapper signatures.
        operands = list(args)
        for index, info in enumerate(op.args_info):
            if index < len(args):
                if info.name in kwargs:
                    raise TypeError(f"{op.name}: multiple values for {info.name!r}")
            elif info.name in kwargs:
                operands.append(kwargs.pop(info.name))
            else:
                raise TypeError(f"{op.name}: missing operand {info.name!r}")
        if kwargs:
            if not op.attrs_type_key:
                raise TypeError(f"{op.name}: unexpected keyword operands {tuple(kwargs)}")
            if attrs is not None:
                raise TypeError(f"{op.name}: cannot mix attrs with attribute keywords")
            attrs = make_node(op.attrs_type_key, **kwargs)
        if ty is not _RESULT_TYPE_UNSET:
            if ret_ty is not _RESULT_TYPE_UNSET:
                raise TypeError(f"{op.name}: ty and ret_ty cannot both be specified")
            ret_ty = ty
        elif (ret_ty is _RESULT_TYPE_UNSET or ret_ty is None) and (
            op.get_attr("TFixedReturnType") is not None or op.get_attr("FInferType") is not None
        ):
            provisional = Call.unchecked(op, operands, attrs=attrs, ty_args=ty_args, span=span)
            ret_ty = reinfer_type(provisional)
        if ret_ty is _RESULT_TYPE_UNSET:
            ret_ty = None
        return Call(op, operands, attrs=attrs, ty_args=ty_args, span=span, ty=ret_ty)

    call.__name__ = op.name.rsplit(".", 1)[-1]
    call.__module__ = module_name
    call.__doc__ = op.doc or f"Construct a call to {op.name}."
    call.__tvm_op__ = op
    call.__tvm_standard_call__ = True
    return call


def _init_op_api(namespace, target_module_name=None):
    """Initialize registered Op callables in an already loaded Python module.

    Like :func:`tvm_ffi.init_ffi_api`, the registry prefix comes first and the
    target defaults to that module name. For example, a backend module uses
    ``_init_op_api("tirx.cuda", __name__)``. Dotted suffixes require existing
    module or SimpleNamespace containers; underscores are ordinary name parts.

    Generated functions accept registered positional/named operands plus
    ``attrs``, ``ty_args``, ``span`` and ``ty`` (also spelled ``ret_ty``).
    Omitting the result invokes an available Op inference hook; without one,
    Call retains a missing type.
    Explicit results and inference/validation errors are preserved. Attribute
    keywords construct the registered attrs schema, when one is declared.

    Existing generated functions are reused. A deliberate wrapper may declare
    ``__tvm_op__ = Op.get(name)`` to retain ownership of that name. This declares
    identity, not semantic equivalence: the wrapper must accept printed calls
    or have an appropriate exceptional printer hook. Other collisions fail
    before any functions are installed. Script namespaces separately publish
    their callable names to the printer after initialization. Returns None.
    """
    if not namespace or any(
        not part.isidentifier() or keyword.iskeyword(part) for part in namespace.split(".")
    ):
        raise ValueError(f"Invalid Op namespace {namespace!r}")
    target = sys.modules[target_module_name or namespace]
    pending = []
    destinations = set()
    for name in sorted(Op.list_op_names()):
        if not name.startswith(namespace + "."):
            continue
        parts = name[len(namespace) + 1 :].split(".")
        if any(not part.isidentifier() or keyword.iskeyword(part) for part in parts):
            raise ValueError(f"Op {name!r} has no Python attribute spelling")
        container = target
        for part in parts[:-1]:
            container = vars(container).get(part)
            if not isinstance(container, ModuleType | SimpleNamespace):
                raise ValueError(f"Op {name!r} requires an existing namespace at {part!r}")
        destination = (id(container), parts[-1])
        if destination in destinations:
            raise ValueError(f"Op {name!r} aliases another exposure destination")
        destinations.add(destination)
        op = Op.get(name)
        if parts[-1] in vars(container):
            current = vars(container)[parts[-1]]
            identity = getattr(current, "__tvm_op__", None)
            if not callable(current) or not isinstance(identity, Op) or not identity.same_as(op):
                raise ValueError(f"Op {name!r} conflicts with an existing Python attribute")
        else:
            pending.append((container, parts[-1], _make_op_api(op, target.__name__)))
    for container, name, call in pending:
        setattr(container, name, call)


@tvm_ffi.register_object("ir.Op")
class Op(Expr):
    """Primitive operator in the IR."""

    def __init__(self):
        raise RuntimeError("Cannot create op, use get instead")

    @staticmethod
    def get(op_name):
        """Get a registered operator by name.

        Parameters
        ----------
        op_name : str
            The canonical operator name.

        Returns
        -------
        Op
            A handle to the registered operator.
        """
        return _ffi_api.GetOp(op_name)

    @staticmethod
    def list_op_names():
        """List registered operator names in unspecified order.

        Returns
        -------
        list[str]
            The registered operator names.
        """
        return _ffi_api.ListOpNames()

    def set_attr(self, attr_name, value, override=False):
        """Set an operator attribute.

        Parameters
        ----------
        attr_name : str
            Attribute column name.
        value : object
            Non-None attribute value.
        override : bool, optional
            Replace an existing value if True. Duplicate registration otherwise
            raises ValueError. Cached views observe replacements; no history is kept.

        Returns
        -------
        None
        """
        self._set_attr(attr_name, value, override)

    def set_signature(self, args=(), *, ty_args=(), var_args=None, var_ty_args=None) -> None:
        """Replace this Op's argument signature and validate Call arity.

        Each entry is a name string or a ``(name, doc)`` tuple of strings.
        Fixed entries form required prefixes; a variadic entry allows zero or
        more additional arguments. Calls must have exactly the required count
        without a tail, or at least that count with one. This method does not
        check argument types or Call attrs. It replaces a generated typed
        validator with a count-only one, while an existing custom validator
        keeps precedence.

        Parameters
        ----------
        args : sequence[str or tuple[str, str]], optional
            Fixed value arguments, in order.
        ty_args : sequence[str or tuple[str, str]], optional
            Fixed type arguments, in order.
        var_args : str or tuple[str, str], optional
            Variadic value-argument tail.
        var_ty_args : str or tuple[str, str], optional
            Variadic type-argument tail.

        Returns
        -------
        None
        """

        def parse_entry(entry, label):
            if isinstance(entry, str):
                return entry, ""
            if (
                isinstance(entry, tuple)
                and len(entry) == 2
                and all(isinstance(part, str) for part in entry)
            ):
                return entry
            raise TypeError(f"{label} must be a name string or a (name, doc) tuple")

        def parse_fixed(entries, label):
            if not isinstance(entries, Sequence) or isinstance(entries, str | bytes):
                raise TypeError(f"{label} must be a sequence of signature entries")
            parsed = [parse_entry(entry, label) for entry in entries]
            return [name for name, _ in parsed], [doc for _, doc in parsed]

        arg_names, arg_docs = parse_fixed(args, "args")
        ty_arg_names, ty_arg_docs = parse_fixed(ty_args, "ty_args")
        value_tail = [] if var_args is None else list(parse_entry(var_args, "var_args"))
        type_tail = [] if var_ty_args is None else list(parse_entry(var_ty_args, "var_ty_args"))
        self._set_signature(arg_names, arg_docs, ty_arg_names, ty_arg_docs, value_tail, type_tail)


def register_op_attr(op_name, attr_key, value=None, override=False):
    """Register an operator property of an operator by name.

    Parameters
    ----------
    op_name : str
        The name of operator

    attr_key : str
        The attribute name.

    value : object, optional
        The value to set

    override : bool, optional
        Replace an existing value if True; otherwise duplicate registration raises
        ValueError. Cached views observe replacements; no priority history is kept.

    Returns
    -------
    result : object or function
        The registered value when supplied, or a decorator that registers and
        returns its argument. The named Op is created if it does not exist.
    """

    def _register(v):
        """internal register function"""
        _ffi_api.RegisterOpAttr(op_name, attr_key, v, override)
        return v

    return _register(value) if value is not None else _register
