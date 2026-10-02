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
# pylint: disable=redefined-builtin, wrong-import-order, no-member, invalid-name
"""Public exports for the Relax builder."""

from tvm.script.ir_builder.ir import constexpr

from . import distributed as dist
from . import ir as _native
from . import op as _op
from . import parser_protocol as _protocol
from .ir import *
from .op import *
from .parser_protocol import *
from .parser_protocol import (
    __tvm_value_if__,
    _check_module_well_formed,
    supports_mutable_declarations,
)

__all__ = [*_native.__all__, *_op.__all__, *_protocol.__all__, "constexpr", "dist"]


def _register_printer_names():
    import types  # pylint: disable=import-outside-toplevel

    from tvm.ir import Op, register_op_attr  # pylint: disable=import-outside-toplevel

    active = set()

    def register(value, path):
        # These constructors use specialized out_ty/argument conventions.
        if not callable(value) or path.rsplit(".", 1)[-1].startswith("call_tir"):
            return
        try:
            op = Op.get(path)
        except AttributeError:
            return
        if op.has_attr("TScriptPrinterName") and op.get_attr("TScriptPrinterName") == path:
            return
        register_op_attr(path, "TScriptPrinterName", path)

    def visit(value, path):
        if id(value) in active:
            return
        active.add(id(value))
        register(value, path)
        for name in getattr(value, "__all__", dir(value)):
            if name.startswith("_"):
                continue
            try:
                member = getattr(value, name)
            except AttributeError:
                continue
            member_path = f"{path}.{name}"
            if isinstance(member, types.ModuleType) and member.__name__.startswith("tvm.relax.op."):
                visit(member, member_path)
            elif callable(member):
                register(member, member_path)
        active.remove(id(value))

    visit(_op, "relax")
    visit(dist, "relax.dist")


_register_printer_names()
