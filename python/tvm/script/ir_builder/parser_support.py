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
"""Shared construction helpers used by generated TVMScript programs.

Generated programs bind ``_PS = I.parser_support`` with a collision-free alias.
``TypeVarFrame`` belongs to one function and remains shared by its signature and
body. ``lookup_global_info`` resolves unchanged module reference strings; dtype
and placement strings are not expression syntax.

Explicit constexpr conditions select Python branches in generated programs;
unmarked expressions use dialect IR constructors directly. Source calls use
``with_at_scope(loc, thunk)``;
non-call values use ``at(loc, value)``. Both preserve returned object identity,
and the former restores caller source context even when evaluation raises.

A standalone value is consumed by ``X.emit_(value)``. Self-emitting builders
return ``BypassEmit(stmt)`` so source attachment reaches the stored statement
without emitting it a second time. Bindings and generated keyword operations
retain explicit ``span=`` for their separate source locations.
"""

import re

from .base import IRBuilder, _construction_span, at
from .type_var_frame import TypeVarFrame as TypeVarFrame


def lookup_global_info(content):
    """Resolve an unchanged module reference, or pass a concrete value through."""
    if not isinstance(content, str):
        return content
    from .ir.frame import IRModuleFrame

    if not IRBuilder.is_in_scope():
        raise ValueError("Global-info lookup requires an enclosing module frame")
    frame = next(
        (
            frame
            for frame in reversed(IRBuilder.current().frames)
            if isinstance(frame, IRModuleFrame)
        ),
        None,
    )
    if frame is None:
        raise ValueError("Global-info lookup requires an enclosing module frame")
    match = re.fullmatch(r"([^\[\]]+)\[(\d+)\]", content)
    if match:
        name, index = match.groups()
        return frame.global_infos[name][int(index)]
    from .ir.ir import lookup_vdevice

    selector = re.fullmatch(r"([^:\[\]]+)(?::(\d+)(?::([^:]+))?)?", content)
    if selector is None:
        raise ValueError(f"Invalid global-info reference: {content!r}")
    target, index, _scope = selector.groups()
    # Printed references include the selected device's memory scope. Selection
    # uses its ordinal among devices of this target kind and retains that exact
    # module-owned object, including its stored scope.
    return lookup_vdevice(target, int(index) if index is not None else 0)


def with_at_scope(location, thunk):
    """Evaluate a source call once under its location and preserve its result."""
    with _construction_span(location):
        return at(location, thunk())
