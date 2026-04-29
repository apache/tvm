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
"""tvm.script.ir_builder is a generic IR builder for TVM.

Per-dialect builder submodules (``tvm.script.ir_builder.tirx`` etc.) resolve
via :data:`tvm.script._DIALECT_REGISTRY`. Each extension dialect's package
registers a mapping from short name to ``tvm.<dialect>.script``; this
module's ``__getattr__`` then looks up ``f"{path}.builder"`` and returns it.
The IR layer is foundational and lives as a real submodule
(``tvm.script.ir_builder.ir``).
"""

import importlib
from typing import Any

from .base import IRBuilder


def __getattr__(name: str) -> Any:
    # Lazy import to avoid loading tvm.script during dialect bootstrap.
    from tvm.script import _DIALECT_REGISTRY  # pylint: disable=import-outside-toplevel

    if name in _DIALECT_REGISTRY:
        module = importlib.import_module(f"{_DIALECT_REGISTRY[name]}.builder")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'tvm.script.ir_builder' has no attribute {name!r}")
