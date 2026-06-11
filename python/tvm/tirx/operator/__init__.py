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

# `tile_primitive` defines Python Op classes (`Zero(UnaryOp)`, etc.) whose
# class bodies call `Op.get("tirx.<name>")` at class-definition time, which
# requires the compiler-side FFI. Load it lazily so that
# `tvm.tirx.operator.intrinsics._common` (pure data) and other runtime-safe
# submodules can be imported under `TVM_USE_RUNTIME_LIB=1`, matching apache's
# discipline for `tvm.tirx`.
def __getattr__(name):
    # `from . import tile_primitive` here would recurse: Python's import
    # machinery does `getattr(self, 'tile_primitive')` to see if the submodule
    # is already loaded, which goes back through this __getattr__. Use
    # importlib.import_module to bypass attribute lookup; it sets the attribute
    # on the parent package as a side effect, so subsequent lookups go through
    # the normal attribute path, not this __getattr__.
    import sys  # pylint: disable=import-outside-toplevel
    from importlib import import_module  # pylint: disable=import-outside-toplevel

    tp_qualname = f"{__name__}.tile_primitive"
    tile_primitive = sys.modules.get(tp_qualname) or import_module(tp_qualname)
    if hasattr(tile_primitive, name):
        return getattr(tile_primitive, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["get_tirx_op"]
