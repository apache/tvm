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
"""TIRX-layer TVMScript pieces (parser, builder).

After the per-dialect TVMScript restructure, the TIRX layer owns its own
``script/{parser,builder}`` subpackages. ``tvm.script.tirx`` resolves to
this module via the dialect registry, so the public parser surface
(``prim_func``, ``Buffer``, ``Ptr``, etc.) is re-exported here.
"""

# pylint: disable=redefined-builtin,wildcard-import,unused-wildcard-import
from .parser import *
from .parser import Buffer, Ptr, prim_func

try:
    from .parser import macro
except ImportError:
    macro = None
from .builder.ir import TensorMap, meta_class
from .builder.tirx import *


def __getattr__(name: str):
    """Resolve undefined attributes as dynamic TilePrimitiveCall ops.

    Registers ``tirx.<name>`` lazily so the op is available for IR walks
    after the prim_func is built.
    """
    if name.startswith("_"):
        raise AttributeError(f"module 'tvm.tirx.script' has no attribute {name!r}")
    import tvm_ffi

    from tvm.ir import Op
    from tvm.tirx.stmt import TilePrimitiveCall

    op_name = "tirx." + name
    _register_op = tvm_ffi.get_global_func("ir.RegisterOp")
    from tvm.ir import register_op_attr

    def _fn(*args, workspace=None, config=None, dispatch=None, **kwargs):
        try:
            op = Op.get(op_name)
        except Exception:
            _register_op(op_name, "")
            register_op_attr(op_name, "TIsTIRxOp", True)
            op = Op.get(op_name)
        if workspace is None:
            workspace = {}
        if config is None:
            config = kwargs or {}
        # Convert Buffer args to BufferRegion (covers full extent)
        from tvm.tirx import Buffer as _TBuffer

        new_args = []
        for a in args:
            if isinstance(a, _TBuffer):
                slices = [slice(None) for _ in range(len(a.shape))]
                a = a[slices]
            new_args.append(a)
        # Insert into the active frame using same FFI hook as registered ops.
        from .builder.tirx import f_insert as _f_insert

        return _f_insert(TilePrimitiveCall(*new_args, op=op, workspace=workspace, config=config))

    _fn.__name__ = name
    return _fn
