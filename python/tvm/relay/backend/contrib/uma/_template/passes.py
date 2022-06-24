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
"""Transform passes for the UltraTrail accelerator"""

import tvm
from tvm import relay, tir
from tvm.topi.utils import prod

from collections import OrderedDict


@tvm.tir.transform.prim_func_pass(opt_level=2)
def my_ai_hw_conv2d_pass(func, mod, ctx):
    _found_blocks = []
    _loops = []
    _handles = []
    _entry_node = None
    _external_function_name = "my_hw_ai_conv2dnchw"

    def _has_block(name: str, func) -> bool:
        """
        Determine of a tir.block with `name` exists in `func`
        """
        def _hb(op):
            if isinstance(op, tvm.tir.Block):
                _found_blocks.append(op.name_hint)

        _found_blocks = []
        tvm.tir.stmt_functor.post_order_visit(func.body, _hb)
        return name in _found_blocks

    def _transform_function2(
        func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.tir.PrimFunc:
        def _replace_conv2d(op):
            if op == _entry_node:
                irb = tvm.tir.ir_builder.create()
                # Collection of buffer address
                buffers = [b[1].data for b in _handles]
                # extraction of loop offsets
                for i in _loops:
                    assert i.min.value == 0
                offsets = [loop.extent.value for loop in _loops]
                args = buffers # + offsets
                external_call = tvm.tir.Evaluate(tir_call(irb, True, _external_function_name, *args))
                mac_calls = tvm.tir.SeqStmt([external_call])
                irb.emit(mac_calls)
                irb_result = irb.get()
                return irb_result
            return op

        sch = tir.Schedule(func)

        if _has_block("conv2d_nchw", func):
            conv2d_block = sch.get_block("conv2d_nchw")

            rv_loops = sch.get_loops(conv2d_block)
            assert len(rv_loops) == 7
            n, co, h, w, ci, kh, hw = rv_loops
            _entry_node = sch.get(rv_loops[1])
            _loops = [sch.get(i) for i in rv_loops]
            _handles = func.buffer_map.items()

            x = tvm.tir.stmt_functor.ir_transform(func.body, None, _replace_conv2d, ["tir.For"])
            return func.with_body(x)
        else:
            return func #sch.mod["main"]

    def _transform_function(
        func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.tir.PrimFunc:
        def _replace_conv2d(op):
            if isinstance(op, tvm.tir.For) and op.loop_var.name == "yy":
                irb = tvm.tir.ir_builder.create()
                # Collection of buffer address
                buffers = [b[1].data for b in func.buffer_map.items()]
                args = buffers # + offsets
                external_call = tvm.tir.Evaluate(tir_call(irb, True, _external_function_name, *args))
                mac_calls = tvm.tir.SeqStmt([external_call])
                irb.emit(mac_calls)
                irb_result = irb.get()
                return irb_result
            return op

        x = tvm.tir.stmt_functor.ir_transform(func.body, None, _replace_conv2d, ["tir.For"])
        return func.with_body(x)

    r = _transform_function2(func, mod, ctx)
    return r


def tir_call(ib: tvm.tir.ir_builder, extern: bool, name: str, *args):
    """
    ib: ir_builder
    extern: bool
        True  --> tvm.tir.call_extern
        False --> tvm.tir.call_packed
    name: str
        function name
    *args:
        arguments for function call
    """

    def buf_from_array(ib, arr, dtype):
        # Allocate enough memory to store the whole array
        var = ib.allocate("int32", (len(arr),), scope="global")
        for i, v in enumerate(arr):
            var[i] = v
        # Declare a buffer, which is basically a view on the chunk of memory that we allocated previously
        buf = tvm.tir.decl_buffer((len(arr),), dtype, data=var, scope="global")
        return buf

    if extern:
        args = [i.data if isinstance(i, tvm.tir.Buffer) else i for i in args]
        call = tvm.tir.call_extern("int32", name, *args)
    else:
        args = [
            buf_from_array(ib, i, "int32") if isinstance(i, (tuple, list, tvm.ir.container.Array)) else i for i in args
        ]
        call = tvm.tir.call_packed(name, *args)

    return call
