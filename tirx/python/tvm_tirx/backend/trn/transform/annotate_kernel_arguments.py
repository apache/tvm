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
"""Apply the Trainium input/output contract after host/device splitting."""

import tvm_ffi

import tvm_tirx
from tvm_tirx import tirx
from tvm_tirx.ir import Bind, Call, GlobalVar, Op, Var
from tvm_tirx.ir.transform import module_pass


@module_pass(opt_level=0, name="AnnotateKernelArguments")
class AnnotateKernelArguments:
    """Order kernel inputs before outputs and annotate their per-kernel input count.

    Run after ``SplitHostDevice`` and before ``MakePackedAPI``. The source
    ``trn.num_input_args`` contract remains on each host function until its
    kernel calls have been matched to the surviving host parameters and captures.
    Functions without extracted kernels retain their own contract.
    """

    def transform_module(self, mod, _ctx):
        attr = "trn.num_input_args"
        kernels = {
            gv.name_hint: gv
            for gv, func in mod.functions.items()
            if isinstance(func, tirx.Function)
            and func.attrs.get("target") is not None
            and func.attrs["target"].kind.name == "trn"
            and func.attrs.get("tirx.is_global_func", False)
            and attr not in func.attrs
        }
        updates = {}
        for gv, host in mod.functions.items():
            if not isinstance(host, tirx.Function) or attr not in host.attrs:
                continue
            num_inputs = host.attrs[attr]
            if not isinstance(num_inputs, int) or not 0 <= num_inputs <= len(host.params):
                raise ValueError("trn.num_input_args must be between zero and the parameter count")
            param_order = {param: i for i, param in enumerate(host.params)}
            has_kernel = False
            bindings = {}
            tvm_ffi.structural_walk(
                host.body, (Bind, lambda bind: bindings.update({bind.var: bind.value}))
            )

            def reorder(call):
                nonlocal has_kernel
                offset = 0
                if isinstance(call.op, GlobalVar):
                    name = call.op.name_hint
                elif isinstance(call.op, Op) and call.op.name == "tirx.call_ffi_kernel":
                    name = call.args[0].value
                    offset = 1
                else:
                    return call
                if name not in kernels:
                    return call
                has_kernel = True
                kernel_gv = kernels[name]
                kernel = mod[kernel_gv]
                args = list(call.args[offset : offset + len(kernel.params)])

                def source_position(arg):
                    # A device region may capture a tensor alias declared on the host.
                    # Follow its pointer projection back to the source parameter.
                    while True:
                        if isinstance(arg, Var):
                            if arg in param_order:
                                return param_order[arg]
                            if arg not in bindings:
                                return None
                            arg = bindings[arg]
                        elif (
                            isinstance(arg, Call)
                            and arg.args
                            and isinstance(arg.op, Op)
                            and arg.op.name in ("tirx.tensor_data_ptr", "tirx.decl_tensor")
                        ):
                            arg = arg.args[0]
                        else:
                            return None

                positions = [source_position(arg) for arg in args]

                # Captures are additional inputs. Only surviving host outputs are mutable
                # outputs; unused host inputs must not inflate the kernel's input count.
                def sort_key(i):
                    position = positions[i]
                    if position is None:
                        return (1, i)
                    return (2 if position >= num_inputs else 0, position)

                order = sorted(range(len(args)), key=sort_key)
                input_count = sum(pos is None or pos < num_inputs for pos in positions)
                updates[kernel_gv] = tirx.Function(
                    [kernel.params[i] for i in order],
                    kernel.body,
                    kernel.ret_type,
                    kernel.attrs,
                    kernel.loc,
                ).with_attr(attr, input_count)
                return Call(
                    call.op,
                    [
                        *call.args[:offset],
                        *(args[i] for i in order),
                        *call.args[offset + len(args) :],
                    ],
                    attrs=call.attrs,
                    ty_args=call.ty_args,
                    loc=call.loc,
                    ty=call.ty,
                )

            updated = host.with_body(
                tvm_ffi.structural_map(host.body, (Call, reorder), order="post")
            )
            if has_kernel:
                updates[gv] = updated.without_attr(attr)
        mod.update(tvm_tirx.IRModule(updates))
        return mod
