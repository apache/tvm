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
"""Transform the layout of nn.conv2d weights to preferred layout for exported subgraphs."""
import numpy as np
import tvm

NN_CONV2D_OP = tvm.relay.op.get("nn.conv2d")


class TransformConv2dWeightLayout(tvm.relay.ExprMutator):
    """Transform the layout of nn.conv2d weights to preferred layout for exported subgraphs.

    Parameters
    ----------------------
    external_compiler: str
        The name of BYOC external compiler.

    target_layout: str
        The target layout for nn.conv2d weights.
    """

    def __init__(self, external_compiler, target_layout):
        super().__init__()
        self._external_compiler = external_compiler
        self._target_layout = target_layout

    def __call__(self, mod, params):
        """Transform the layout of nn.conv2d weights to preferred layout for exported subgraphs.

        Parameters
        ----------
        mod: tvm.IRModule
            The transform target module.

        params: dict of str to tvm.runtime.NDArray
            The corresponding parameter inputs to mod.

        Returns
        -------
        mod: tvm.IRModule
            The transformed mod.

        params: dict of str to tvm.runtime.NDArray
            The transformed params.
        """
        assert isinstance(mod, tvm.IRModule)
        assert isinstance(params, dict)

        self._mod = mod
        self._params = params
        self._call_stack = []
        self._transformed_vars = []
        self._in_export_func = False
        self._mod["main"] = self.visit(mod["main"])
        self._mod = tvm.relay.transform.InferType()(self._mod)

        return self._mod, self._params

    def visit_call(self, call):
        if isinstance(call.op, tvm.ir.Op):
            if (
                call.op == NN_CONV2D_OP
                and call.attrs["kernel_layout"] != self._target_layout
                and self._in_export_func
            ):
                transpose_idx = [call.attrs["kernel_layout"].index(d) for d in self._target_layout]
                assert len(self._call_stack) % 2 == 1

                self._call_stack.append(call)

                # Check if kernel is straight var
                weight_arg_idx = 1
                for i in range(len(self._call_stack) - 1, -1, -1):
                    func = self._call_stack[i]
                    if i % 2 == 0:
                        assert isinstance(func, tvm.relay.Function)
                        weight_arg_idx = list(func.params).index(weight_arg)
                    else:
                        assert isinstance(func, tvm.relay.Call)
                        weight_arg = func.args[weight_arg_idx]
                        if not isinstance(weight_arg, tvm.relay.Var):
                            self._call_stack.pop()
                            return super().visit_call(call)
                call = self._call_stack.pop()

                # Mutate
                new_attrs = dict(call.attrs)
                new_attrs["kernel_layout"] = self._target_layout
                call = tvm.relay.nn.conv2d(call.args[0], call.args[1], **new_attrs)
                self._call_stack.append(call)
                weight_arg_idx = 1
                for i in range(len(self._call_stack) - 1, -1, -1):
                    func = self._call_stack[i]
                    if i % 2 == 0:  # Function
                        weight_param = weight_arg
                        weight_param_idx = list(func.params).index(weight_param)

                        new_params = [
                            p for p_i, p in enumerate(func.params) if p_i != weight_param_idx
                        ]
                        new_weight_param = self.memo_map[weight_param]
                        new_params.insert(weight_param_idx, new_weight_param)
                        func = tvm.relay.Function(
                            params=list(new_params),
                            body=func.body,
                            ret_type=func.ret_type,
                            type_params=func.type_params,
                            attrs=func.attrs,
                        )

                        weight_arg_idx = weight_param_idx
                    else:  # Call
                        weight_arg = func.args[weight_arg_idx]
                        if (
                            weight_arg in self._transformed_vars
                        ):  # visited, perhaps it's a shared weight?
                            call = self._call_stack.pop()
                            return super().visit_call(call)
                        new_args = [a for a_i, a in enumerate(func.args) if a_i != weight_arg_idx]
                        new_shape = tuple(
                            [weight_arg.checked_type.shape[idx] for idx in transpose_idx]
                        )
                        new_weight_arg = tvm.relay.var(
                            name_hint=weight_arg.name_hint,
                            shape=new_shape,
                            dtype=weight_arg.checked_type.dtype,
                        )
                        self.memo_map[weight_arg] = new_weight_arg
                        self._transformed_vars.append(weight_arg)
                        new_args.insert(weight_arg_idx, new_weight_arg)
                        func = tvm.relay.Call(
                            op=func.op,
                            args=new_args,
                            attrs=func.attrs,
                            type_args=func.type_args,
                        )
                    self._call_stack[i] = func
                call = self._call_stack.pop()

                main_weight_param = str(self._mod["main"].params[weight_arg_idx].name_hint)
                if main_weight_param in self._params:
                    weight_ndarray = self._params[main_weight_param]
                    self._params[main_weight_param] = tvm.runtime.ndarray.array(
                        np.transpose(weight_ndarray.asnumpy(), transpose_idx)
                    )
        elif isinstance(call.op, (tvm.relay.Function, tvm.relay.GlobalVar)):
            self._call_stack.append(call)
            self.visit(call.op)
            call = self._call_stack.pop()
        return super().visit_call(call)

    def visit_function(self, fn):
        is_export_fn = getattr(fn.attrs, "Compiler", "") == self._external_compiler
        self._call_stack.append(fn)
        if is_export_fn:
            assert not self._in_export_func
            self._in_export_func = True
        self.visit(fn.body)
        if is_export_fn:
            assert self._in_export_func
            self._in_export_func = False
        fn = self._call_stack.pop()
        return super().visit_function(fn)

    def visit_global_var(self, gvar):
        if isinstance(self._mod[gvar], tvm.relay.Function):
            self._mod[gvar] = self.visit_function(self._mod[gvar])
        return super().visit_global_var(gvar)

    def visit_var(self, var):
        assert var not in self._transformed_vars
        return super().visit_var(var)
