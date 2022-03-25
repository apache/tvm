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


class LayerConfigGenerator(relay.ExprVisitor):
    def __init__(self, acc_spec, config):
        super(LayerConfigGenerator, self).__init__()
        self.acc_spec = acc_spec
        self.config = config

    def _extract_config_from_call(self, call):
        # Config extraction assumes that the pattern matching already performed all validity checks
        if call.op.name == "nn.conv1d":
            self.config["ch_in_len"] = int(call.type_args[0].shape[2])
            self.config["ch_in_blk"] = int(call.type_args[0].shape[1]) // self.acc_spec["array_dim"]
            self.config["kernel_shape"] = int(call.attrs.kernel_size[0])
            self.config["stride"] = int(call.attrs.strides[0])
            self.config["pad"] = int(call.attrs.padding[0] > 0)
        elif call.op.name == "nn.bias_add":
            self.config["bias"] = 1
        elif call.op.name == "nn.relu":
            self.config["relu"] = 1

    def visit_call(self, call):
        self._extract_config_from_call(call)
        for a in call.args:
            self.visit(a)


class SubgraphConfigGenerator(relay.ExprVisitor):
    def __init__(self, acc_spec):
        super(SubgraphConfigGenerator, self).__init__()
        self.acc_spec = acc_spec
        self.config = []

    def _compute_dataflow_control(self):
        # The graph-based execution order and corresponding configuration should be computed here.
        # Currently only switches between two feature memories (no parallel pathes).
        self.config.reverse()

        mem_ctrl = 0x0
        for c in self.config:
            c["mem_ctrl"] = mem_ctrl
            mem_ctrl = 0x2 if mem_ctrl == 0x0 else 0x0

        self.config[-1]["last"] = 1

    def visit_function(self, func):
        layer_config = {k: 0 for k in self.acc_spec["conf_reg_layer_bits"].keys()}
        layer_config["ch_out_len"] = int(func.ret_type.shape[2])
        layer_config["ch_out_blk"] = int(func.ret_type.shape[1]) // self.acc_spec["array_dim"]
        LayerConfigGenerator(self.acc_spec, layer_config).visit(func.body)
        self.config.append(layer_config)

    def generate_config(self, func):
        self.visit(func.body)
        self._compute_dataflow_control()
        return self.config


@tvm.ir.transform.module_pass(opt_level=1)
class ConfigGenerator:
    """This pass generates a configuration string for the UltraTrail accelerator 
    for each partitioned relay subgraph."""

    def __init__(self):
        # Note: This information should eventually be passed as an accelerator description.
        self.acc_spec = {
            "array_dim": 8,
            "conf_reg_layers": 16,
            "conf_reg_layer_bits": OrderedDict(
                {
                    "mem_ctrl": 4,
                    "ch_in_len": 7,
                    "ch_in_blk": 4,
                    "ch_out_len": 7,
                    "ch_out_blk": 4,
                    "kernel_shape": 4,
                    "stride": 3,
                    "avg_pool_exp": 3,
                    "pad": 1,
                    "relu": 1,
                    "bias": 1,
                    "avg": 1,
                    "early_exit": 1,
                    "last": 1,
                }
            ),
        }

    def _config_to_bitstring(self, config):
        bitstring = ""
        for layer in config:
            for k, v in self.acc_spec["conf_reg_layer_bits"].items():
                bitstring += "{:0{}b}".format(layer[k], v)
        return bitstring

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:

        # Create UltraTrail specific config based on matched subgraph
        for gv, func in mod.functions.items():
            if func.attrs is not None and func.attrs["Compiler"] == "ultra_trail":
                config = SubgraphConfigGenerator(self.acc_spec).generate_config(func)
                config_bitstring = self._config_to_bitstring(config)
                mod.update_func(gv, func.with_attr("ut_config", config_bitstring))

        return mod


@tvm.ir.transform.module_pass(opt_level=1)
class BufferScopeAnnotator:
    """This pass annotates the params of an offloaded function with the target UltraTrail memory."""

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        class FunctionVisitor(relay.ExprVisitor):
            def __init__(self):
                super(FunctionVisitor, self).__init__()
                self._var_scope_map = {}

            def visit_call(self, call):
                if call.op.name == "nn.conv1d":
                    self._var_scope_map[call.args[1]] = "wmem"
                elif call.op.name == "nn.bias_add":
                    self._var_scope_map[call.args[1]] = "bmem"

                for a in call.args:
                    self.visit(a)

            def var_scope_map(self, func):
                self.visit(func.body)
                return self._var_scope_map

        # Annotate buffer scopes
        for gv, func in mod.functions.items():
            if func.attrs is not None and func.attrs["Compiler"] == "ultra_trail":
                var_scope_map = FunctionVisitor().var_scope_map(func)
                scopes = [
                    var_scope_map[var] if var in var_scope_map else "fmem" for var in func.params
                ]
                mod.update_func(gv, func.with_attr("ut_buffer_scopes", scopes))

        return mod

def insert_extern_calls(sch):
    def extern_calls():
        calls = []
        buffer_scopes = list(sch.mod["main"].attrs["relay_attrs"]["ut_buffer_scopes"])
        buffer_scopes.reverse() # for some reason TIR params are reversed to relay function
        for i, buffer_scope in enumerate(buffer_scopes):
            buffer = sch.mod["main"].buffer_map[sch.mod["main"].params[i]]
            size = prod(buffer.shape)
            var = buffer.data
            call = tir.call_extern("int32", f"load_{buffer_scope}", var, size)
            calls.append(tir.Evaluate(call))
        seq = tir.stmt_seq(*calls)
        return tir.Block([], [], [], "call_extern", seq)

    root_sref = sch.get_sref(sch.get_block("root"))
    sch.state.replace(root_sref, extern_calls())

    return sch

@tvm.tir.transform.prim_func_pass(opt_level=1)
class CodegenGenerateExternCalls:
    def transform_function(
        self, func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.tir.PrimFunc:
        sch = tir.Schedule(func)
        sch = insert_extern_calls(sch)
        return sch.mod["main"]
