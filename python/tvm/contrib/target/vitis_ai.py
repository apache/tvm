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
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

"""Utility to offload (sub-)models to Vitis-AI"""

import warnings

import pyxir
import pyxir.frontend.tvm

from tvm.relay.expr import Tuple, Call, TupleGetItem
import tvm._ffi


class CodegenVitisAI:

    """Traverse Relay expression and convert into PyXIR XGraph format"""

    def __init__(self, model_name, function):
        self.model_name = model_name
        self.function = function
        self.params = {}

    def convert_pyxir(self, target):
        """Convert Relay expression to PyXIR XGraph"""
        xgraph = pyxir.frontend.tvm.from_relay(
            self.function, params=self.params, postprocessing=None
        )
        xgraph = pyxir.partition(xgraph, targets=[target])
        return xgraph

    def get_output_names(self):
        """Get output names from Relay expression"""
        func = self.function
        output_relay_ids = []
        expr = func.body
        if isinstance(expr, Tuple):
            for field in expr.fields:
                output_relay_ids.append(hash(field))
        elif isinstance(expr, Call):
            output_relay_ids.append(hash(expr))
        elif isinstance(expr, TupleGetItem):
            output_relay_ids.append(hash(expr.tuple_value))
        else:
            raise ValueError("Vitis-AI codegen does not support {} as output".format(type(expr)))
        return output_relay_ids


@tvm._ffi.register_func("relay.ext.vitis_ai")
def vitis_ai_compiler(ref):
    """Create a Vitis-AI runtime from the provided Relay expression"""
    assert isinstance(ref, tvm.relay.function.Function)

    out_tensor_names = []
    name = str(ref.attrs.global_symbol)

    pass_context = tvm.get_global_func("transform.GetCurrentPassContext")()

    # The target Vitis-AI accelerator device
    target = (
        str(pass_context.config["relay.ext.vitis_ai.options.target"])
        if "relay.ext.vitis_ai.options.target" in pass_context.config
        else None
    )

    # (Optional configs) The build and work directories to be used by Vitis-AI
    vai_build_dir = (
        str(pass_context.config["relay.ext.vitis_ai.options.build_dir"])
        if "relay.ext.vitis_ai.options.build_dir" in pass_context.config
        else tvm.contrib.utils.tempdir().relpath("")
    )
    vai_work_dir = (
        str(pass_context.config["relay.ext.vitis_ai.options.work_dir"])
        if "relay.ext.vitis_ai.options.work_dir" in pass_context.config
        else tvm.contrib.utils.tempdir().relpath("")
    )

    # (Optional configs) Export and load PyXIR runtime module to file if provided. This is used to
    #   compile and quantize a model on the host and deploy it at the edge
    export_runtime_module = (
        str(pass_context.config["relay.ext.vitis_ai.options.export_runtime_module"])
        if "relay.ext.vitis_ai.options.export_runtime_module" in pass_context.config
        else ""
    )
    load_runtime_module = (
        str(pass_context.config["relay.ext.vitis_ai.options.load_runtime_module"])
        if "relay.ext.vitis_ai.options.load_runtime_module" in pass_context.config
        else ""
    )

    # Config checks
    if load_runtime_module and target is not None:
        warnings.warn(
            "Both `load_runtime_module` and `target` configs were specified."
            " The `load_runtime_module` points to a prebuilt runtime module with"
            " an internal target so the `target` config will be ignored"
        )
    if load_runtime_module and "relay.ext.vitis_ai.options.build_dir" in pass_context.config:
        warnings.warn(
            "Both `load_runtime_module` and `build_dir` configs were specified."
            " The `load_runtime_module` points to a prebuilt runtime module with"
            " an internal build directory so the `build_dir` config will be ignored"
        )
    if load_runtime_module and "relay.ext.vitis_ai.options.work_dir" in pass_context.config:
        warnings.warn(
            "Both `load_runtime_module` and `work_dir` configs were specified."
            " The `load_runtime_module` points to a prebuilt runtime module with"
            " an internal work directory so the `work_dir` config will be ignored"
        )

    # If load_runtime_module is not set, we will build the PyXIR runtime module from scratch
    if load_runtime_module == "":
        # Convert Relay expression into XGraph and do partitioning inside PyXIR
        builder = CodegenVitisAI(name, ref)
        xgraph = builder.convert_pyxir(target)
        output_relay_ids = builder.get_output_names()
        layers = xgraph.get_layers()

        # Get the output tensor names using XGraph and output Relay ids
        out_tensor_names = []
        for layer in layers:
            if not layer.internal:
                for relay_id in layer.attrs["relay_id"]:
                    if relay_id in output_relay_ids:
                        out_tensor_names.append(layer.name)
                        break
        if not out_tensor_names:
            raise ValueError(
                "During codegeneration the loading of subexpression \
                             failed due to output tensor name mismatch in Relay PyXIR interface."
            )
        xgraph.meta_attrs["tvm_out_tensors"] = out_tensor_names
        xgraph_str = pyxir.get_xgraph_str(xgraph)

        runtime_func = "tvm.vitis_ai_runtime.from_xgraph"
        fcreate = tvm._ffi.get_global_func(runtime_func)
        return fcreate(name, xgraph_str, target, vai_build_dir, vai_work_dir, export_runtime_module)

    runtime_func = "tvm.vitis_ai_runtime.from_rt_mod"
    fcreate = tvm._ffi.get_global_func(runtime_func)
    return fcreate(name, load_runtime_module, export_runtime_module)
