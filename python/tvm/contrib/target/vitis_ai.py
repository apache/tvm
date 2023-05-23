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
import importlib

from tvm.relay.expr import Tuple, Call, TupleGetItem
import tvm._ffi

# Placeholder for PyXIR module
pyxir = None


def vitis_ai_available():
    """Return whether Vitis AI tools are available"""
    pyxir_spec = importlib.util.find_spec("pyxir")
    if not tvm.get_global_func("tvm.vitis_ai_runtime.from_xgraph", True) or pyxir_spec is None:
        return False
    return True


class CodegenVitisAI:

    """Traverse Relay expression and convert into PyXIR XGraph format

    Parameters
    ----------
    function : Function
        The Relay function
    dpu_target : str
        The Vitis AI DPU target identifier
    """

    def __init__(self, function, dpu_target):
        global pyxir
        try:
            if pyxir is None:
                pyxir = __import__("pyxir")
                __import__("pyxir.frontend.tvm")
        except ImportError:
            # add "from None" to silence
            # "During handling of the above exception, another exception occurred"
            raise ImportError(
                "The pyxir package is required for the Vitis AI backend. "
                "Please install it first. "
                "Help: (https://tvm.apache.org/docs/deploy/vitis_ai.html) "
            ) from None

        self.function = function
        self.dpu_target = dpu_target
        self.params = {}

    def build(self):
        """ "Convert the Relay expression to a PyXIR XGraph to instantiate
        the Vitis AI runtime

        Returns
        -------
        xgraph_str : str
            Serialized XGraph
        """
        xgraph = pyxir.frontend.tvm.from_relay(
            self.function, params=self.params, postprocessing=None
        )
        xgraph = pyxir.partition(xgraph, targets=[self.dpu_target])
        output_relay_ids = self.get_output_names()
        layers = xgraph.get_layers()

        # Get the output tensor names using XGraph and output Relay ids
        out_tensor_names = ["unknown_name"] * len(output_relay_ids)
        for layer in layers:
            if not layer.internal:
                for relay_id in layer.attrs["relay_id"]:
                    if relay_id in output_relay_ids:
                        out_tensor_names[output_relay_ids.index(relay_id)] = layer.name
                        break
        if any([name == "unkown_name" for name in out_tensor_names]):
            raise ValueError(
                "During codegeneration the loading of subexpression"
                " failed due to output tensor name mismatch in Relay PyXIR interface."
            )
        xgraph.meta_attrs["tvm_out_tensors"] = out_tensor_names
        xgraph_str = pyxir.get_xgraph_str(xgraph)
        return xgraph_str

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
            raise ValueError(f"Vitis-AI codegen does not support {type(expr)} as output")
        return output_relay_ids


@tvm._ffi.register_func("relay.ext.vitis_ai")
def vitis_ai_compiler(ref):
    """Create a Vitis-AI runtime from the provided Relay expression"""
    assert isinstance(ref, tvm.relay.function.Function)

    name = str(ref.attrs.global_symbol)

    pass_context = tvm.get_global_func("transform.GetCurrentPassContext")()

    cfg = (
        pass_context.config["relay.ext.vitis_ai.options"]
        if "relay.ext.vitis_ai.options" in pass_context.config
        else None
    )

    # Backward compatibility with old pass context configs
    if cfg is None:
        warnings.warn(
            "You are using a deprecated way of passing build configs (e.g."
            " `relay.ext.vitis_ai.options.target`). Check out the Vitis AI "
            " documentation here: https://tvm.apache.org/docs/deploy/vitis_ai.html"
            " to switch to recommended way for passing build configs."
        )

        # The target Vitis-AI accelerator device
        dpu_target = (
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

        # (Optional configs) Export and load PyXIR runtime module to file if provided. This is
        #   used to compile and quantize a model on the host and deploy it at the edge
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
    else:
        dpu_target = cfg.dpu if cfg.dpu else None
        # (Optional configs) The build and work directories to be used by Vitis AI
        vai_build_dir = cfg.build_dir if cfg.build_dir else tvm.contrib.utils.tempdir().relpath("")

        # (Optional configs) Export and load PyXIR runtime module to file if provided. This is
        #   used to compile and quantize a model on the host and deploy it at the edge
        vai_work_dir = cfg.work_dir if cfg.work_dir else tvm.contrib.utils.tempdir().relpath("")
        export_runtime_module = cfg.export_runtime_module
        load_runtime_module = cfg.load_runtime_module

    # Config checks
    if load_runtime_module and dpu_target is not None:
        warnings.warn(
            "Both `load_runtime_module` and `dpu` configs were specified."
            " The `load_runtime_module` points to a prebuilt runtime module with"
            " an internal DPU target so the `dpu` config will be ignored"
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
        codegen = CodegenVitisAI(ref, dpu_target)
        xgraph_str = codegen.build()

        runtime_func = "tvm.vitis_ai_runtime.from_xgraph"
        fcreate = tvm._ffi.get_global_func(runtime_func)
        return fcreate(
            name, xgraph_str, dpu_target, vai_build_dir, vai_work_dir, export_runtime_module
        )

    runtime_func = "tvm.vitis_ai_runtime.from_rt_mod"
    fcreate = tvm._ffi.get_global_func(runtime_func)
    return fcreate(name, load_runtime_module, export_runtime_module)
