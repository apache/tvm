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
"""Utility to compile Vitis-AI models"""

import os

from tvm.relay.expr import Tuple, Call, TupleGetItem
import tvm._ffi

import pyxir
import pyxir.frontend.tvm


class CodegenVitisAI:
    """
    Traverse Relay expression and convert into PyXIR XGraph format
    """
    def __init__(self, model_name, function):
        self.model_name = model_name
        self.function = function
        self.params = {}

    def convert_pyxir(self, target):
        """
        Convert Relay expression to PyXIR XGraph
        """
        xgraph = pyxir.frontend.tvm.from_relay(self.function,
                                               params=self.params, postprocessing=None)
        xgraph = pyxir.partition(xgraph, targets=[target])
        return xgraph

    def get_output_names(self):
        """
        Get output names from Relay expression
        """
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
    """
    Create a Vitis-AI runtime from the provided Relay expression
    """
    assert isinstance(ref, tvm.relay.function.Function)

    model_dir = os.getcwd()
    out_tensor_names = []
    name = str(ref.attrs.global_symbol)

    pass_context = tvm.get_global_func("transform.GetCurrentPassContext")()
    target = str(pass_context.config['relay.ext.vitis_ai.options.target'])
    vai_build_dir = str(pass_context.config['relay.ext.vitis_ai.options.build_dir']) \
        if 'relay.ext.vitis_ai.options.build_dir' in pass_context.config else None
    if vai_build_dir and not os.path.exists(vai_build_dir):
        raise ValueError("Provided Vitis-AI build dir: `{}` could not be found"
                         .format(vai_build_dir))

    # If build directory is not passed as a parameter in transform.PassContext,
    # we will build the Vitis-AI PyXIR runtime from scratch
    if not vai_build_dir:
        # Convert Relay expression into XGraph and do partitioning inside PyXIR
        builder = CodegenVitisAI(name, ref)
        model_dir = target + "_build/"
        xgraph = builder.convert_pyxir(target)
        output_relay_ids = builder.get_output_names()
        layers = xgraph.get_layers()

        # Get the output tensor names using XGraph and output Relay ids
        out_tensor_names = []
        for layer in layers:
            if not layer.internal:
                if layer.attrs['relay_id'][0] in output_relay_ids:
                    out_tensor_names.append(layer.name)
        if len(out_tensor_names) == 0:
            raise ValueError("During codegeneration the loading of subexpression \
                             failed due to output tensor name mismatch in Relay PyXIR interface.")

        # Save/serialize XGraph
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        xgraph.meta_attrs['tvm_out_tensors'] = out_tensor_names
        pyxir.graph.io.xgraph_io.XGraphIO.save(xgraph, model_dir + 'dpu_xgraph')
    else:
        model_dir = vai_build_dir

    # Create Vitis-AI runtime module
    runtime_func = "tvm.vitis_ai_runtime.create"
    fcreate = tvm._ffi.get_global_func(runtime_func)
    return fcreate(name, model_dir, target)

