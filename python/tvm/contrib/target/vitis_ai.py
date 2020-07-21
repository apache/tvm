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
"""Utility to compile VITISAI models"""

import os

from tvm.relay.expr import Tuple, Call
import tvm._ffi

import pyxir
import pyxir.frontend.tvm

from .. import vitis_ai_runtime

class CodegenVitisAI:
    """
    Traverse subgraphs and build XGraph
    """
    def __init__(self, model_name, function):

        self.model_name = model_name
        self.function = function
        self.params = {}



    def convert_pyxir(self, target):
        """
         Convert relay submodule expression to PYXIR(XGRAPH)
        """
        xgraph = pyxir.frontend.tvm.from_relay(self.function,
                                               params=self.params, postprocessing=None)
        xgraph = pyxir.partition(xgraph, targets=[target])
        return xgraph

    def get_output_names(self):
        """
        Get output names from subgraph
        """
        func = self.function
        output_relay_ids = []
        expr = func.body
        if isinstance(expr, Tuple):
            for field in expr.fields:
                output_relay_ids.append(hash(field))
        elif isinstance(expr, Call):
            output_relay_ids.append(hash(expr))
        else:
            raise ValueError("does not support {}".format(type(expr)))
        return output_relay_ids

@tvm._ffi.register_func("relay.ext.vai")
def vai_compiler(ref):
    """
    Create a VAI runtime from a Relay module.
    """
    assert isinstance(ref, tvm.relay.function.Function)

    model_dir = os.getcwd()
    out_tensor_names = []
    name = str(ref.attrs.global_symbol)

    pass_context = tvm.get_global_func("transform.GetCurrentPassContext")()
    target = str(pass_context.config['target_'])
    vai_build_dir = str(pass_context.config['vai_build_dir_']) \
        if 'vai_build_dir_' in pass_context.config else None
    if vai_build_dir and not os.path.exists(vai_build_dir):
        raise ValueError("Provided Vitis-AI build dir: `{}` could not be found"
                         .format(vai_build_dir))
    if not vai_build_dir:
        builder = CodegenVitisAI(name, ref)
        model_dir = target + "_build/"
        xgraph = builder.convert_pyxir(target)
        output_relay_ids = builder.get_output_names()
        layers = xgraph.get_layers()
        # get the output tensor names using xgraph and output relay ids
        out_tensor_names = []
        for layer in layers:
            if not layer.internal:
                if layer.attrs['relay_id'][0] in output_relay_ids:
                    out_tensor_names.append(layer.name)
        if len(out_tensor_names) == 0:
            raise ValueError("During codegeneration the loading of subexpression \
                             failed due to output tensorname mismatch in relay pyxir interface.")

        # Save/serialize XGraph
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        xgraph.meta_attrs['tvm_out_tensors'] = out_tensor_names
        pyxir.graph.io.xgraph_io.XGraphIO.save(xgraph, model_dir + 'dpu_xgraph')
    else:
        model_dir = vai_build_dir

    return vitis_ai_runtime.create(name, model_dir, target).module
