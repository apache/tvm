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
# pylint: disable=invalid-name, unused-argument, no-else-return, E1102
"""Vitis-AI codegen annotation of supported operators"""

import warnings
import numpy as np

from tvm import relay
import tvm._ffi
from tvm.relay import transform
from tvm.relay.expr import Tuple, TupleGetItem
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.annotation import compiler_begin, compiler_end

# Placeholder for PyXIR module
pyxir = None


def enabled():
    """Return whether Vitis-AI support is available"""
    if not tvm.get_global_func("relay.ext.vitis_ai.available", True):
        print("Skip because Vitis-AI codegen is not available.")
        return False
    return True


@transform.function_pass(opt_level=0)
class VitisAIAnnotationPass:
    """Responsible for annotating Relay expressions for Vitis-AI DPU accelerators

    Parameters
    ----------
    compiler : str
        The compiler name used for annotations (`vitis_ai`).
    dpu_target : str
        The Vitis AI DPU target identifier.
    params : dict
        A dictionary containing the module's parameters.
    """

    def __init__(self, compiler, dpu_target, params):
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

        self.compiler = compiler
        self.dpu_target = dpu_target
        self.params = params

    def transform_function(self, func, mod, ctx):
        """Transform function for annotating Relay module"""
        annotator = self

        class Annotator(tvm.relay.ExprMutator):
            """Annotator for Vitis-AI DPU accelerators"""

            def visit_tuple(self, tup):
                """Add compiler_begin and compiler_end annotations to Tuple"""
                field_list = []
                cond = int(hash(tup))
                for field in tup.fields:
                    if cond in annotator.relay_ids:
                        field_list.append(compiler_begin(super().visit(field), annotator.compiler))
                    else:
                        field_list.append(super().visit(field))
                if cond in annotator.relay_ids:
                    return compiler_end(Tuple(field_list), annotator.compiler)
                else:
                    return Tuple(field_list)

            def visit_tuple_getitem(self, op):
                """Add compiler_begin and compiler_end annotations to TupleGetItem"""
                if int(hash(op.tuple_value)) in annotator.relay_ids:
                    tuple_value = compiler_begin(super().visit(op.tuple_value), annotator.compiler)
                    return compiler_end(TupleGetItem(tuple_value, op.index), annotator.compiler)
                else:
                    tuple_value = super().visit(op.tuple_value)
                    return TupleGetItem(tuple_value, op.index)

            def visit_call(self, call):
                """Add compiler_begin and compiler_end annotations to the Call expr"""
                if int(hash(call)) in annotator.relay_ids:
                    new_args = []
                    for arg in call.args:
                        ann = compiler_begin(super().visit(arg), annotator.compiler)
                        new_args.append(ann)
                    new_call = relay.Call(call.op, new_args, call.attrs, call.type_args)
                    return compiler_end(new_call, annotator.compiler)

                else:
                    return super().visit_call(call)

        xgraph = pyxir.frontend.tvm.from_relay(mod, self.params, postprocessing=None)
        xgraph = pyxir.partition(xgraph, targets=[self.dpu_target])

        layers = xgraph.get_layers()
        relay_ids = [
            list(np.array(layer.attrs["relay_id"]).flatten())
            for layer in layers
            if layer.target == self.dpu_target
        ]
        self.relay_ids = [item for sublist in relay_ids for item in sublist]

        return Annotator().visit(func)


def annotation(mod, params, target):
    """DEPRECATED

    Annotate Relay expression for offloading operators to Vitis AI DPU accelerators
    NOTE: This function does the same as the next one (`partition_for_vitis_ai`) but is
    still here for backward compatibility"""
    # We need type information for supporting models that contain operations that don't
    #   have a Relay to XLayer translation
    warnings.warn(
        "tvm.relay.op.contrib.vitis_ai.annotation() is being deprecated."
        " Please use tvm.relay.op.contrib.vitis_ai.partition_for_vitis_ai() instead. "
        " Check out https://tvm.apache.org/docs/deploy/vitis_ai.html for documentation. "
    )
    mod = relay.transform.InferType()(mod)
    mod = VitisAIAnnotationPass("vitis_ai", target, params)(mod)
    return mod


def partition_for_vitis_ai(mod, params=None, dpu=None, **opts):
    """Partition the Relay expression for offloading operators to Vitis AI DPU

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    dpu : str
        The DPU identifier (e.g. DPUCZDX8G-zcu104, DPUCADF8H)

    Returns
    -------
    ret : Module
    """

    if dpu is None:
        raise ValueError("Please pass Vitis AI DPU identifier to the partitioning function")

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    desired_layouts_in_partition = {
        "nn.conv2d": ["NHWC", "default"],
        "nn.upsampling": ["NHWC"],
        "image.resize2d": ["NHWC"],
    }
    desired_layouts_in_main = {
        "nn.conv2d": ["NCHW", "default"],
        "nn.upsampling": ["NCHW"],
        "image.resize2d": ["NCHW"],
    }
    seq = tvm.transform.Sequential(
        [
            transform.RemoveUnusedFunctions(),
            transform.ConvertLayout(desired_layouts_in_partition),
            transform.FoldConstant(),
            transform.InferType(),
            VitisAIAnnotationPass("vitis_ai", dpu, params),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            transform.RemoveUnusedFunctions(),
            transform.ConvertLayout(desired_layouts_in_main),
            transform.FoldConstant(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        return seq(mod)
