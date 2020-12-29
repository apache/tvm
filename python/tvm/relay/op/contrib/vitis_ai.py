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

import numpy as np

import pyxir
import pyxir.frontend.tvm

from tvm import relay
import tvm._ffi
from tvm.relay.expr import Tuple, TupleGetItem
from tvm.relay import transform
from tvm.relay.op.annotation import compiler_begin, compiler_end


@transform.function_pass(opt_level=0)
class VitisAIAnnotationPass:
    """Responsible for annotating Relay expressions for Vitis-AI DPU accelerators"""

    def __init__(self, compiler, relay_ids):
        self.compiler = compiler
        self.relay_ids = relay_ids

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

        return Annotator().visit(func)


def annotation(mod, params, target):
    """Annotate Relay expression for Vitis-AI DPU accelerators"""
    xgraph = pyxir.frontend.tvm.from_relay(mod, params, postprocessing=None)
    xgraph = pyxir.partition(xgraph, targets=[target])

    layers = xgraph.get_layers()
    relay_ids = [
        list(np.array(layer.attrs["relay_id"]).flatten())
        for layer in layers
        if layer.target == target
    ]
    relay_ids_flatten = [item for sublist in relay_ids for item in sublist]
    mod = VitisAIAnnotationPass("vitis_ai", relay_ids_flatten)(mod)

    return mod
