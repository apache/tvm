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
# pylint: disable=wildcard-import,unused-wildcard-import
"""Compile a Relay IR Function to its Android NNAPI equivalence."""
import copy
import tvm
import tvm.relay
from .error import *
from .operation_utils import relay_op
from .export_object import ExportObject


class FunctionToJsonCompiler(tvm.relay.ExprVisitor):
    """Compile a Relay IR Function to an imtermediate JSON format for json2nnapi.

    Parameters
    ----------
    options: dict
        The compiler option dict.
    """

    def __init__(self, options):
        super().__init__()
        self._options = options
        self._export_obj = ExportObject(self._options)

    def __call__(self, func):
        """Compile a Relay IR Function to an imtermediate JSON format for json2nnapi.

        Parameters
        ----------
        func: tvm.relay.Function
            The Relay IR Function to be compiled.

        Returns
        -------
        json: dict
            A Python dict acting as the resulting JSON of the conversion.
        """
        assert isinstance(func, tvm.relay.Function)
        self.visit(func.body)
        self._export_obj.helper.node_to_operand_idxs_map[func] = copy.deepcopy(
            self._export_obj.helper.node_to_operand_idxs_map[func.body]
        )

        # identify Android NNAPI model inputs
        for p in func.params:
            for i in self._export_obj.helper.node_to_operand_idxs_map[
                p
            ]:  # param may be a tuple, which results in multiple indices
                if i not in self._export_obj["inputs"]:
                    self._export_obj["inputs"].append(i)

        # identify Android NNAPI model outputs
        for i in self._export_obj.helper.node_to_operand_idxs_map[
            func
        ]:  # again, the output may be a tuple, which results in multiple indices
            if i not in self._export_obj["outputs"]:
                self._export_obj["outputs"].append(i)
        # for now, let's force the function to return a single value,
        # i.e. denying tuple as return type
        assert len(self._export_obj["outputs"]) == 1

        # set resulting memory for outputs
        for i, op_i in enumerate(self._export_obj["outputs"]):
            op = self._export_obj["operands"][op_i]
            assert "value" not in op
            op["value"] = {
                "type": "memory_ptr",
                "value": "out",  # no real formatting since len(outs) == 1
            }

        return self._export_obj

    @property
    def export_obj(self):
        """The associated ExportObject of this compiler instance."""
        return self._export_obj

    @property
    def options(self):
        """The associated compiler option dict."""
        return self._options

    def visit_function(self, f):
        raise AndroidNNAPICompilerIncompatibleError(
            f"Conversion of tvm.relay.Function not supported"
        )

    def visit_let(self, let):
        raise AndroidNNAPICompilerIncompatibleError(f"Conversion of tvm.relay.Let not supported")

    def visit_call(self, call):
        if isinstance(call.op, tvm.ir.Op):
            op_handler_module = relay_op
            for namespace in call.op.name.split("."):  # lookup the handler dynamically
                op_handler_module = getattr(op_handler_module, namespace, None)
                assert_anc_compatibility(
                    op_handler_module is not None, f"Relay IR Op { call.op } not implemented"
                )
            op_handler_module.handler(self, call)
        else:
            raise AndroidNNAPICompilerIncompatibleError(
                f"Conversion of { call.op.type_key } not supported"
            )

    def visit_var(self, var):
        self._export_obj.add_operand(
            type_idx=self._export_obj.get_type_idx(
                (var.checked_type.shape, var.checked_type.dtype)
            ),
            node=var,
            value={
                "type": "memory_ptr",
                "value": var.name_hint,
            },
        )

    def visit_type(self, typ):
        raise AndroidNNAPICompilerIncompatibleError(f"Conversion of tvm.relay.Type not supported")

    def visit_if(self, i):
        raise AndroidNNAPICompilerIncompatibleError(f"Conversion of tvm.relay.If not supported")

    def visit_tuple(self, tup):
        field_idxs = []
        for f in tup.fields:
            self.visit(f)
            field_idxs += self._export_obj.helper.node_to_operand_idxs_map[f]
        self._export_obj.helper.node_to_operand_idxs_map[tup] = copy.deepcopy(field_idxs)

    def visit_tuple_getitem(self, t):
        self.visit(tgi.tuple_value)
        self._export_obj.helper.node_to_operand_idxs_map[tgi] = [
            self._export_obj.helper.node_to_operand_idxs_map[tgi.tuple_value][tgi.index]
        ]

    def visit_global_var(self, _):
        raise AndroidNNAPICompilerIncompatibleError(
            f"Conversion of tvm.relay.GlobalVar not supported"
        )

    def visit_op(self, _):
        assert False, "Unreachable"

    def visit_constant(self, const):
        assert_anc_compatibility(
            isinstance(const.checked_type, tvm.relay.TensorType),
            f"Unsupported type { const.checked_type.type_key }",
        )
        shape, dtype = const.data.shape, const.data.dtype
        type_idx = self._export_obj.get_type_idx((shape, dtype))

        if shape == ():
            const_idx = self._export_obj.add_scalar_constant(const.data.asnumpy().item(), dtype)
        elif isinstance(shape, tuple):
            assert_anc_compatibility(len(shape) == 1, "Only flat array constants are supported")
            constants = list(map(lambda i: i.item(), const.data.asnumpy()))
            const_idx = self._export_obj.add_array_constant(constants, dtype)
        else:
            assert False, "Unreachable"

        self._export_obj.add_operand(
            type_idx=type_idx,
            value={
                "type": "constant_idx",
                "value": const_idx,
            },
            node=const,
        )

    def visit_ref_create(self, _):
        raise AndroidNNAPICompilerIncompatibleError(
            f"Conversion of Relay IR reference not supported"
        )

    def visit_ref_write(self, _):
        raise AndroidNNAPICompilerIncompatibleError(
            f"Conversion of Relay IR reference not supported"
        )

    def visit_ref_read(self, _):
        raise AndroidNNAPICompilerIncompatibleError(
            f"Conversion of Relay IR reference not supported"
        )

    def visit_constructor(self, _):
        raise AndroidNNAPICompilerIncompatibleError(f"Conversion of Relay IR ADT not supported")

    def visit_match(self, _):
        raise AndroidNNAPICompilerIncompatibleError(f"Conversion of Relay IR ADT not supported")
