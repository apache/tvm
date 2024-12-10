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

# pylint: disable=invalid-name, inconsistent-return-statements, unidiomatic-typecheck
# pylint: disable=import-outside-toplevel
"""PyTorch ExportedProgram of Relax."""
from collections import ChainMap, OrderedDict
from functools import partial
from typing import Callable, Dict, List, Tuple

import torch
import tvm
from tvm import relax

from .base_fx_graph_translator import BaseFXGraphImporter


class ExportedProgramImporter(BaseFXGraphImporter):
    """An importer from ExportedProgram to Relax."""

    from torch import fx

    ########## Unary Ops ##########

    def _hardtanh(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        x = args[0]
        min_val = node.args[1] if len(args) > 1 else node.kwargs("min_val", -1.0)
        max_val = node.args[2] if len(args) > 2 else node.kwargs("max_val", 1.0)
        return self.block_builder.emit(relax.op.clip(x, min_val, max_val))

    ########## Neural Network ##########

    def _batch_norm_legit_no_training(self, node: fx.Node) -> relax.Var:
        import numpy as np

        x = self.env[node.args[0]]
        channel = int(self.shape_of(x)[1])
        dtype = x.struct_info.dtype
        weight = self.env.get(node.args[1], relax.const(np.ones(channel), dtype=dtype))
        bias = self.env.get(node.args[2], relax.const(np.zeros(channel), dtype=dtype))
        running_mean = self.env.get(node.args[3], relax.const(np.zeros(channel), dtype=dtype))
        running_var = self.env.get(node.args[4], relax.const(np.ones(channel), dtype=dtype))
        momentum = node.args[5] if len(node.args) > 5 else node.kwargs.get("momentum", 0.1)
        eps = node.args[6] if len(node.args) > 6 else node.kwargs.get("eps", 1e-05)

        return self.block_builder.emit(
            relax.op.nn.batch_norm(
                x,
                weight,
                bias,
                running_mean,
                running_var,
                axis=1,
                epsilon=eps,
                momentum=momentum,
            )
        )

    def _group_norm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        num_groups = node.args[1]
        gamma = self.env[node.args[2]] if len(node.args) > 2 else None
        beta = self.env[node.args[3]] if len(node.args) > 3 else None
        eps = node.args[4] if len(node.args) > 4 else 1e-05

        dim = len(self.shape_of(x))
        return self.block_builder.emit(
            relax.op.nn.group_norm(
                x,
                gamma,
                beta,
                num_groups=num_groups,
                channel_axis=1,
                axes=list(range(2, dim)),
                epsilon=eps,
            )
        )

    def _upsample_impl(
        self, x: relax.Expr, size, align_corners: bool, scale_factor, method: str
    ) -> relax.Var:
        coord_trans = "align_corners" if align_corners else "half_pixel"

        if size is None:
            shape = self.shape_of(x)
            assert isinstance(shape, relax.ShapeExpr)
            if isinstance(scale_factor, (tuple, list)):
                assert len(scale_factor) == len(shape) - 2
                size = tuple(
                    int(shape[i].value * scale_factor[i - 2]) for i in range(2, len(shape))
                )
            else:
                size = tuple(int(shape[i].value * scale_factor) for i in range(2, len(shape)))

        return self.block_builder.emit(
            relax.op.image.resize2d(
                x, size, layout="NCHW", method=method, coordinate_transformation_mode=coord_trans
            )
        )

    def _upsample_bilinear2d(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        size = node.args[1] if len(node.args) > 1 else node.kwargs.get("size", None)
        align_corners = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("align_corners", True)
        )
        scale_factor = node.args[3] if len(node.args) > 3 else node.kwargs.get("scale_factor", None)
        return self._upsample_impl(x, size, align_corners, scale_factor, "linear")

    def _upsample_nearest2d(self, node: fx.node) -> relax.Var:
        x = self.env[node.args[0]]
        size = node.args[1] if len(node.args) > 1 else node.kwargs.get("size", None)
        align_corners = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("align_corners", True)
        )
        scale_factor = node.args[3] if len(node.args) > 3 else node.kwargs.get("scale_factor", None)
        return self._upsample_impl(x, size, align_corners, scale_factor, "nearest_neighbor")

    ########## Manipulation ##########

    def _select(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1]
        index = relax.const(node.args[2], "int64")
        return self.block_builder.emit(relax.op.take(x, index, dim))

    def _slice(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        axes = [node.args[1]]
        begin = [node.args[2]]
        end = [node.args[3]]
        stride = [node.args[4] if len(node.args) > 4 else 1]
        return self.block_builder.emit(relax.op.strided_slice(x, axes, begin, end, stride))

    ########## Others ##########

    def create_convert_map(
        self,
    ) -> Dict[str, Callable[[fx.Node], relax.Var]]:
        import operator

        return {
            # unary
            "acos.default": self._unary_op(relax.op.acos),
            "acosh.default": self._unary_op(relax.op.acosh),
            "asin.default": self._unary_op(relax.op.asin),
            "asinh.default": self._unary_op(relax.op.asinh),
            "atan.default": self._unary_op(relax.op.atan),
            "atanh.default": self._unary_op(relax.op.atanh),
            "clamp.default": self._clamp,
            "cos.default": self._unary_op(relax.op.cos),
            "cosh.default": self._unary_op(relax.op.cosh),
            "dropout.default": lambda node: self.env[node.args[0]],
            "exp.default": self._unary_op(relax.op.exp),
            "gelu.default": self._gelu,
            "hardsigmoid.default": self._hardsigmoid,
            "hardswish.default": self._hardswish,
            "hardtanh.default": self._hardtanh,
            "leaky_relu.default": self._leakyrelu,
            "log_softmax.int": self._log_softmax,
            "neg.default": self._unary_op(relax.op.negative),
            "relu.default": self._unary_op(relax.op.nn.relu),
            "round.default": self._round,
            "rsqrt.default": self._unary_op(relax.op.rsqrt),
            "sigmoid.default": self._unary_op(relax.op.sigmoid),
            "silu.default": self._unary_op(relax.op.nn.silu),
            "sin.default": self._unary_op(relax.op.sin),
            "sinh.default": self._unary_op(relax.op.sinh),
            "softmax.int": self._softmax,
            "sqrt.default": self._unary_op(relax.op.sqrt),
            "tan.default": self._unary_op(relax.op.tan),
            "tanh.default": self._unary_op(relax.op.tanh),
            "tril.default": self._tril_triu(relax.op.tril),
            "triu.default": self._tril_triu(relax.op.triu),
            # binary
            "add.Tensor": self._binary_op(relax.op.add, operator.add),
            "div.Tensor": self._binary_op(relax.op.divide, operator.truediv),
            "eq.Scalar": self._binary_op(relax.op.equal, operator.eq),
            "eq.Tensor": self._binary_op(relax.op.equal, operator.eq),
            "floor_divide.default": self._binary_op(relax.op.floor_divide, operator.floordiv),
            "lt.Scalar": self._binary_op(relax.op.less, operator.lt),
            "lt.Tensor": self._binary_op(relax.op.less, operator.lt),
            "matmul.default": self._binary_op(
                partial(relax.op.linear_algebra.matmul, out_dtype="float32"), operator.matmul
            ),
            "max.other": self._binary_op(relax.op.maximum, max),
            "mul.Tensor": self._binary_op(relax.op.multiply, operator.mul),
            "pow.Tensor_Scalar": self._binary_op(relax.op.power, operator.pow),
            "pow.Tensor_Tensor": self._binary_op(relax.op.power, operator.pow),
            "sub.Tensor": self._binary_op(relax.op.subtract, operator.sub),
            # neural network
            "_native_batch_norm_legit_no_training.default": self._batch_norm_legit_no_training,
            "adaptive_avg_pool2d.default": self._adaptive_avg_pool2d,
            "addmm.default": self._addmm,
            "avg_pool2d.default": self._avg_pool2d,
            "baddbmm.default": self._baddbmm,
            "bmm.default": self._binary_op(
                partial(relax.op.linear_algebra.matmul, out_dtype="float32"), operator.matmul
            ),
            "conv_transpose1d.default": self._conv_transpose1d,
            "conv_transpose2d.input": self._conv_transpose2d,
            "conv1d.default": self._conv1d,
            "conv2d.default": self._conv2d,
            "conv3d.default": self._conv3d,
            "einsum.default": self._einsum,
            "embedding.default": lambda node: self._embedding_impl(
                self.env[node.args[1]], self.env[node.args[0]]
            ),
            "group_norm.default": self._group_norm,
            "layer_norm.default": self._layer_norm,
            "linear.default": self._linear,
            "max_pool2d.default": self._max_pool2d,
            "scaled_dot_product_attention.default": self._scaled_dot_product_attention,
            "unbind.int": self._unbind,
            "upsample_bilinear2d.vec": self._upsample_bilinear2d,
            "upsample_nearest2d.vec": self._upsample_nearest2d,
            # statistical
            "mean.dim": self._mean,
            "sum.dim_IntList": self._sum,
            # search
            "argmax.default": self._argmax_argmin(relax.op.argmax),
            "argmin.default": self._argmax_argmin(relax.op.argmin),
            # tensor manipulation
            "cat.default": self._cat,
            "concat.default": self._cat,
            "cumsum.default": self._cumsum,
            "expand.default": self._expand,
            "permute.default": self._permute,
            "repeat.default": self._repeat,
            "select.int": self._select,
            "slice.Tensor": self._slice,
            "split.Tensor": self._split,
            "squeeze.default": self._squeeze,
            "squeeze.dim": self._squeeze,
            "tile.default": self._tile,
            "transpose.int": self._transpose,
            "unsqueeze.default": lambda node: self.block_builder.emit(
                relax.op.expand_dims(self.env[node.args[0]], node.args[1])
            ),
            "view.default": self._reshape,
            # tensor creation
            "_to_copy.default": self._to_copy,
            "arange.start": self._arange,
            "clone.default": lambda node: self.env[node.args[0]],
            "empty.memory_format": self._empty,
            "fill.Scalar": self._fill,
            "new_ones.default": self._new_ones,
            # other
            "getitem": self._getitem,
        }

    def create_input_vars(
        self, exported_program: torch.export.ExportedProgram
    ) -> Tuple[Dict[str, relax.Var], Dict[str, relax.Var]]:
        """Create relax input vars."""
        parameters_buffers_constants = OrderedDict()
        user_inputs = OrderedDict()
        for spec in exported_program.graph_signature.input_specs:
            name_hint = spec.arg.name
            if spec.kind is torch.export.graph_signature.InputKind.CONSTANT_TENSOR:
                shape = exported_program.tensor_constants[spec.target].shape
                torch_dtype = exported_program.tensor_constants[spec.target].dtype
            elif spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
                for node in exported_program.graph.find_nodes(op="placeholder", target=spec.target):
                    if node.name == name_hint:
                        shape = node.meta["tensor_meta"].shape
                        torch_dtype = node.meta["tensor_meta"].dtype
                        break
            else:
                # PARAMETER or BUFFER
                shape = exported_program.state_dict[spec.target].shape
                torch_dtype = exported_program.state_dict[spec.target].dtype

            dtype = self._convert_data_type(torch_dtype)
            relax_var = relax.Var(name_hint, relax.TensorStructInfo(shape, dtype))
            if spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
                user_inputs[name_hint] = relax_var
            else:
                parameters_buffers_constants[name_hint] = relax_var

        return parameters_buffers_constants, user_inputs

    def from_exported_program(
        self,
        exported_program: torch.export.ExportedProgram,
        keep_params_as_input: bool,
        unwrap_unit_return_tuple: bool,
        no_bind_return_tuple: bool,
    ) -> tvm.IRModule:
        """Convert a PyTorch ExportedProgram to a Relax program."""
        from torch import fx  # type: ignore

        # Create input variables.
        parameter_buffer_constant_vars, user_input_vars = self.create_input_vars(exported_program)
        inputs_vars = user_input_vars.copy()
        inputs_vars.update(parameter_buffer_constant_vars)

        # Initialize the block builder with a function and a dataflow block.
        self.block_builder = relax.BlockBuilder()
        func_name = "main"
        func_attrs = {"num_input": len(user_input_vars)} if keep_params_as_input else None

        nodes: List[fx.Node] = exported_program.graph.nodes
        with self.block_builder.function(
            name=func_name, params=list(inputs_vars.values()).copy(), attrs=func_attrs
        ):
            output = None
            with self.block_builder.dataflow():
                # Translate the model.
                for node in nodes:
                    if node.op == "placeholder":
                        if "grapharg" in node.meta and node.meta["grapharg"].fake_tensor is None:
                            # Ignore sym input
                            continue

                        self.env[node] = inputs_vars[node.name]
                    elif node.op == "output":
                        args = self.retrieve_args(node)
                        assert len(args) == 1
                        assert isinstance(args[0], (tuple, relax.Tuple))

                        if unwrap_unit_return_tuple and len(args[0]) == 1:
                            output = self.block_builder.emit_output(args[0][0])
                        elif no_bind_return_tuple:
                            output = []
                            for ret in args[0]:
                                output.append(self.block_builder.emit_output(ret))
                        else:
                            output = self.block_builder.emit_output(args[0])
                        break
                    elif node.op == "get_attr":
                        self.env[node] = getattr(exported_program.graph_module, node.target)
                    elif node.op == "call_function":
                        func_name = node.target.__name__
                        assert (
                            func_name in self.convert_map
                        ), f"Unsupported function type {func_name}"
                        self.env[node] = self.convert_map[func_name](node)
                    else:
                        raise ValueError(f"Unsupported op {node.op}")
            assert output is not None
            self.block_builder.emit_func_output(output)

        to_bind_parameters = ChainMap(
            OrderedDict(exported_program.named_buffers()), exported_program.constants
        )
        if not keep_params_as_input:
            to_bind_parameters = to_bind_parameters.new_child(
                OrderedDict(exported_program.named_parameters())
            )

        binding = {}
        for tensor_name, tensor_value in to_bind_parameters.items():
            # find relax var name from graph signature
            for spec in exported_program.graph_signature.input_specs:
                if tensor_name == spec.target:
                    bind_name = spec.arg.name
                    break
            binding[bind_name] = tvm.nd.from_dlpack(tensor_value.detach())

        mod = self.block_builder.get()
        mod = relax.transform.BindParams("main", binding)(mod)

        if keep_params_as_input:
            parameters = dict(exported_program.named_parameters())
            params = [tvm.nd.from_dlpack(p.detach()) for p in parameters.values()]
            mod["main"] = mod["main"].with_attr("params", params)

        return mod


def from_exported_program(
    exported_program: torch.export.ExportedProgram,
    *,
    keep_params_as_input: bool = False,
    unwrap_unit_return_tuple: bool = False,
    no_bind_return_tuple: bool = False,
) -> tvm.IRModule:
    """Convert a PyTorch ExportedProgram to a Relax program

    Parameters
    ----------
    exported_program : torch.export.ExportedProgram
        The PyTorch ExportedProgram to convert.

    keep_params_as_input : bool
        Whether to keep model parameters as input variables.

    unwrap_unit_return_tuple : bool
        A boolean flag indicating if to the return value when it is an unit tuple.
        When the return value is not a unit tuple, no unwrap will take place.

    no_bind_return_tuple : bool
        A boolean flag indicating whether to bind the return tuple as a relax var.
        If the flag is true and the return value is a tuple, it will not bind it to a var.

    Returns
    -------
    output : tvm.IRModule
        The import result IRModule, with the function "main" containing the
        translated logic.

    Examples
    --------
    Users can use the torch.export.export() to extract a torch.export.ExportedProgram
    from a PyTorch model. The following codes show how to convert a PyTorch model to
    a Relax program.

    .. code-block:: python

        # Import the importer.
        import tvm
        from tvm.relax.frontend.torch import from_exported_program
        import torch
        from torch.export import export

        # Define the module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(in_features=10, out_features=7, bias=True)

            def forward(self, input):
                return self.linear(input)

        # Instantiate the model and create the input info dict.
        torch_model = MyModule()

        # Use torch.export.export() to convert the PyTorch model into ExportedProgram.
        example_args = (torch.rand(128, 10, dtype=torch.float32),)
        exported_program = export(torch_model, args=example_args)

        # Use the importer to import the ExportedProgram to Relax.
        mod: tvm.IRModule = from_exported_program(exported_program)
    """
    # decompose into Core ATen operators
    exported_program.run_decompositions()

    return ExportedProgramImporter().from_exported_program(
        exported_program,
        keep_params_as_input,
        unwrap_unit_return_tuple,
        no_bind_return_tuple,
    )
