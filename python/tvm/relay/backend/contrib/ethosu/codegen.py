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
"""Codegen for Arm(R) Ethos(TM)-U NPU"""

import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu.tir.compiler import lower_to_tir
from tvm.relay.backend.contrib.ethosu.tir.scheduler import copy_constants
from tvm.relay.backend.contrib.ethosu.legalize import LegalizeEthosU
from tvm.relay.backend.contrib.ethosu import tir_to_cs_translator
from tvm.relay.backend.contrib.ethosu import util
from tvm.relay.expr_functor import ExprMutator
from tvm.ir.transform import Pass

# pylint: disable=unused-import
from tvm.relay.backend.contrib.ethosu.op import op_attrs
from tvm.relay.backend.contrib.ethosu import op


class OptimizeLUTs(ExprMutator):
    """A pass to merge an identity operator with a LUT based activation function with
    a preceding operator provided that operator can do a table lookup for the activation
    in the hardware"""

    def __init__(self):
        super().__init__()
        self.lut_ops = {
            "contrib.ethosu.conv2d": op.ethosu_conv2d,
            "contrib.ethosu.depthwise_conv2d": op.ethosu_depthwise_conv2d,
            "contrib.ethosu.pooling": op.ethosu_pooling,
        }

    def create_op_with_lut(self, call):
        """Extract the parameters and attributes from the NPU operator and create
        a new operator with LUT.

        Parameters
        ----------
        call : tvm.relay.expr.Call
            The current call node being visited.

        Returns
        -------
        tvm.relay.expr.Call
            The new operator with LUT.
        """
        identity = call
        ethosu_op = call.args[0]
        lut = identity.args[1]
        activation = identity.attrs.activation

        new_attrs = dict(ethosu_op.attrs)
        new_attrs["activation"] = activation

        # Assume that LUT is always the last argument
        new_args = ethosu_op.args[:-1] + [lut]
        assert ethosu_op.op.name in self.lut_ops.keys()

        return self.lut_ops[ethosu_op.op.name](*new_args, **new_attrs)

    def visit_call(self, call: tvm.relay.expr.Call) -> tvm.relay.expr.Call:
        """Recursively visit call nodes in the input graph and if an ethosu.identity
        operator with LUT is found and the preceding operator has a LUT attribute, create
        a new NPU operator.

        Parameters
        ----------
        call : tvm.relay.expr.Call
            The current call node being visited.

        Returns
        -------
        tvm.relay.expr.Call
            The input call node in the case the current call node does
            not refer to an Op. Else, a new call node with a new operator.
        """
        new_call = call
        lut_activations = ["TANH", "LUT", "SIGMOID"]

        if isinstance(call.op, tvm.ir.Op) and isinstance(call.args[0], tvm.relay.expr.Call):
            producer_op = call.args[0]
            # Check if the producer can do a LUT operation
            if (
                producer_op.op.name in self.lut_ops.keys()
                and call.op.name == "contrib.ethosu.identity"
                and call.attrs.activation in lut_activations
            ):
                # Check the producer doesn't already have a LUT
                has_lut = producer_op.attrs.activation in lut_activations
                if not has_lut:
                    new_call = self.create_op_with_lut(call)

        new_call = super().visit_call(new_call)

        return new_call


@relay.transform.function_pass(opt_level=1, name="LUTsOptimizer")
class LUTsOptimizer(Pass):
    """Register LUTsOptimizer as a relay pass."""

    def transform_function(
        self, func: tvm.relay.function.Function, mod: tvm.IRModule, _
    ) -> tvm.IRModule:
        """Visit relay nodes in the given module.

        Parameters
        ----------
        func : tvm.relay.function.Function
            The function to apply the optimization pass for multiple LUTs to.
        mod : tvm.IRModule
            The module to apply the optimization pass for multiple LUTs to.

        Returns
        -------
        mod : tvm.IRModule
            New module with optimized LUTs.
        """
        assert len(mod.functions.items()) == 1, "Module can only contain one function."
        return OptimizeLUTs().visit(func)


class LayoutOptimization(ExprMutator):
    """A pass to optimize the layout of NPU operations. If both the
    producer and consumer of a tensor are NPU operators, then the
    layout is converted from NHWC to NHCWB16.

    Attributes
    ----------
    children : Dict[tvm.relay.expr.Call, List[tvm.relay.expr.Call]]
        A map from current call to a list of calls that rely on the current
        call. This allows the graph to be traversed backwards, which is useful
        for checking whether the output layouts can be rewritten.
    optimize_op : Dict[str, Callable]
        A map from NPU op name to function that creates NPU op.
    """

    def __init__(self):
        self.children = {}
        self.optimize_op = {
            "contrib.ethosu.conv2d": op.ethosu_conv2d,
            "contrib.ethosu.depthwise_conv2d": op.ethosu_depthwise_conv2d,
            "contrib.ethosu.pooling": op.ethosu_pooling,
            "contrib.ethosu.binary_elementwise": op.ethosu_binary_elementwise,
            "contrib.ethosu.unary_elementwise": op.ethosu_unary_elementwise,
        }

        super().__init__()

    def alter_ethosu_op_layout(self, call: tvm.relay.expr.Call) -> tvm.relay.expr.Call:
        """Alter the input and output layouts of an NPU operation if needed.
        Input layout is only altered if the producing operation is an NPU
        operation. Likewise, the output layout is only altered if the consuming
        operation is an NPU operation.

        Parameters
        ----------
        call : tvm.relay.expr.Call
            The call pointing to an NPU operation that will be checked if
            the layout needs altering.

        Returns
        -------
        new_call : tvm.relay.expr.Call
            New call with altered layouts.
        """
        assert isinstance(call.attrs, tvm.ir.Attrs), (
            f"The attributes for operator '{call.op.name}' could not be "
            "found. Did you register the relay.attrs.Ethosu<opname>Attrs "
            "object in python api?"
        )

        new_attrs = dict(call.attrs)
        parents = []

        # Check if we can rewrite the input layouts
        input_count = 0
        for arg in call.args:
            input_count += 1
            if not isinstance(arg, tvm.relay.expr.Call):
                continue
            if isinstance(arg.op, tvm.ir.op.Op) and arg.op.name in self.optimize_op:
                layout_string = "ifm_layout" if input_count <= 1 else f"ifm{input_count}_layout"
                new_attrs[layout_string] = "NHCWB16"
            parents.append(arg)

        # Check if we can rewrite the output layouts
        if call in self.children:
            children = self.children[call]
            if all(
                isinstance(child, tvm.relay.expr.Call)
                and isinstance(child.op, tvm.ir.op.Op)
                and child.op.name in self.optimize_op
                and child.attrs["ifm_layout"] == "NHCWB16"
                for child in children
            ):
                new_attrs["ofm_layout"] = "NHCWB16"

        name = call.op.name
        assert name in self.optimize_op, (
            f"Could not create operator '{name}' as the creation function "
            "is unknown. Please provide a mapping."
        )
        new_call = self.optimize_op[name](*call.args, **new_attrs)

        # Update map of children
        for input_arg in parents:
            if input_arg in self.children:
                self.children[input_arg].append(new_call)
            else:
                self.children[input_arg] = [new_call]

        return super().visit_call(new_call)

    def visit_call(self, call: tvm.relay.expr.Call) -> tvm.relay.expr.Call:
        """Recursively visit call nodes in the input graph and alter the
        layout of an op if needed.

        Parameters
        ----------
        call : tvm.relay.expr.Call
            The current call node being visited.

        Returns
        -------
        tvm.relay.expr.Call
            The input call node in the case the current call node does
            not refer to an Op. Else, a new call node with altered Op
            attributes.
        """
        if isinstance(call.op, tvm.ir.op.Op) and call.op.name in self.optimize_op:
            return self.alter_ethosu_op_layout(call)
        return super().visit_call(call)


@relay.transform.function_pass(opt_level=1, name="LayoutOptimizer")
class LayoutOptimizer(Pass):
    """Register LayoutOptimizer as a Relay pass."""

    def transform_function(
        self, func: tvm.relay.function.Function, mod: tvm.IRModule, _
    ) -> tvm.IRModule:
        """A pass to optimize the layout of NPU operations. If both the
        producer and consumer of a tensor are NPU operators, then the
        layout is converted from NHWC to NHCWB16 as this is the layout NPU
        uses internally."""
        assert len(mod.functions.items()) == 1, "Module can only contain one function."
        return LayoutOptimization().visit(func)


@tvm._ffi.register_func("relay.ext.ethos-u.constant_updater")
def constant_updater(expr, symbol):  # pylint: disable=unused-argument
    """
    The constant updater process happen after lowering in the core compiler.
    For the NPU, we dont want the build process to extract constants to be loaded in
    the runtime as we are embedding them inside the C runtime.Module.
    """
    return dict()


@tvm._ffi.register_func("relay.ext.ethos-u.relay_to_tir_func")
def relay_to_tir_func(ext_func: relay.Function) -> tvm.tir.PrimFunc:
    """
    This is the hook for python-based lowering of relay function
    that gets offloaded to the microNPU.

    Parameters
    ----------
    ext_func : relay.Function
        This is the partitioned relay function

    Returns
    -------
    primfunc : tir.PrimFunc
        This returns the scheduled PrimFunc
    """
    assert len(ext_func.params) == 1
    input_size = util.calculate_size_bytes(ext_func.params[0])
    output_size = util.calculate_size_bytes(ext_func.body)
    mod = tvm.IRModule()
    mod["main"] = ext_func
    mod = LegalizeEthosU()(mod)
    mod = LUTsOptimizer()(mod)
    mod = relay.transform.InferType()(mod)
    # We are currently using copy_constants scheduler In the long run,
    # this should be a single intelligent and a composite scheduler
    # that can perform scheduling based on user inputs such as
    # scratch memory size.
    tir_mod, const_dict = lower_to_tir(mod["main"], copy_constants())

    for idx in const_dict.keys():
        const_dict[idx] = tvm.nd.array(const_dict[idx])

    primfunc = tir_mod["main"]
    primfunc = primfunc.with_attr("global_symbol", ext_func.attrs["global_symbol"])
    primfunc = primfunc.with_attr("ethos-u.constants", const_dict)
    primfunc = primfunc.with_attr("ethos-u.input_size", input_size)
    primfunc = primfunc.with_attr("ethos-u.output_size", output_size)
    return primfunc


@tvm._ffi.register_func("relay.ext.ethos-u.primfunc_to_artifact")
def primfunc_to_artifact(primfunc: tvm.tir.PrimFunc) -> util.CompilationArtifact:
    """
    This is the hook for python-based lowering of TIR PrimFunc
    that has undergone unified optimization to compilation
    artifact destined for the microNPU.

    Parameters
    ----------
    primfunc : tir.PrimFunc
        TIR PrimFunc that has undergone unified optimizations

    Returns
    -------
    CompilationArtifact
        This is a structure that holds the binary artifacts
        for the microNPU
    """
    symbol = str(primfunc.attrs["global_symbol"])
    const_dict = primfunc.attrs["ethos-u.constants"]
    input_size = primfunc.attrs["ethos-u.input_size"]
    output_size = primfunc.attrs["ethos-u.output_size"]
    tir_mod = tvm.IRModule()
    tir_mod[symbol] = primfunc

    const_dict_with_int_keys = dict()
    for idx in const_dict.keys():
        const_dict_with_int_keys[int(idx)] = const_dict[idx].numpy()

    cmms, encoded_constants, scratch_size = tir_to_cs_translator.translate(
        tir_mod, const_dict_with_int_keys
    )
    return util.CompilationArtifact(
        cmms, encoded_constants, scratch_size, input_size, output_size, symbol
    )
