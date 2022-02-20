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
from collections import defaultdict

import tvm
from tvm import relay
from tvm import ir
from tvm.relay.backend.contrib.ethosu.tir.compiler import lower_to_tir
from tvm.relay.backend.contrib.ethosu.tir.scheduler import copy_constants
from tvm.relay.backend.contrib.ethosu.legalize import LegalizeEthosU
from tvm.relay.backend.contrib.ethosu import tir_to_cs_translator
from tvm.relay.backend.contrib.ethosu import util
from tvm.relay.expr_functor import ExprMutator, ExprVisitor

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


@ir.transform.module_pass(opt_level=1, name="LUTsOptimizer")
class LUTsOptimizer:
    """Register LUTsOptimizer as a relay pass."""

    def transform_module(self, mod: tvm.ir.IRModule, _) -> tvm.IRModule:
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
        global_var, func = mod.functions.items()[0]
        optimized_func = OptimizeLUTs().visit(func)
        mod.update_func(global_var, optimized_func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


class AnalyzeConsumers(ExprVisitor):
    """Traverses the graph to determine consumers that are NPU operations. The
    result is maintained in `npu_consumers`.

    Attributes
    ----------
    npu_consumers : Dict[tvm.relay.expr.Call, List[bool]]
        Mapping from NPU operation to list of boolean values that represent
        whether or not each consumer is an NPU operation.
    optimize_ops : Dict[str, Callable]
        A map from NPU operation name to function that creates NPU operation.
    """

    def __init__(self, optimize_ops):
        self.npu_consumers = defaultdict(list)
        self.optimize_ops = optimize_ops
        super().__init__()

    def visit_call(self, call: relay.Call):
        is_npu_consumer = call.op.name in self.optimize_ops
        args = []

        # Expand tuples
        for arg in call.args:
            if isinstance(arg, relay.Tuple):
                args.extend(arg.fields)
            else:
                args.append(arg)

        for arg in args:
            if isinstance(arg, relay.Call) and arg.op.name in self.optimize_ops:
                self.npu_consumers[arg].append(is_npu_consumer)

        super().visit_call(call)


class LayoutOptimization(ExprMutator):
    """A pass to optimize the layout of NPU operations by converting to brick format (NHCWB16).
    This pass traverses the graph and attempts to alter the input/output layouts when an NPU
    operation is visited. Whether or not the input/output layout can be altered for a given NPU
    operation depends on the following:

    Check alter input layout: For each argument, if the producer is also an NPU operation and
        its output is altered to brick format, then the input layout with respect to the current
        argument is altered to brick format.

    Check alter output layout: If all consumers (child nodes) are an NPU operation, then the
        output layout is altered to brick format.

    Note
    ----
    In order for this pass to be run, the consumers of each NPU operation must first be analyzed
    by the `AnalyzeConsumers` pass, since Relay doesn't keep a reference to child nodes.

    Attributes
    ----------
    npu_consumers : Dict[tvm.relay.expr.Call, bool]
        A map from current call to a list boolean values that state whether or not each consumer
        is an NPU operation.
    optimize_ops : Dict[str, Callable]
        A map from NPU operation name to function that creates NPU operation.
    """

    def __init__(self, npu_consumers, optimize_ops):
        self.npu_consumers = npu_consumers
        self.optimize_ops = optimize_ops
        super().__init__()

    def alter_ethosu_op_layout(self, call: tvm.relay.expr.Call) -> tvm.relay.expr.Call:
        """Alter the layouts of given NPU operation to brick format if possible.

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

        # Check if we can rewrite the input layouts
        input_count = 0
        for arg in call.args:
            input_count += 1
            if arg not in self.npu_consumers:
                continue
            consumers = self.npu_consumers[arg]
            parent_has_brick_output = consumers and all(consumers)
            if parent_has_brick_output:
                layout_string = "ifm_layout" if input_count <= 1 else f"ifm{input_count}_layout"
                new_attrs[layout_string] = "NHCWB16"

        # Check if we can rewrite the output layouts
        consumers = self.npu_consumers[call]
        if consumers and all(consumers):
            new_attrs["ofm_layout"] = "NHCWB16"

        name = call.op.name
        return self.optimize_ops[name](*call.args, **new_attrs)

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
        if isinstance(call.op, tvm.ir.Op) and call.op.name in self.optimize_ops:
            call = self.alter_ethosu_op_layout(call)
        return super().visit_call(call)


@ir.transform.module_pass(opt_level=1, name="LayoutOptimizer")
class LayoutOptimizer:
    """Register LayoutOptimizer as a Relay pass."""

    OPTIMIZE_OPS = {
        "contrib.ethosu.conv2d": op.ethosu_conv2d,
        "contrib.ethosu.depthwise_conv2d": op.ethosu_depthwise_conv2d,
        "contrib.ethosu.pooling": op.ethosu_pooling,
        "contrib.ethosu.binary_elementwise": op.ethosu_binary_elementwise,
        "contrib.ethosu.unary_elementwise": op.ethosu_unary_elementwise,
    }

    def transform_module(self, mod: tvm.ir.IRModule, _) -> tvm.IRModule:
        """A pass to optimize the layout of NPU operations. If both the
        producer and consumer of a tensor are NPU operators, then the
        layout is converted from NHWC to NHCWB16 as this is the layout NPU
        uses internally."""
        assert len(mod.functions.items()) == 1, "Module can only contain one function."
        global_var, func = mod.functions.items()[0]
        analyze = AnalyzeConsumers(self.OPTIMIZE_OPS)
        analyze.visit(func)
        optimized_func = LayoutOptimization(analyze.npu_consumers, self.OPTIMIZE_OPS).visit(func)
        mod.update_func(global_var, optimized_func)
        return mod

    def __call__(self, *args, **kwargs):
        pass


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
    mod = tvm.IRModule()
    mod["main"] = ext_func
    mod = LegalizeEthosU()(mod)
    mod = LUTsOptimizer()(mod)
    mod = LayoutOptimizer()(mod)
    mod = relay.transform.InferType()(mod)
    # We are currently using copy_constants scheduler In the long run,
    # this should be a single intelligent and a composite scheduler
    # that can perform scheduling based on user inputs such as
    # scratch memory size.
    tir_mod, const_dict = lower_to_tir(mod["main"], copy_constants())

    for param in const_dict.keys():
        const_dict[param] = tvm.nd.array(const_dict[param])

    primfunc = tir_mod["main"]
    primfunc = primfunc.with_attr("global_symbol", ext_func.attrs["global_symbol"])
    primfunc = primfunc.with_attr("ethos-u.constants", const_dict)
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
    tir_mod = tvm.IRModule()
    tir_mod[symbol] = primfunc

    const_dict_np = dict()
    for buffer_var in const_dict.keys():
        const_dict_np[buffer_var] = const_dict[buffer_var].numpy()

    cmms, encoded_constants, base_addresses = tir_to_cs_translator.translate(tir_mod, const_dict_np)
    return util.CompilationArtifact(symbol, cmms, encoded_constants, base_addresses)
