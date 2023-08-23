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
from typing import List, Callable

from ethosu.vela import api as vapi
import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu.tir.compiler import LowerToTIR
from tvm.relay.backend.contrib.ethosu.tir.scheduler import copy_constants
from tvm.contrib.ethosu.cascader import (
    cascade,
    EthosuDeviceConfig,
    CascaderOptions,
    MemoryRegion,
    extract_memory_info,
)
from tvm.relay.backend.contrib.ethosu.legalize import LegalizeEthosU
from tvm.relay.backend.contrib.ethosu import tir_to_cs_translator, util, vela_api
from tvm.relay.expr_functor import ExprMutator, ExprVisitor, Call
from tvm.relay import expr as _expr

# pylint: disable=unused-import
from tvm.relay.backend.contrib.ethosu.op import op_attrs
from tvm.relay.backend.contrib.ethosu import op

from . import _ffi_api


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
            "contrib.ethosu.binary_elementwise": op.ethosu_binary_elementwise,
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


@util.create_npu_function_pass(opt_level=1)
class LUTsOptimizer:
    """Register LUTsOptimizer as a relay pass."""

    def transform_npu_function(self, _, func: relay.Function) -> relay.Function:
        """Visit relay nodes in the given NPU function.

        Parameters
        ----------
        func : tvm.relay.function.Function
            The function to apply the optimization pass for multiple LUTs to.

        Returns
        -------
        mod : tvm.IRModule
            New module with optimized LUTs.
        """
        return OptimizeLUTs().visit(func)

    def __call__(self, *args, **kwargs):
        pass


class AnalyzeConsumers(ExprVisitor):
    """Traverses the graph to determine consumers that are NPU operations and
    which have restrictions to use NHCWB16 layout. The result is maintained in
    `npu_consumers` and `restrictions`.

    Attributes
    ----------
    npu_consumers : Dict[tvm.relay.expr.Call, List[bool]]
        Mapping from NPU operation to list of boolean values that represent
        whether or not each consumer is an NPU operation.
    restrictions : Dict[tvm.relay.expr.Call, List[bool]]
        Mapping from NPU operation to list of boolean values that represent
        whether or not operation has restrictions to use NHCWB16 layout.
    optimize_ops : Dict[str, Callable]
        A map from NPU operation name to function that creates NPU operation.
    """

    def __init__(self, optimize_ops):
        self.npu_consumers = defaultdict(list)
        self.restrictions = defaultdict(list)
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
                # ReduceSum requires NHWC input in case input tensor has type int32 or
                # accelerator is Ethos_U65_512
                # https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-vela/+/refs/tags/3.7.0/ethosu/vela/graph_optimiser_util.py#126
                has_restrictions = (
                    call.op.name == "contrib.ethosu.pooling"
                    and call.attrs["pooling_type"] == "SUM"
                    and (
                        arg.checked_type.dtype == "int32"
                        or vela_api.get_accelerator_config() == vapi.NpuAccelerator.Ethos_U65_512
                    )
                )
                self.restrictions[arg].append(has_restrictions)

        super().visit_call(call)


class LayoutOptimization(ExprMutator):
    """A pass to optimize the layout of NPU operations by converting to brick format (NHCWB16).
    This pass traverses the graph and attempts to alter the input/output layouts when an NPU
    operation is visited. Whether or not the input/output layout can be altered for a given NPU
    operation depends on the following:

    Check alter input layout: For each argument, if the producer is also an NPU operation and
        its output is altered to brick format and there are no restrictions, then the input layout
        with respect to the current argument is altered to brick format.

    Check alter output layout: If all consumers (child nodes) are an NPU operation and
        there are no restrictions, then the output layout is altered to brick format.

    Note
    ----
    In order for this pass to be run, the consumers of each NPU operation must first be analyzed
    by the `AnalyzeConsumers` pass, since Relay doesn't keep a reference to child nodes.

    Attributes
    ----------
    npu_consumers : Dict[tvm.relay.expr.Call, List[bool]]
        A map from current call to a list boolean values that state whether or not each consumer
        is an NPU operation.
    restrictions : Dict[tvm.relay.expr.Call, List[bool]]
        A map from current call to a list boolean values that state
        whether or not operation has restrictions to use NHCWB16 layout.
    optimize_ops : Dict[str, Callable]
        A map from NPU operation name to function that creates NPU operation.
    """

    def __init__(self, npu_consumers, restrictions, optimize_ops):
        self.npu_consumers = npu_consumers
        self.restrictions = restrictions
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

        def are_all_consumers_npu(call):
            """
            Check whether or not each consumer is an NPU operation.
            Parameters
            ----------
            call : tvm.relay.expr.Call
                The call pointing to an NPU operation.

            Returns
            -------
            all_consumers_npu : bool
                Whether each consumer is an NPU operation.
            """
            consumers = self.npu_consumers[call]
            return consumers and all(consumers)

        def check_restrictions(call):
            """
            Check if there are any restrictions for call to use NHCWB16 layout.
            Parameters
            ----------
            call : tvm.relay.expr.Call
                The call pointing to an NPU operation.

            Returns
            -------
            any_restrictions : bool
                Whether there are restrictions.
            """
            restrictions = self.restrictions[call]
            return restrictions and any(restrictions)

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
            parent_has_brick_output = are_all_consumers_npu(arg)
            parent_has_restrictions = check_restrictions(arg)
            if parent_has_brick_output and not parent_has_restrictions:
                layout_string = "ifm_layout" if input_count <= 1 else f"ifm{input_count}_layout"
                new_attrs[layout_string] = "NHCWB16"

        # Check if we can rewrite the output layouts
        has_brick_output = are_all_consumers_npu(call)
        has_restrictions = check_restrictions(call)
        if has_brick_output and not has_restrictions:
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


@util.create_npu_function_pass(opt_level=1)
class LayoutOptimizer:
    """Register LayoutOptimizer as a Relay pass."""

    def transform_npu_function(self, _, func: relay.Function) -> relay.Function:
        """A pass to optimize the layout of NPU operations. If both the
        producer and consumer of a tensor are NPU operators, then the
        layout is converted from NHWC to NHCWB16 as this is the layout NPU
        uses internally."""

        optimize_ops = {
            "contrib.ethosu.conv2d": op.ethosu_conv2d,
            "contrib.ethosu.depthwise_conv2d": op.ethosu_depthwise_conv2d,
            "contrib.ethosu.pooling": op.ethosu_pooling,
            "contrib.ethosu.binary_elementwise": op.ethosu_binary_elementwise,
            "contrib.ethosu.unary_elementwise": op.ethosu_unary_elementwise,
        }

        analyze = AnalyzeConsumers(optimize_ops)
        analyze.visit(func)
        return LayoutOptimization(analyze.npu_consumers, analyze.restrictions, optimize_ops).visit(
            func
        )

    def __call__(self, *args, **kwargs):
        pass


class PadsWithMultipleConsumersReplicator(ExprMutator):
    """A pass to handle the situation when nn.pad operator has
    more than one qnn.conv2d consumer.

             pad
           /     \
       Conv2D   Conv2D

    In this case, because of the peculiarities of pattern parsing,
    conv2d does not get into the composite for the NPU.
    Therefore, pads are added so that each has only one consumer.
    """

    def __init__(self):
        super().__init__()
        # a set to record hashes of an pads which already have one qnn.conv2d consumer
        self.hashes = set()

    def visit_call(self, call: tvm.relay.expr.Call) -> tvm.relay.expr.Call:
        if (
            isinstance(call.op, tvm.ir.Op)
            and isinstance(call.args[0], Call)
            and isinstance(call.args[0].op, tvm.ir.Op)
            and call.op == relay.op.get("qnn.conv2d")
            and call.args[0].op == relay.op.get("nn.pad")
        ):
            if tvm.ir.structural_hash(call.args[0]) not in self.hashes:
                # add the hash of nn.pad to set
                self.hashes.add(tvm.ir.structural_hash(call.args[0]))
            else:
                # if this pad already has a conv2d consumer, duplicate the pad
                # and make it an input for current conv2d
                used_pad = self.visit(call.args[0])
                used_pad_args = [self.visit(arg) for arg in used_pad.args]
                new_pad = Call(
                    used_pad.op, used_pad_args, used_pad.attrs, used_pad.type_args, used_pad.span
                )
                new_conv2d_args = []
                for i, arg in enumerate(call.args):
                    if i == 0:
                        new_conv2d_args.append(self.visit(new_pad))
                    else:
                        new_conv2d_args.append(self.visit(arg))
                new_conv2d_op = self.visit(call.op)
                expr__ = _expr.CallWithFields(
                    call,
                    new_conv2d_op,
                    new_conv2d_args,
                    call.attrs,
                    call.type_args,
                    None,
                    call.span,
                )
                return expr__

        new_args = [self.visit(arg) for arg in call.args]
        new_op = self.visit(call.op)
        expr__ = _expr.CallWithFields(
            call, new_op, new_args, call.attrs, call.type_args, None, call.span
        )
        return expr__


def replicate_pads(mod):
    """Traverses the Relay graph to replicate nn.pad operators if thay have
    multiple qnn.conv2d consumers. That making remove the situation when
    e.g. pad+conv2d corresponds qnn_conv2d_pattern, but can not be grouped
    because several conv2d use the same pad operation.

    Parameters
    ----------
    tvm.ir.IRModule
        The IRModule that gets generated from a relay frontend.

    Returns
    -------
    tvm.ir.IRModule
        The IRModule without nn.pad operators with multiple consumers.
    """
    replicator = PadsWithMultipleConsumersReplicator()
    for global_var, func in mod.functions.items():
        func = replicator.visit(func)
        mod.update_func(global_var, func)
    return mod


class AnalyzeConcatArgs(ExprVisitor):
    """Traverses the graph to determine which arguments were passed into the
    concatenation operation and how many times they are used. The result is
    maintained in `args_usage` and is a dictionary where the key is the concatenation argument and
    the value is the number of uses of this argument.

    Attributes
    ----------
    args_usage : Dict[tvm.relay.expr.Call, int]
        Mapping from concatenation arguments to count their usage as concatenate arguments.
    """

    def __init__(self):
        self.args_usage = defaultdict(int)
        super().__init__()

    def visit_call(self, call: relay.Call):
        args = []

        # Expand tuples
        for arg in call.args:
            if isinstance(arg, relay.Tuple):
                args.extend(arg.fields)
            else:
                args.append(arg)

        if isinstance(call.op, tvm.ir.Op) and call.op.name == "concatenate":
            for arg in args:
                if isinstance(arg, relay.Call):
                    self.args_usage[arg] += 1

        super().visit_call(call)


class ConcatArgsCopier(ExprMutator):
    """A pass for copying concatenation arguments that are used in multiple concatenation
    operations. For a concatenation argument that is used n times, n - 1 copy operations
    will be created.

    Attributes
    ----------
    args_usage : Dict[tvm.relay.expr.Call, int]
        Mapping from concatenation arguments to count their usage as concatenate arguments.
    """

    def __init__(self, args_usage):
        super().__init__()
        self.args_usage = args_usage

    def visit_call(self, call: tvm.relay.expr.Call) -> tvm.relay.expr.Call:
        if isinstance(call.op, tvm.ir.Op) and call.op == relay.op.get("concatenate"):
            args = []

            # Expand tuples
            for arg in call.args:
                if isinstance(arg, relay.Tuple):
                    args.extend(arg.fields)
                else:
                    args.append(arg)
            new_args = []
            for arg in args:
                visited = self.visit(arg)
                if self.args_usage[arg] > 1:
                    # Add copy operation
                    lut = relay.const([], "int8")
                    new_op = op.ethosu_identity(visited, lut)
                    new_args.append(new_op)
                    self.args_usage[arg] -= 1
                else:
                    new_args.append(visited)

            new_args = [relay.Tuple(new_args)]
        else:
            new_args = [self.visit(arg) for arg in call.args]
        new_op = self.visit(call.op)
        new_call = _expr.CallWithFields(
            call, new_op, new_args, call.attrs, call.type_args, None, call.span
        )
        return new_call


@util.create_npu_function_pass(opt_level=1)
class CopyReusedConcatBuffers:
    """Register CopyReusedConcatBuffers as a Relay pass."""

    def transform_npu_function(self, _, func: relay.Function) -> relay.Function:
        """A pass to copy concatenation arguments which are used more than once in
        concatenation operation. This is the preparation for the next RemoveConcatenates
        pass to prevent a situation where an argument used in multiple concatenations
        will be written to only one resulting buffer."""

        analyze = AnalyzeConcatArgs()
        analyze.visit(func)

        return ConcatArgsCopier(analyze.args_usage).visit(func)

    def __call__(self, *args, **kwargs):
        pass


def IdentityOptimizer():  # pylint: disable=invalid-name
    """Pass that removes redundant identities

    Return
    ------
    Pass
        The module pass.
    """
    return _ffi_api.IdentityOptimizer()


def OutlineCompilerFunctions(compiler_name):  # pylint: disable=invalid-name
    """Pass that outlines functions given a named Compiler attribute.

    Parameters
    ----------
    compiler_name
        The name of the compiler to look for and outline.

    Return
    ------
    Pass
        The module pass.
    """
    return _ffi_api.OutlineCompilerFunctions(compiler_name)


@tvm._ffi.register_func("relay.ext.ethos-u.constant_updater")
def constant_updater(expr, symbol):  # pylint: disable=unused-argument
    """
    The constant updater process happen after lowering in the core compiler.
    For the NPU, we dont want the build process to extract constants to be loaded in
    the runtime as we are embedding them inside the C runtime.Module.
    """
    return dict()


def _create_cascader(
    options: CascaderOptions,
    io_region: MemoryRegion,
    constant_region: MemoryRegion,
    working_regions: List[MemoryRegion],
    device_config: EthosuDeviceConfig,
) -> Callable:
    def _cascader(te_graph, const_dict, sch):
        cascade(
            sch,
            te_graph,
            const_dict,
            options,
            io_region,
            constant_region,
            working_regions,
            device_config,
        )

    return _cascader


def _ethos_u55_cascader(sram, enable_striping) -> Callable:
    # TODO(ekalda): Extract the flash info from ConstantPools once it is implemented
    flash = MemoryRegion(name="FLASH", size=10**7, read_bandwidth=4, write_bandwidth=4)

    device_config = EthosuDeviceConfig(util.get_accelerator_config())
    cascader_options = CascaderOptions(
        cascade_region=sram,
        max_proposals=64,
        stripe_factors=5,
        max_plan_size=10,
        always_copy_size=1024,
        max_open_plans=8,
        max_closed_plans=32,
        enable_striping=enable_striping,
    )
    return _create_cascader(
        options=cascader_options,
        io_region=sram,
        constant_region=flash,
        working_regions=[sram],
        device_config=device_config,
    )


def _calculate_memory_pressure(mod: tvm.ir.IRModule) -> int:
    """
    Calculates a worst-case estimate of the memory consumed at the callsite of
    each microNPU function. This value can be used as a hint to guide the cascader,
    indicating how aggressively it will need to optimize the input module to fit
    into the memory that remains in the memory workspace.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The input module

    Returns
    -------
    int
        Memory pressure value for the module.
    """
    memory_pressure = 0

    @util.create_npu_function_pass(opt_level=1)
    class CalculateMemoryPressure:
        """
        Traverse the module and get total memory used by external NPU functions.
        """

        def transform_npu_function(self, _, func: relay.Function) -> relay.Function:
            nonlocal memory_pressure
            max_val = max(func.attrs["used_memory"])
            memory_pressure += max_val
            return func

    CalculateMemoryPressure()(mod)  # pylint: disable=not-callable

    io_used_memory = 0
    if not tvm.tir.usmp.utils.use_workspace_io_is_enabled():
        io_used_memory = int(mod["main"].attrs["io_used_memory"])

    return memory_pressure - io_used_memory


@tvm._ffi.register_func("relay.ext.ethos-u.relay_to_tir")
def relay_to_tir(mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
    """
    This is the hook for python-based lowering of a Relay module which lowers NPU
    external functions to TIR.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        This is the Relay module.

    Returns
    -------
    mod : tvm.ir.IRModule
        The Relay module with scheduled NPU external functions.
    """
    mod = OutlineCompilerFunctions("ethos-u")(mod)
    mod = LegalizeEthosU()(mod)
    mod = CopyReusedConcatBuffers()(mod)
    mod = LUTsOptimizer()(mod)
    mod = relay.transform.InferType()(mod)
    mod = IdentityOptimizer()(mod)
    mod = LayoutOptimizer()(mod)
    mod = relay.transform.InferType()(mod)

    device_contexts = {
        gv: "ethos-u" for gv, _ in filter(lambda x: util.is_npu_func(x[1]), mod.functions.items())
    }
    mod = mod.with_attr("device_contexts", device_contexts)

    # Use the cascader if it is enabled for the U55 accelerator, otherwise use copy_constants
    # scheduler
    if util.is_cascader_enabled():
        if util.get_accelerator_config() == "ethos-u65-256":
            raise ValueError("Cascading is not supported for the U65 accelerator")

        workspace_memory_pools = mod.attrs["workspace_memory_pools"]

        if not workspace_memory_pools:
            raise ValueError("Workspace memory pool needs to be provided for the U55 cascader")
        if len(workspace_memory_pools.pools) != 1:
            raise ValueError("Exactly one workspace pool needs to be provided for the U55 cascader")

        memory_pressure = _calculate_memory_pressure(mod)
        sram = extract_memory_info(workspace_memory_pools.pools[0], memory_pressure)
        tir_mod = LowerToTIR(_ethos_u55_cascader(sram, util.is_striping_enabled()))(mod)
    else:
        scheduler = None if util.is_copying_constants_disabled() else copy_constants()
        tir_mod = LowerToTIR(scheduler)(mod)

    return tir_mod


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
