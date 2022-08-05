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
"""Backend base class of the Universal Modular Accelerator Interface (UMA)"""

from abc import ABC, abstractmethod
from typing import Union, Dict, Callable, Optional, Any

import tvm
from tvm.relay.backend.contrib.uma.api.codegen import UMACodegen
from tvm.relay.backend.contrib.uma.api.lower import UMALower
from tvm.relay.backend.contrib.uma.api.partitioner import UMAPartitioner
from tvm.relay.backend.contrib.uma.api.utils import PassPhase


class UMABackend(ABC):
    """Backend base class of the Universal Modular Accelerator Interface (UMA)"""

    def __init__(self, merge_compiler_regions: bool = True) -> None:
        self._target_attrs: Dict = {}
        self._target_preprocessor: Callable[[str], Dict[str, Any]] = None
        self._relay_to_relay = UMAPartitioner(self.target_name, merge_compiler_regions)
        self._relay_to_tir = UMALower(self.target_name)
        self._tir_to_runtime = UMACodegen(self.target_name)

    @property
    @abstractmethod
    def target_name(self) -> str:
        """Name of the hardware target.

        Returns
        -------
        out : str
            The hardware target name.
        """
        ...

    # Target configuration
    def _register_target_attr(
        self,
        name: str,
        default: Optional[Union[str, int, bool]] = "",
    ) -> None:
        """Register a target attribute name that can be used during target instantiation.
        Parameters
        ----------
        name: str
           The name of the target attribute.

        default: Optional[Union[str, int, bool]]
            A default value for the attribute.
            If none is provided, the attribute will be treated as a string.

        Example
        -------
        Here is an example of how two attribute options are registered.

        .. code-block:: python

            self._register_target_attr("attrA", default=0)
            self._register_target_attr("attrB", default=False)
        """
        self._target_attrs[name] = default

    # Relay to Relay function registration
    def _register_relay_pass(self, phase: PassPhase, relay_pass: tvm.transform.Pass) -> None:
        """Registers a relay pass at the given phase in the lowering process.

        Parameters
        ----------
        phase: PassPhase
           The phase at which the pass is registered.

        relay_pass: tvm.transform.Pass
            The relay pass to be registered.

        Example
        -------
        Here is an example of how two relay passes are registered.
        Passes of the same phase are executed in the order they are registered.

        .. code-block:: python

            self._register_relay_pass(PassPhase.PRE_PARTITIONING, MyPassA)
            self._register_relay_pass(PassPhase.POST_PARTITIONING, MyPassB)

        Where a relay pass can look like this:

        .. code-block:: python

            @tvm.ir.transform.module_pass(opt_level=0)
            class MyPassA:
                def transform_module(self, mod, ctx):
                    # My pass functionality...
                    return mod
        """
        self._relay_to_relay._relay_passes.append((phase, relay_pass))

    def _register_pattern(
        self,
        name: str,
        pattern: tvm.relay.dataflow_pattern.DFPattern,
        predicate: Optional[Callable] = None,
    ) -> None:
        """Registers a dataflow pattern that is used to partition the relay graph.

        Parameters
        ----------
        name: str
           The name of the pattern

        pattern: tvm.relay.dataflow_pattern.DFPattern
            Relay DFPattern

        predicate: Optional[Callable]
            Optional predicate for Relay DFPattern
        Example
        -------
        Here is an example of how two dataflow patterns are registered.
        During partioning, patterns are searched in order of registration.

        .. code-block:: python

            self._register_pattern("conv1d", conv1d_pattern)
            self._register_pattern("conv2d", conv2d_pattern)

        Where a dataflow pattern can look like this:

        .. code-block:: python

            conv1d_pattern = is_op("nn.conv1d")(wildcard(), wildcard())
            optional_bias = lambda x: is_op("nn.bias_add")(x, wildcard())
            optional_relu = lambda x: is_op("nn.relu")(x)
            conv1d_pattern = conv1d_pattern.optional(optional_bias).optional(optional_relu)
        """
        self._relay_to_relay.add_pattern(name, pattern, predicate)

    # Relay to TIR function registration
    def _register_operator_strategy(
        self,
        op: str,
        strategy: Callable[
            [tvm.ir.Attrs, tvm.ir.Array, tvm.ir.TensorType, tvm.target.Target],
            tvm.relay.op.op.OpStrategy,
        ],
        plevel: Optional[int] = 11,
    ) -> None:
        """Registers an operator strategy that is used to partition the relay graph.

        Parameters
        ----------
        op: str
           The name of the operator for which this strategy will be registered.

        strategy: Callable[[tvm.ir.Attrs, tvm.ir.Array, tvm.ir.TensorType, tvm.target.Target],
                            tvm.relay.op.op.OpStrategy]
            The strategy function.

        plevel: Optional[int] = 11
            The priority level of the strategy. Higher plevel equals higher priorization.
            The TVM default for topi strategies is 10 so by default new UMA strategies are
            always used.

        Example
        -------
        Here is an example of how two operator strategies are registered.

        .. code-block:: python

            self._register_operator_strategy("nn.conv1d", custom_conv1d_strategy)
            self._register_operator_strategy("nn.conv2d", custom_conv2d_strategy)

        Where a strategy function can look like this:

        .. code-block:: python

            @relay.op.strategy.override_native_generic_func("custom_conv1d_strategy")
            def custom_conv1d_strategy(attrs, inputs, out_type, target):
                strategy = _op.OpStrategy()
                strategy.add_implementation(
                    wrap_compute_conv1d(custom_conv1d_compute),
                    wrap_topi_schedule(custom_conv1d_schedule),
                    name="custom_conv1d.generic",
                return strategy
        """
        self._relay_to_tir._operator_strategies.append((op, strategy, plevel))

    def _register_tir_pass(
        self, phase: PassPhase, tir_pass: tvm.tir.transform.PrimFuncPass
    ) -> None:
        """Registers a TIR pass at the given phase in the lowering process.

        Parameters
        ----------
        phase: PassPhase
           The phase at which the pass is registered.

        tir_pass: tvm.tir.transform.PrimFuncPass
            The TIR pass to be registered.
        Example
        -------
        Here is an example of how two TIR passes are registered.
        Passes of the same phase are executed in the order they are registered.

        .. code-block:: python

            self._register_tir_pass(PassPhase.TIR_PHASE_0, MyPassA)
            self._register_tir_pass(PassPhase.TIR_PHASE_1, MyPassB)

        Where a TIR pass can look like this:

        .. code-block:: python

            @tvm.tir.transform.prim_func_pass(opt_level=0)
            class MyPassA:
                def transform_function(self, func, mod, ctx):
                    # My pass functionality...
                    return func
        """
        self._relay_to_tir._tir_passes.append((phase, tir_pass))

    # TIR to runtime function registration
    def _register_codegen(self, fmt: str = "c", **kwargs) -> None:
        """Registers a codegen which is used in place of the default C-codegen.

        Parameters
        ----------
        fmt: str
            The codegen format. For now, only C-codegen is supported by UMA.

        **kwargs
            Keyword arguments for the chosen codegen.

        Example
        -------
        Here is an example of how the custom C-codegen is registered and configured.
        Passes of the same phase are executed in the order they are registered.

        .. code-block:: python

            self._register_codegen(
                fmt="c", includes=gen_includes
            )

        The C-codegen currently provides one hook which allows the user to insert code through
        the python API.
            - `includes` hooks into the include stream and allows insertion of custom includes.


        The code generation functions can look like this:

        .. code-block:: python

            def gen_includes() -> str:
                includes = "#include <my_custom_header.h>\n"
                return includes
        """
        self._tir_to_runtime._register_codegen(fmt, **kwargs)

    # Backend functions
    def register(self) -> None:
        """
        Registering UMABackend:
         registering target attributes, relay_to_relay, relay_to_tir and tir_to_runtime
        """
        registration_func = tvm.get_global_func("relay.backend.contrib.uma.RegisterTarget")

        for name, attr in self._target_attrs:
            if attr is None:
                raise ValueError("Target attribute None is not supported.")

        if registration_func(self.target_name, self._target_attrs):
            self._relay_to_relay.register()
            self._relay_to_tir.register()
            self._tir_to_runtime.register()

    def partition(
        self, mod: tvm.IRModule, params: Optional[Dict[str, tvm.runtime.NDArray]] = None
    ) -> tvm.IRModule:
        return self._relay_to_relay.partition(mod, params)
