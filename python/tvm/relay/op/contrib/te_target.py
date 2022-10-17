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
"""Support a Relay partitioning target using Tensor Expressions."""
from typing import Callable, List, Dict

import tvm
import tvm.ir
from tvm import relay
from tvm import te


_LowerFunc = Callable[[relay.Call, List[te.Tensor]], te.Tensor]
_LOWER_MAP: Dict[str, _LowerFunc] = {}


def lower_composite(comp_name: str) -> Callable[[_LowerFunc], _LowerFunc]:
    """Register a lowering function for a given composite function name."""

    def _register(f: _LowerFunc) -> _LowerFunc:
        _LOWER_MAP[comp_name] = f
        return f

    return _register


def relay_to_runtime(target: tvm.target.Target) -> Callable[[relay.Function], tvm.runtime.Module]:
    """Create a Relay to runtime module lowering function using Tensor Expressions for lowering."""

    def _relay_to_runtime(partition: relay.Function) -> tvm.runtime.Module:
        """Compile Relay functions to a runtime module using Tensor Expressions."""
        assert isinstance(partition, relay.Function)
        assert isinstance(partition.body, relay.Call)
        assert isinstance(partition.body.op, relay.Function)

        global_name = str(partition.attrs.global_symbol)
        comp_func = partition.body.op
        comp_name = comp_func.attrs["Composite"]
        assert comp_name in _LOWER_MAP
        assert isinstance(comp_func.body, relay.Call)

        op = comp_func.body
        inputs = []
        for i, param in enumerate(comp_func.params):
            inputs.append(
                te.placeholder(
                    param.checked_type.shape,
                    name=f"input_{i}",
                    dtype=param.checked_type.dtype,
                )
            )

        output = _LOWER_MAP[comp_name](op, inputs)
        prim_func = te.create_prim_func(inputs + [output])
        return tvm.build(prim_func, target=target, name=global_name)

    return _relay_to_runtime
