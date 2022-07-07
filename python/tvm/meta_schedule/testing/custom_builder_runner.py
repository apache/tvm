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
"""Customized builder and runner methods"""
# pylint: disable=import-outside-toplevel

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    import numpy as np  # type: ignore
    from tvm.ir import IRModule
    from tvm.meta_schedule.runner import EvaluatorConfig, RPCConfig
    from tvm.runtime import Device, Module, NDArray
    from tvm.target import Target
    from tvm.runtime.vm import Executable


def build_relay(
    mod: "IRModule",
    target: "Target",
    params: Dict[str, "NDArray"],
) -> "Module":
    """Build a Relay IRModule

    Parameters
    ----------
    mod : IRModule
        The Relay IRModule to build.
    target : Target
        The target to build the module for.
    params : Dict[str, NDArray]
        The parameter dict to build the module with.

    Returns
    -------
    mod : runtime.Module
        The built module.
    """
    from tvm.relay.build_module import _build_module_no_factory as relay_build
    from tvm.runtime import Module

    result = relay_build(mod, target=target, target_host=None, params=params)
    assert isinstance(result, Module)
    return result


def build_relay_with_tensorrt(
    mod: "IRModule",
    target: "Target",
    params: Dict[str, "NDArray"],
) -> "Module":
    """Build a Relay IRModule with TensorRT BYOC

    Parameters
    ----------
    mod : IRModule
        The Relay IRModule to build.

    target : Target
        The target to build the module for.

    params : Dict[str, NDArray]
        The parameter dict to build the module with.

    Returns
    -------
    mod : runtime.Module
        The built module.
    """
    from tvm.ir.transform import PassContext
    from tvm.relay.build_module import _build_module_no_factory as relay_build
    from tvm.relay.op.contrib import tensorrt
    from tvm.runtime import Module

    mod = tensorrt.partition_for_tensorrt(mod, params)
    with PassContext(opt_level=3):
        result = relay_build(mod, target=target, target_host=None, params=params)
    assert isinstance(result, Module)
    return result


def run_with_graph_executor(
    rt_mod: "Module",
    device: "Device",
    evaluator_config: "EvaluatorConfig",
    repeated_args: List["NDArray"],
) -> List[float]:
    """Run a Relay module with GraphExecutor

    Parameters
    ----------
    rt_mod : Module
        The Relay module to run.
    device : Device
        The device to run the module on.
    evaluator_config : EvaluatorConfig
        The evaluator configuration to run the module with.
    repeated_args : List[NDArray]
        The list of repeated arguments to run the module with.

    Returns
    -------
    results : List[float]
        The list of results.
    """
    import itertools

    from tvm.contrib.graph_executor import GraphModule

    graph_mod = GraphModule(rt_mod["default"](device))
    evaluator = graph_mod.module.time_evaluator(
        func_name="run",
        dev=device,
        number=evaluator_config.number,
        repeat=evaluator_config.repeat,
        min_repeat_ms=evaluator_config.min_repeat_ms,
        f_preproc="cache_flush_cpu_non_first_arg"
        if evaluator_config.enable_cpu_cache_flush
        else "",
    )
    repeated_costs = []
    for args in repeated_args:
        profile_result = evaluator(*args)
        repeated_costs.append(profile_result.results)
    costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]
    return costs


def run_module_via_rpc(
    rpc_config: "RPCConfig",
    lib: Union["Module", "Executable"],
    dev_type: str,
    args: Dict[str, "np.ndarray"],
    continuation: Callable,
    backend: Optional[str] = "graph",
):
    """Execute a tvm.runtime.Module on RPC remote"""
    # pylint: disable=import-outside-toplevel
    import os
    import tempfile

    from tvm.contrib.tar import tar
    from tvm.runtime import ndarray

    # pylint: enable=import-outside-toplevel

    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = os.path.join(tmp_dir, "tvm_tmp_mod." + tar.output_format)
        if backend == "vm":
            code, lib = lib.save()
        lib.export_library(filename, tar)
        session = rpc_config.connect_server()
        session.upload(filename)
        _, filename = os.path.split(filename)
        rt_mod = session.load_module(filename)
        if backend == "vm":
            rt_mod = session.get_function("runtime.Load_Executable")(code, rt_mod)
        dev = session.device(dev_type=dev_type, dev_id=0)
        nd_args = {k: ndarray.array(v, dev) for k, v in args.items()}
        return continuation(rt_mod, dev, nd_args)
