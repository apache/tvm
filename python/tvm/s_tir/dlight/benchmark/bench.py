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
"""Extract self-contained benchmarking scripts for dynamic shape workloads"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import tvm
from tvm import relax
from tvm.ir import IRModule
from tvm.s_tir.meta_schedule.runner import EvaluatorConfig
from tvm.s_tir.meta_schedule.testing.tune_utils import generate_input_data
from tvm.tir import PrimFunc

from .extract import extract_all_func_info_from_relax, extract_func_info_from_prim_func
from .utils import (
    default_dym_var_sample_func,
    dym_var_sample_str,
    get_func_name_from_gv,
    populuate_input_shape,
    print_results,
)

if TYPE_CHECKING:
    from tvm.s_tir.meta_schedule.runner import RPCConfig


def benchmark(
    mod_or_func: PrimFunc | IRModule,
    *,
    dym_var_sample: dict[str, int],
    args: list[relax.TensorStructInfo | tuple[tuple[int | str, ...], str]] | None,
    target: str | tvm.target.Target | None = None,
    func_name: str | None = None,
    evaluator_config: EvaluatorConfig | None = None,
    rpc_config: RPCConfig | None = None,
) -> tuple[list[tuple[tuple[int, ...], str]], float, float]:
    """Benchmark a PrimFunc or IRModule with dynamic input shapes.

    Parameters
    ----------
    mod_or_func : Union[PrimFunc, IRModule]
        The PrimFunc or IRModule to be benchmarked.
    dym_var_sample : Optional[Dict[str, int]]
        The dynamic shape variable sample, e.g., {"n": 64, "m": 128}.
    args : Optional[List[Union[relax.TensorStructInfo, Tuple[Tuple[Union[int, str], ...], str]]]]
        The input tensor information, including shape and dtype. If none, will use
        the input information from the PrimFunc or IRModule.
    target : Optional[Union[str, tvm.target.Target]]
        The target to be benchmarked on, if none, will get the target from context.
    func_name : Optional[str]
        The name of the function to be benchmarked, will use "main" by default.
    evaluator_config : Optional["EvaluatorConfig"]
        The evaluator configuration to use.
        If none, will use default evaluator configuration.
    rpc_config : Optional["RPCConfig"]
        The RPC configuration to connect to the remote device.
        If none, will use local mode.

    Returns
    -------
    input_infos : List[Tuple[Tuple[int, ...], str]]
        The input tensor information, including shape and dtype.
    median : float
        The median of the benchmarking results.
    std : float
        The standard deviation of the benchmarking results.
    """
    # produce IRModule and function name
    if isinstance(mod_or_func, PrimFunc):
        func_name = "main" if func_name is None else func_name
        mod = IRModule.from_expr(mod_or_func.with_attr("global_symbol", func_name))
    else:
        mod = mod_or_func
        # assume only one global function
        (func_name,) = mod.get_global_vars()
        func_name = func_name.name_hint
    # produce input shapes
    if args is None:
        args, _ = extract_func_info_from_prim_func(mod[func_name])
    # produce target & device
    target = tvm.target.Target.current() if target is None else tvm.target.Target(target)
    if target is None:
        raise ValueError("Target is not specified")
    if target.kind.name == "llvm":
        dev = tvm.cpu()
    elif target.kind.name == "cuda":
        dev = tvm.cuda()
    else:
        raise ValueError(f"Unsupported device type from {target.kind.name}")
    # populate input shapes
    input_infos = populuate_input_shape(args, dym_var_sample)
    # generate input tensors, including scalars
    # scalars are appended to the end of the list due to parsing order
    input_tensors: list[tvm.runtime.Tensor | int] = []
    scalar_input_tensors: list[int] = []
    for input_shape, input_dtype in input_infos:
        if input_dtype == "scalar":
            # special case like [n], generate int value
            assert len(input_shape) == 1
            scalar_input_tensors.append(input_shape[0])
        else:
            # normal case like [1, n, 128], generate random tensor
            input_tensors.append(
                tvm.runtime.tensor(generate_input_data(list(input_shape), input_dtype), device=dev)
            )
    # append scalar input tensors for rotary embedding
    input_tensors.extend(scalar_input_tensors)
    # build locally
    rt_mod = tvm.tir.build(mod, target=target)
    # set up evaluator config
    evaluator_config = EvaluatorConfig._normalized(evaluator_config)
    # run benchmark
    if rpc_config is None:
        profile_result = rt_mod.time_evaluator(
            func_name,
            dev=dev,
            number=evaluator_config.number,
            repeat=evaluator_config.repeat,
            min_repeat_ms=evaluator_config.min_repeat_ms,
            f_preproc=(
                "cache_flush_cpu_non_first_arg" if evaluator_config.enable_cpu_cache_flush else ""
            ),
        )(*input_tensors)
    else:
        from tvm.testing import rpc_run

        _, profile_result = rpc_run(
            rt_mod,
            device_type=dev._DEVICE_TYPE_TO_NAME[dev.dlpack_device_type()],
            args=[w.numpy() if isinstance(w, tvm.runtime.Tensor) else w for w in input_tensors],
            rpc_config=rpc_config,
            evaluator_config=evaluator_config,
        )
    # return input infos, median, std
    return input_infos, profile_result.median, profile_result.std


def benchmark_prim_func(
    mod_or_func: PrimFunc | IRModule,
    *,
    dym_var_sample_func: Callable[[dict[str, str]], dict[str, int]] = default_dym_var_sample_func,
    args: list[relax.TensorStructInfo | tuple[tuple[int | str, ...], str]] | None = None,
    dym_var_dict: dict[str, str] | None = None,
    sample_number: int = 5,
    target: str | tvm.target.Target | None = None,
    weight: int | None = 1,
    relax_func_name: str | None = None,
    prim_func_name: str | None = None,
    evaluator_config: EvaluatorConfig | None = None,
    rpc_config: RPCConfig | None = None,
    sort_by: str | None = None,
    desc: bool | None = True,
):
    """Benchmark a PrimFunc or IRModule with dynamic input shapes and show results.

    Parameters
    ----------
    mod_or_func : Union[PrimFunc, IRModule]
        The PrimFunc or IRModule to be benchmarked.
    dym_var_sample_func : Callable[[Dict[str, str]], Dict[str, int]]
        The function to sample dynamic shape variables.
    dym_var_dict : Optional[Dict[str, str]]
        Dynamic shape variable dictionary, e.g., {"n": "int32", "m": "int32"}. If none, will use
        the input information from the PrimFunc or IRModule.
    args : Optional[List[Union[relax.TensorStructInfo, Tuple[Tuple[Union[int, str], ...], str]]]]
        The input tensor information, including shape and dtype. If none, will use
        the input information from the PrimFunc or IRModule.
    sample_number : int
        The number of times to sample dynamic shape variables.
    target: Optional[Union[str, tvm.target.Target]]
        The target to be benchmarked on, if none, will get the target from context.
    weight : Optional[int]
        The weight of this PrimFunc.
    relax_func_name : Optional[str]
        The name of the relax function.
    prim_func_name : Optional[str]
        The name of the PrimFunc.
    evaluator_config : Optional["EvaluatorConfig"]
        The evaluator configuration to use.
        If none, will use default evaluator configuration.
    rpc_config : Optional["RPCConfig"]
        The RPC configuration to connect to the remote device.
        If none, will use local mode.
    sort_by : Optional[str]
        Sort results by this key, if None, no sorting.
    desc : Optional[bool]
        Whether to sort results in descending order.
    """
    results = []
    if dym_var_dict is None or args is None:
        args, dym_var_dict = extract_func_info_from_prim_func(mod_or_func)
    for _ in range(sample_number):
        dym_var_sample = dym_var_sample_func(dym_var_dict)
        _, median, std = benchmark(
            mod_or_func,
            args=args,
            dym_var_sample=dym_var_sample,
            target=target,
            evaluator_config=evaluator_config,
            rpc_config=rpc_config,
        )
        row = {
            "InputInfo": ", ".join([f"{k} = {v}" for k, v in dym_var_sample.items()]),
            "Time(us)": median * 1e6,
            "Std(us)": std * 1e6,
        }
        if relax_func_name is not None:
            row["RelaxFunc"] = relax_func_name
        if prim_func_name is not None:
            row["PrimFunc"] = prim_func_name
        weight = 1 if weight is None else weight
        row["Weight"] = weight
        row["WxTime(ms)"] = weight * median * 1e3
        results.append(row)
    print_results(results, sort_by=sort_by, desc=desc)


def benchmark_relax_func(
    mod: tvm.ir.IRModule,
    relax_func: tvm.ir.GlobalVar | str,
    sample_number: int = 2,
    dym_var_sample_func: Callable[
        [dict[str, str]],
        dict[str, int],
    ] = default_dym_var_sample_func,
    target: str | dict | tvm.target.Target = None,
    evaluator_config: EvaluatorConfig | None = None,
    rpc_config: RPCConfig | None = None,
) -> None:
    """Benchmark a relax function with dynamic input shapes.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The IRModule to be benchmarked.
    relax_func : Union[tvm.ir.GlobalVar, str]
        The relax function to be benchmarked.
    sample_number : int
        The number of times to sample dynamic shape variables.
    dym_var_sample_func : Callable[[Dict[str, str]], Dict[str, int]]
        The function to sample dynamic shape variables.
    target : Union[str, tvm.target.Target]
        The target to be benchmarked on.
    dev : tvm.runtime.Device
        The device to be benchmarked on.
    evaluator_config : Optional["EvaluatorConfig"]
        The evaluator configuration to use.
        If none, will use default evaluator configuration.
    rpc_config : Optional["RPCConfig"]
        The RPC configuration to connect to the remote device.
    """
    if target is None:
        target = {"kind": "llvm", "num-cores": 4}
    # extract function information
    relax_funcs, dynamic_var_dict = extract_all_func_info_from_relax(mod)
    # find the relax function global var
    if isinstance(relax_func, str):
        for gv in relax_funcs:
            if get_func_name_from_gv(gv) == relax_func:
                relax_func = gv
                break
        if not isinstance(relax_func, tvm.ir.GlobalVar):
            raise ValueError(
                f"Cannot find relax function with name {relax_func}, "
                + f"candidates are: {[get_func_name_from_gv(gv) for gv in relax_funcs]}"
            )
    # benchmark
    for _ in range(sample_number):
        dym_var_sample = dym_var_sample_func(dynamic_var_dict[relax_func])
        bench_results = []
        # enumerate all functors
        for functor in relax_funcs[relax_func]:
            for args, weight in relax_funcs[relax_func][functor]:
                _, median, _ = benchmark(
                    mod[functor],
                    args=args,
                    dym_var_sample=dym_var_sample,
                    target=target,
                    evaluator_config=evaluator_config,
                    rpc_config=rpc_config,
                )
                bench_results.append(
                    {
                        f"PrimFuncs in {get_func_name_from_gv(relax_func)}": get_func_name_from_gv(
                            functor
                        ),
                        f"InputInfo({dym_var_sample_str(dym_var_sample)})": ", ".join(
                            [str(w) for w in args]
                        ),
                        "Time(us)": median * 1e6,
                        # "Std(us)": std * 1e6,
                        "Weight": weight,
                        "WxTime(ms)": median * weight * 1e3,
                    }
                )
        print_results(bench_results)
