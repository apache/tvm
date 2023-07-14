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

from typing import Dict, List, Union, Callable, Tuple

import tvm
from tvm import relax
from tvm.ir import IRModule
from tvm.tir import PrimFunc
from tvm.meta_schedule.testing.tune_utils import generate_input_data

from .extract import extract_func_info
from .utils import (
    populuate_input_shape,
    default_dym_var_sample_func,
    get_func_name_from_gv,
    dym_var_sample_str,
    print_results,
)


def benchmark(
    mod_or_func: Union[PrimFunc, IRModule],
    args: List[Union[relax.TensorStructInfo, Tuple[Tuple[Union[int, str], ...], str]]],
    dym_var_sample: Dict[str, int],
    target: Union[str, tvm.target.Target] = "llvm -num-cores=4",
    dev: tvm.runtime.Device = tvm.cpu(),
    number=10,
    repeat=10,
) -> Tuple[List[Tuple[Tuple[int, ...], str]], float, float]:
    """Benchmark a PrimFunc or IRModule with dynamic input shapes.

    Parameters
    ----------
    mod_or_func : Union[PrimFunc, IRModule]
        The PrimFunc or IRModule to be benchmarked.
    args : List[relax.TensorStructInfo]
        The input tensor information, including shape and dtype.
    dym_var_sample : Dict[Union[relax.expr.Call, str], int]
        The dynamic shape variable sample, e.g., {n: 64, m: 128}.
    target : Union[str, tvm.target.Target]
        The target to be benchmarked on.
    dev : tvm.runtime.Device
        The device to be benchmarked on.
    number : int
        The number of times to run the measurement.
    repeat : int
        The number of times to repeat the measurement.

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
        mod = IRModule.from_expr(mod_or_func.with_attr("global_symbol", "main"))
        func_name = "main"
    else:
        mod = mod_or_func
        # assume only one global function
        (func_name,) = mod.get_global_vars()
    # produce target
    target = tvm.target.Target(target)
    # populate input shapes
    input_infos = populuate_input_shape(args, dym_var_sample)
    # generate input tensors, including scalars
    # scalars are appended to the end of the list due to parsing order
    input_tensors = []
    scalar_input_tensors: List[Tuple[Tuple[int], str]] = []
    for input_shape, input_dtype in input_infos:
        if input_dtype == "scalar":
            # special case like [n], generate int value
            assert isinstance(input_shape, int)
            scalar_input_tensors.append(input_shape)
        else:
            # normal case like [1, n, 128], generate random tensor
            input_tensors.append(
                tvm.nd.array(generate_input_data(list(input_shape), input_dtype), device=dev)
            )
    # append scalar input tensors
    input_tensors.extend(scalar_input_tensors)
    # build locally
    print("before build")
    rt_mod = tvm.build(mod, target=target)
    print("after build")
    # benchmark locally
    result = rt_mod.time_evaluator(func_name, dev=dev, number=number, repeat=repeat)(*input_tensors)
    print("after time_evaluator")
    # return input infos, median, std
    return input_infos, result.median, result.std


def benchmark_relax_func(
    mod: tvm.ir.IRModule,
    relax_func: Union[tvm.ir.GlobalVar, str],
    sample_number: int = 2,
    dym_var_sample_func: Callable[
        [Dict[str, str]],
        Dict[str, int],
    ] = default_dym_var_sample_func,
    target: Union[str, tvm.target.Target] = "llvm -num-cores=4",
    dev: tvm.runtime.Device = tvm.cpu(),
    number=10,
    repeat=10,
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
    number : int
        The number of times to run the measurement.
    repeat : int
        The number of times to repeat the measurement.
    """
    relax_funcs, dynamic_var_dict = extract_func_info(mod)

    if isinstance(relax_func, str):
        for gv in relax_funcs:  # pylint: disable=invalid-name
            if get_func_name_from_gv(gv) == relax_func:
                relax_func = gv
                break
        if not isinstance(relax_func, tvm.ir.GlobalVar):
            raise ValueError(f"Cannot find relax function with name {relax_func}")
    for _ in range(sample_number):
        dym_var_sample = dym_var_sample_func(dynamic_var_dict[relax_func])
        bench_results = []
        for functor in relax_funcs[relax_func]:
            for args, weight in relax_funcs[relax_func][functor]:
                _, median, _ = benchmark(
                    mod[functor],
                    args,
                    dym_var_sample,
                    target=target,
                    dev=dev,
                    number=number,
                    repeat=repeat,
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
