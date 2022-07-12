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
"""Testing utility functions in meta schedule"""
from typing import Callable, Optional, Union, List, Dict
from statistics import median
import json
import warnings
import numpy as np  # type: ignore

import tvm
from tvm.runtime import NDArray


def generate_input_data(
    input_shape: List[int],
    input_dtype: str,
    *,
    low: Optional[int] = None,
    high: Optional[int] = None,
) -> np.ndarray:
    """Generate input date with given shape and data type.

    Parameters
    ----------
    input_shape : List[int]
        The shape of the input data.
    input_dtype : str
        The data type of the input date.

    Returns
    -------
    input_data : np.ndarray
        The generated input data with given shape and data type in numpy ndarray.
    """
    if input_dtype.startswith("float"):
        return np.random.uniform(size=input_shape).astype(input_dtype)
    if low is None or high is None:
        warnings.warn(
            f"Model input value range for shape {input_shape} of {input_dtype} is not set!"
        )
    range_map = {
        "uint8": (0, 255),
        "int8": (-128, 127),
        "int32": (0, 10000),
        "int64": (0, 10000),
    }
    if input_dtype in range_map:
        _low, _high = range_map[input_dtype]
        return np.random.randint(
            low=_low if low is None else low,
            high=_high if high is None else high,
            size=input_shape,
            dtype=input_dtype,
        )
    raise ValueError("Unsupported input datatype!")


def create_timer(backend: str) -> Callable:
    """Create a function to run and benchmark the performance of whole given runtime module,
    or Executable in relay vm.

    Parameters
    ----------
    backend : str
        The backend to use, graph / vm.

    Returns
    -------
    func : Callable
        The function to benchmark the workload.
    """

    def f_timer(
        rt_mod: Union[tvm.runtime.Module, tvm.runtime.vm.Executable],
        dev: tvm.device,
        input_data: Dict[str, NDArray],
    ) -> None:
        """Run and benchmark the given runtime module, print out the result.

        Parameters
        ----------
        rt_mod : Union[tvm.runtime.Module, tvm.runtime.vm.Executable]
            The runtime module or vm executable.
        dev : tvm.device
            The device type to run workload.
        input_data : Dict[str, np.ndarray]
            The input data as a dictionary.
        """
        from tvm.contrib.graph_executor import GraphModule  # pylint:disable=import-outside-toplevel
        from tvm.runtime.vm import VirtualMachine  # pylint:disable=import-outside-toplevel

        try:
            if backend == "vm":
                vm = VirtualMachine(rt_mod, dev)  # pylint: disable=invalid-name
                ftimer = vm.benchmark(
                    dev, min_repeat_ms=500, repeat=5, number=1, end_to_end=False, **input_data
                )
            elif backend == "graph":
                mod = GraphModule(rt_mod["default"](dev))
                for input_name, input_value in input_data.items():
                    mod.set_input(input_name, input_value)
                ftimer = mod.module.time_evaluator(
                    "run", dev, min_repeat_ms=500, repeat=5, number=1
                )()
            else:
                raise ValueError(f"Backend {backend} not supported in f_timer!")

            results = list(np.array(ftimer.results) * 1000.0)  # type: ignore

            print("Running time in time_evaluator: ", results)
            print("-------------------------------")
            print(f"    Min (ms) : {min(results)}")
            print(f"    Max (ms) : {max(results)}")
            print(f" Median (ms) : {median(results)}")
            print(f"Average (ms) : {sum(results) / len(results)}")
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"Run module f_timer via RPC failed, exception: {exc}",
            )

    return f_timer


def create_time_per_layer(graph: str) -> Callable:
    """Create a function to run and benchmark the per-layer performance of given runtime module,
    given the graph output of the module from graph compiler.

    Parameters
    ----------
    graph : str
        The json format graph output of the module from graph compiler.

    Returns
    -------
    func : Callable
        The function using the json format graph.
    """

    def f_time_per_layer(
        rt_mod: tvm.runtime.Module,
        dev: tvm.device,
        input_data: Dict[str, NDArray],
    ) -> None:
        """Run and benchmark the per-layer performance of given runtime module,
        print out the result.

        Parameters
        ----------
        rt_mod : tvm.runtime.Module
            The runtime module.
        dev : tvm.device
            The device type to run workload.
        input_data : Dict[str, np.ndarray]
            The input data as a dictionary.
        """
        # pylint:disable=import-outside-toplevel
        from tvm.contrib.debugger.debug_executor import create

        # pylint:enable=import-outside-toplevel

        try:
            mod = create(graph, rt_mod, dev)
            for input_name, input_value in input_data.items():
                mod.set_input(input_name, input_value)
            graph_nodes = [n["name"] for n in json.loads(graph)["nodes"]]
            graph_time = mod.run_individual(number=10, repeat=1, min_repeat_ms=5000)

            print("Running time of each layer:")
            print("---------------------------")
            print("|graph_nodes| = ", len(graph_nodes))
            print("|graph_time| = ", len(graph_time))

            for k, v in zip(graph_nodes, graph_time):
                print(k, float(v) * 1e6, "us")
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"Run module f_time_per_layer via RPC failed, exception: {exc}",
            )

    return f_time_per_layer
