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
from statistics import median
from typing import Callable, Dict, Optional, Union, List
import json
import numpy as np  # type: ignore

import tvm
from tvm.ir import IRModule
from tvm.relay import Function as RelayFunc
from tvm.runtime import NDArray
from tvm.target import Target
from tvm.tir import Schedule
from tvm import meta_schedule as ms


def apply_fixed_schedules(
    relay_mod: Union[RelayFunc, IRModule],
    target: Union[str, Target],
    params: Optional[Dict[str, NDArray]],
    schedule_fn: Callable[[ms.ExtractedTask, Schedule], bool],
    te_filter_func=None,
):
    """Apply fixed schedules (manually written, without any tunable knobs) as specified by
    schedule_fn to extracted tasks, and return a database that can be passed to ApplyHistoryBest.

    Parameters
    ----------
    mod : Union[RelayFunc, IRModule]
        The Relay module to apply fixed schedules.
    target : Union[str, Target]
        The target used to extract tasks.
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        The associated parameters of the module.
    schedule_fn : Callable[[ExtractedTask, Schedule], bool]
        A callable that is applied for each extracted task and the corresponding default schedule.
        Returns True if the given schedule should be committed to the database, False otherwise.
    te_filter_func : Union[str, None, Callable[[List[Tensor]], PrimFunc]] = None
        The filtering function for TE computation
        If it's a string, it's the name of the filtering function. Built in functions are
          - "meta_schedule.DefaultTaskFilter"
          - "meta_schedule.DefaultTaskFilterAllowExtern"
        If it's None, it's the default filtering function
        If it's a callable, it's the filtering function

    Returns
    -------
    database : Database
        The database containing dummy tuning records for manually scheduled traces.
    """
    target = Target(target) if isinstance(target, str) else target
    extracted_tasks = ms.extract_task_from_relay(
        relay_mod,
        target,
        params,
        te_filter_func=te_filter_func,
    )
    database = ms.database.MemoryDatabase()
    for task in extracted_tasks:
        mod = ms.default_config.mod(task.dispatched[0])
        sch = Schedule(mod)

        if schedule_fn(task, sch):
            workload = database.commit_workload(mod)
            tune_rec = ms.database.TuningRecord(sch.trace, workload, [0.0], target, [])
            database.commit_tuning_record(tune_rec)

    return database


def generate_input_data(input_shape: List[int], input_dtype: str) -> np.ndarray:
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
    if input_dtype in ["uint8", "int8"]:
        return np.random.randint(
            low=0,
            high=127,
            size=input_shape,
            dtype="int32",  # TODO(zxybazh): fix the datatype when int8 / uint8 is supported better
        )
    if input_dtype in ["int32", "int64"]:
        return np.random.randint(low=0, high=10000, size=input_shape, dtype=input_dtype)
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
