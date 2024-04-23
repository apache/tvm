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

"""Tuning capabilities for external BYOC runtime"""

import tvm
from tvm import relax

from typing import List, Dict, Any

from . import _ffi_api


@tvm._ffi.register_object("relax.backend.contrib.AlgoDataBase")
class AlgoDatabase(tvm.runtime.Object):
    """Database with codegen specific algo objects"""

    def __enter__(self) -> "AlgoDatabase":
        """Entering the scope of the context manager"""
        _ffi_api.AlgoDatabaseEnterWithScope(self)
        return self

    def __exit__(self, ptype, value, trace) -> None:
        """Exiting the scope of the context manager"""
        _ffi_api.AlgoDatabaseExitWithScope(self)

    def to_json(self) -> str:
        """Serialize to json format"""
        return _ffi_api.AlgoDatabaseToJSON(self)

    @staticmethod
    def from_json(json: str) -> "AlgoDatabase":
        return _ffi_api.AlgoDatabaseFromJSON(json)


def ExtractTuningTasks(mod: tvm.IRModule, codegen_name: str) -> List[relax.Function]:
    """
    Extract algo tuning tasks from provided IR module.

    Parameters
    ----------
    mod: tvm.IRModule
        The IRModule to be parsed.

    codegen_name: str
        The name of codegen target to looking for.

    Returns
    -------
    tasks: list[Function]
        List of composite functions assigned to specified codegen target.
    """
    codegen_funcs = []
    for _, func in mod.functions_items():
        if "Codegen" in func.attrs and func.attrs["Codegen"] == codegen_name:
            codegen_funcs.append(func)

    composite_funcs = []
    @relax.expr_functor.visitor
    class TaskExtractorlVisitor(relax.expr_functor.PyExprVisitor):
        def visit_var_binding_(self, binding: relax.VarBinding) -> None:
            if isinstance(binding.value, relax.Function) and "Composite" in binding.value.attrs:
                composite_funcs.append(binding.value)
    
    txv = TaskExtractorlVisitor()
    for f in codegen_funcs:
        txv.visit_expr(f)

    return composite_funcs


def TuneTasks(tasks: List[relax.Function], codegen_name: str, cfg_map: Dict[str, Any] = {}) -> AlgoDatabase:
    """
    Tune tasks for specific codegen target.

    Performe kernel benchmarking with algo from search space and select best one for  
    each particular value of dynamic dimension.

    Parameters
    ----------
    tasks: list(relax.Function)
        The list of tasks to tune.

    codegen_name: str
        The name of codegen target to tune.
    
    cfg_map: Dict[str, Any]
        Dictionary with tuning specific configuration parameters.

        List of supported configuration params:
        - "dyn_m_range" | "dyn_m_step" | "dyn_m_offset" : int
            In case of tuning kernels with dynamic shapes thsi parameters 
            defines the region of dynamic dimension values to benchmark.
        
        - "num_repeats" | "repeat_size" : int
            Define procedure of benchmarking of one single algo candidate. Benchmark kernel 
            repeat_size times and get average duration. Repeat it num_repeats times and get
            median value.

        - "treshold_percent" : float
            Do not fully benchmark algo candidate which slower than reference time of execution
            on treshold_percent.
            Default value: 10% 
        
        - "mode" : str
            Define mode of generation of searchspace. Supported values "complete_search"
            and "heuristic_top1".
            Dfault: "complete_search"
        
        - "verbose" : bool
            To print tuning process information.

    Returns
    -------
    db: AlgoDatabase
        Resulting database with best algo object. 
    """
    tune_func_name = "contrib." + codegen_name + ".TuneAlgoTasks"
    tune_func = tvm.get_global_func(tune_func_name, allow_missing=True)

    if tune_func is None:
        print("WARNING. Is not able to tune ", codegen_name, " codegen. Tuning function is not implemented.")
        return

    db = tune_func(tasks, cfg_map)
    return db


def TuneCodegenAlgo(mod: tvm.IRModule, codegen_name, cfg_map: Dict[str, Any] = {}) -> AlgoDatabase:
    """
    Tune codegen kernels available in provided IR module.
    
    Parameters
    ----------
    mod: tvm.IRModule
        The IRModule to be tuned.
    
    codegen_name: str
        The name of codegen target to tune for.

    cfg_map: Dict[str, Any]
        Additional configuration parameters. Availabel options see in TuneTasks.

    Returns
    -------
    db: AlgoDatabase
        Resulting database object. 
    """
    tasks = ExtractTuningTasks(mod, codegen_name)
    db = TuneTasks(tasks, codegen_name, cfg_map)
    return db
