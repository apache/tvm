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
"""Performance debug tool for dynamic shape workloads"""

from typing import List, Dict, Union, Tuple, Iterator
from pathlib import Path

import cloudpickle

import tvm
from tvm import relax
from .utils import default_dym_var_sample_func, get_func_name_from_gv

SKETCH = """import pickle

import tvm
from tvm import relax
from tvm.script import tir as T

from mlc_bench.benchmark import MLCBench

MODEL_NAME = "{model_name}"
RELAX_FUNC_NAME = "{relax_func_name}"
PRIM_FUNC_NAME = "{prim_func_name}"
FUNC_HASH = {func_hash}
WEIGHT = {weight}
CAT = {category}
SAMPLE_NUMBER = {sample_number}

INPUT_ARGS = pickle.loads({input_args})
DYM_VAR_SAMPLE_FUNC = pickle.loads({dym_var_sample_func})
DYM_VAR_DICT = pickle.loads({dym_var_dict})

{func_script}

if __name__ == "__main__":
    bench = MLCBench()
    target = tvm.target.Target("{target}")
    dev = {dev}
    print("Input args:", INPUT_ARGS)
    for _ in range(SAMPLE_NUMBER):
        dym_var_sample = DYM_VAR_SAMPLE_FUNC(DYM_VAR_DICT)
        input_infos, median, std = bench.benchmark(
            main,
            INPUT_ARGS,
            dym_var_sample=dym_var_sample,
            target=target,
            dev=dev,
        )
        bench.record(
            {{
                "RelaxFunc": RELAX_FUNC_NAME,
                "PrimFunc": PRIM_FUNC_NAME,
                "InputInfo": ", ".join(
                    [f"{{k}} = {{v}}" for k, v in dym_var_sample.items()]
                ),
                "Time(us)": median*1e6,
                "Std(us)": std*1e6,
                "Weight": WEIGHT,
                "WxTime(ms)": WEIGHT*median*1e3,
            }}
        )
    bench.show()
"""


def extract_shape(
    arg: Union[Tuple, List, relax.Tuple, relax.ShapeStructInfo]
) -> List[relax.ShapeStructInfo]:
    """Extract shape information from a relax argument.

    Parameters
    ----------
    arg : Union[Tuple, List, relax.Tuple, relax.ShapeStructInfo]
        The relax argument to be extracted.

    Returns
    -------
    result : List[relax.ShapeStructInfo]
        The extracted shape information.
    """
    if isinstance(arg, (tuple, list, tvm.relax.Tuple)):
        results = []
        for sub_arg in arg:
            results.extend(extract_shape(sub_arg))
        return results
    else:
        return [arg.struct_info]


def prim_func_usage_gen(
    mod: tvm.ir.IRModule,
) -> Iterator[Tuple[tvm.ir.GlobalVar, tvm.ir.GlobalVar, List[relax.ShapeStructInfo]]]:
    """Generate the usage of prim functions in a relax module.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The relax module to be analyzed.

    Yields
    ------
    result : Tuple[tvm.ir.GlobalVar, tvm.ir.GlobalVar, List[relax.ShapeStructInfo]]
        The usage of prim functions in a relax module.
    """
    for gv, func in mod.functions.items():  # pylint: disable=invalid-name
        if isinstance(func, tvm.relax.Function):
            for block in func.body.blocks:
                for binding in block.bindings:
                    if isinstance(binding.value, tvm.relax.expr.Call):
                        raw_args = binding.value.args
                        functor = raw_args[0]
                        if isinstance(functor, tvm.ir.GlobalVar):
                            if isinstance(mod.functions[functor], tvm.tir.PrimFunc):
                                args = extract_shape(raw_args[1:]) + extract_shape(binding.value)
                                yield gv, functor, args


def extract_dynamic_var(
    func_dict: Dict,
) -> Dict[tvm.ir.GlobalVar, Dict[str, str]]:
    """Extract dynamic shape variables from a relax function dictionary."""
    dym_var_dict: Dict[tvm.ir.GlobalVar, Dict[str, str]] = {}
    for gv in func_dict:  # pylint: disable=invalid-name
        dym_var_dict[gv] = {}
        for functor in func_dict[gv]:
            for arg_list, _ in func_dict[gv][functor]:
                for arg in arg_list:
                    if isinstance(arg, tvm.relax.TensorStructInfo):
                        for v in arg.shape.values:
                            if isinstance(v, tvm.tir.Var):
                                dym_var_dict[gv][str(v)] = v.dtype
                    elif isinstance(arg, tvm.relax.ShapeStructInfo):
                        for v in arg.values:
                            if isinstance(v, tvm.tir.Var):
                                dym_var_dict[gv][str(v)] = v.dtype
                    else:
                        raise NotImplementedError
    return dym_var_dict


def extract_func_info(
    mod: tvm.ir.IRModule,
) -> Tuple[
    Dict[tvm.ir.GlobalVar, Dict[tvm.ir.GlobalVar, List[Tuple[List, int]]]],
    Dict[tvm.ir.GlobalVar, Dict[str, str]],
]:
    """Extract function information from a relax module."""

    def update_records(records, new_args):
        for i, (args, count) in enumerate(records):
            if new_args == args:
                records[i] = (args, count + 1)
                return
        records.append((new_args, 1))

    relax_func_dict: Dict[tvm.ir.GlobalVar, Dict[tvm.ir.GlobalVar, List[Tuple[List, int]]]] = {}
    for gv, functor, args in prim_func_usage_gen(mod):  # pylint: disable=invalid-name
        if isinstance(functor, tvm.ir.GlobalVar):
            if not gv in relax_func_dict:
                relax_func_dict[gv] = {}
            if not functor in relax_func_dict[gv]:
                relax_func_dict[gv][functor] = []
            update_records(relax_func_dict[gv][functor], args)

    dym_var_dict = extract_dynamic_var(relax_func_dict)
    return relax_func_dict, dym_var_dict


def extract_prim_func(
    model_name: str,
    relax_func_name: str,
    prim_func_name: str,
    func: tvm.tir.PrimFunc,
    func_args: List[Tuple[tvm.relax.expr.Call, str]],
    dym_var_dict: Dict[str, str],
    weight: int,
    file_path: str,
    sample_number: int = 5,
    target: Union[str, tvm.target.Target] = "llvm -num-cores=4",
) -> None:
    """Extract a self-contained PrimFunc test file from a Relax module.

    Parameters
    ----------
    model_name: str
        The name of the model.
    relax_func_name: str
        The name of the Relax function.
    prim_func_name: str
        The name of the prim function.
    func: tvm.tir.PrimFunc
        The name of the prim function to be extracted.
    func_args: List[Tuple[tvm.relax.expr.Call, str]]
        The arguments of the prim function, including both static and dynamic shape arguments.
        Given in format [ ..., ((1, n, 128), "float32"), ... ].
    dym_var_dict: Dict[str, str]
        The dictionary of dynamic shape variables. Given in format {"n": "int32", "m": "int32"}.
    weight: int
        The weight of the prim function.
    file_path: str
        The path to store the extracted file.
    sample_number: int
        The number of times to sample dynamic shape variables.
    """
    if isinstance(target, str):
        target_str = target
        target = tvm.target.Target(target)
    elif isinstance(target, tvm.target.Target):
        target_str = str(target)
    else:
        raise TypeError("Unsupported target type: " + str(type(target)))

    if target.kind.name == "cuda":
        dev_str = "tvm.cuda()"
    elif target.kind.name == "llvm":
        dev_str = "tvm.cpu()"
    else:
        raise NotImplementedError("Only support cuda and llvm runtime device.")

    file = open(file_path, "w")

    print(
        SKETCH.format(
            **{
                "model_name": model_name,
                "relax_func_name": relax_func_name,
                "prim_func_name": prim_func_name,
                "func_hash": tvm.ir.structural_hash(func),
                "weight": weight,
                "sample_number": sample_number,
                "dym_var_dict": cloudpickle.dumps(dym_var_dict),
                "input_args": cloudpickle.dumps(func_args),
                "dym_var_sample_func": cloudpickle.dumps(default_dym_var_sample_func),
                "func_script": func.script(),
                "target": target_str,
                "dev": dev_str,
            }
        ),
        file=file,
    )


def extract_from_relax(mod: tvm.ir.IRModule, model_name: str, file_path: str) -> None:
    """Extract self-contained PrimFunc test files from a Relax module.

    Parameters
    ----------
    mod: tvm.ir.IRModule
        The Relax module to be extracted.
    model_name: str
        The name of the model.
    file_path: str
        The path to store the extracted files.
    """
    relax_funcs, dym_var_dict = extract_func_info(mod)
    Path(file_path).mkdir(parents=True, exist_ok=True)
    for relax_func_gv in relax_funcs:
        relax_func_name = get_func_name_from_gv(relax_func_gv)
        for prim_func_gv in relax_funcs[relax_func_gv]:
            prim_func_name = get_func_name_from_gv(prim_func_gv)
            for func_args, weight in relax_funcs[relax_func_gv][prim_func_gv]:
                extract_prim_func(
                    model_name=model_name,
                    relax_func_name=relax_func_name,
                    prim_func_name=prim_func_name,
                    func=mod[prim_func_gv],
                    dym_var_dict=dym_var_dict[relax_func_gv],
                    func_args=func_args,
                    weight=weight,
                    file_path=f"{file_path}/{relax_func_name}_{prim_func_name}.py",
                )
