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

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cloudpickle

import tvm
from tvm import relax

from .utils import default_dym_var_sample_func, get_func_name_from_gv

SKETCH = """import pickle

import tvm
from tvm import relax
from tvm.script import tir as T

from tvm.dlight.benchmark import benchmark_prim_func

MODEL_NAME = "{model_name}"
RELAX_FUNC_NAME = "{relax_func_name}"
PRIM_FUNC_NAME = "{prim_func_name}"
FUNC_HASH = {func_hash}
WEIGHT = {weight}
SAMPLE_NUMBER = {sample_number}

DYM_VAR_SAMPLE_FUNC = {dym_var_sample_func}

# None means extract from PrimFunc
INPUT_ARGS = {input_args}
DYM_VAR_DICT = {dym_var_dict}

{func_script}

if __name__ == "__main__":
    target = tvm.target.Target("{target}")
    benchmark_prim_func(
        main,
        args = INPUT_ARGS,
        dym_var_dict = DYM_VAR_DICT,
        dym_var_sample_func = DYM_VAR_SAMPLE_FUNC,
        sample_number = SAMPLE_NUMBER,
        target = target,
        weight = WEIGHT,
        relax_func_name = RELAX_FUNC_NAME,
        prim_func_name = PRIM_FUNC_NAME,
    )
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
    return [arg.struct_info]


def extract_dynamic_var(
    func_dict: Dict[
        tvm.ir.GlobalVar,
        Dict[
            tvm.ir.GlobalVar,
            List[Tuple[List, int]],
        ],
    ],
) -> Dict[tvm.ir.GlobalVar, Dict[str, str]]:
    """Extract dynamic shape variables from a relax function dictionary.

    Parameters
    ----------
    func_dict : Dict[
        tvm.ir.GlobalVar,
        Dict[
            tvm.ir.GlobalVar,
            List[Tuple[List, int]],
        ],
        The relax function dictionary, containing the input arguments' shape information of each
        PrimFunc in a Relax function.

    Returns
    -------
    result : Dict[tvm.ir.GlobalVar, Dict[str, str]]
        The dictionary of dynamic shape variables. Given in format {"n": "int32", "m": "int32"}.
    """
    dym_var_dict: Dict[tvm.ir.GlobalVar, Dict[str, str]] = {}
    for gv in func_dict:  # pylint: disable=invalid-name,too-many-nested-blocks
        dym_var_dict[gv] = {}
        for functor in func_dict[gv]:
            for arg_list, _ in func_dict[gv][functor]:
                flattened_arg_list = []
                for arg in arg_list:
                    if isinstance(arg, relax.TupleStructInfo):
                        flattened_arg_list.extend(arg.fields)
                    else:
                        flattened_arg_list.append(arg)
                for arg in flattened_arg_list:
                    if isinstance(arg, relax.TensorStructInfo):
                        for val in arg.shape.values:
                            if isinstance(val, tvm.tir.Var):
                                dym_var_dict[gv][str(val)] = val.dtype
                    elif isinstance(arg, relax.ShapeStructInfo):
                        for val in arg.values:
                            if isinstance(val, tvm.tir.Var):
                                dym_var_dict[gv][str(val)] = val.dtype
                    else:
                        raise NotImplementedError
    return dym_var_dict


def update_records(
    records: Dict[List[relax.ShapeStructInfo], int], new_args: List[relax.ShapeStructInfo]
) -> None:
    """Update the count of a function input argument config.

    Parameters
    ----------
    records : Dict[List[relax.ShapeStructInfo], int]
        The dictionary to count how many times a function input argument config appears.
    new_args : List[relax.ShapeStructInfo]
        The new input argument config.
    """
    for i, (args, count) in enumerate(records):
        if new_args == args:
            records[i] = (args, count + 1)
            return
    records.append((new_args, 1))


def extract_func_info_from_prim_func(
    func: tvm.tir.PrimFunc,
) -> Tuple[List[Tuple[Tuple[Union[tvm.tir.Var, int], ...], str]], Dict[str, str]]:
    """Extract function input information from a PrimFunc.

    Parameters
    ----------
    func : tvm.tir.PrimFunc
        The PrimFunc to be analyzed.

    Returns
    -------
    result : Tuple[
        List[Tuple[Tuple[Union[tvm.tir.Var, int], ...], str]],
        Dict[str, str],
    ]
        The function input information and dynamic shape variable dictionary.
    """
    func_args = []
    dym_var = {}
    for param in func.params:
        buffer = func.buffer_map[param]
        shape = []
        for dim in buffer.shape:
            if isinstance(dim, tvm.tir.IntImm):
                shape.append(dim.value)
            elif isinstance(dim, tvm.tir.Var):
                dym_var[str(dim)] = str(dim.dtype)
                shape.append(dim)
            else:
                raise ValueError(f"Unknown shape: {buffer.shape}")
        func_args.append((tuple(shape), str(buffer.dtype)))
    return func_args, dym_var


def extract_all_func_info_from_relax(
    mod: tvm.ir.IRModule,
) -> Tuple[
    Dict[tvm.ir.GlobalVar, Dict[tvm.ir.GlobalVar, List[Tuple[List, int]]]],
    Dict[tvm.ir.GlobalVar, Dict[str, str]],
]:
    """Extract function input information from a relax module.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The Relax module to be analyzed.

    Returns
    -------
    result : Tuple[
        Dict[tvm.ir.GlobalVar, Dict[tvm.ir.GlobalVar, List[Tuple[List, int]]]],
        Dict[tvm.ir.GlobalVar, Dict[str, str]],
    ]
        The function input information and dynamic shape variable dictionary.
    """
    relax_func_dict: Dict[tvm.ir.GlobalVar, Dict[tvm.ir.GlobalVar, List[Tuple[List, int]]]] = {}
    for gv, func in mod.functions_items():  # pylint: disable=invalid-name,too-many-nested-blocks
        if isinstance(func, tvm.relax.Function):
            for block in func.body.blocks:
                for binding in block.bindings:
                    if isinstance(binding.value, tvm.relax.expr.Call):
                        raw_args = binding.value.args
                        functor = raw_args[0]
                        if isinstance(functor, tvm.ir.GlobalVar) and isinstance(
                            mod.functions[functor], tvm.tir.PrimFunc
                        ):
                            args = extract_shape(raw_args[1:]) + extract_shape(binding.value)
                            if isinstance(functor, tvm.ir.GlobalVar):
                                if not gv in relax_func_dict:
                                    relax_func_dict[gv] = {}
                                if not functor in relax_func_dict[gv]:
                                    relax_func_dict[gv][functor] = []
                                update_records(relax_func_dict[gv][functor], args)

    return relax_func_dict, extract_dynamic_var(relax_func_dict)


def extract_prim_func(  # pylint: disable=too-many-arguments
    model_name: str,
    relax_func_name: str,
    prim_func_name: str,
    func: tvm.tir.PrimFunc,
    *,
    func_args: Optional[List[Tuple[Tuple[Union[tvm.relax.expr.Call, int], ...], str]]] = None,
    dym_var_dict: Optional[Dict[str, str]] = None,
    weight: int = 1,
    sample_number: int = 5,
    target: Optional[Union[str, tvm.target.Target]] = None,
) -> str:
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
        The PrimFunc to be extracted.
    func_args: Optional[List[Tuple[Tuple[Union[tvm.relax.expr.Call, int], ...], str]]]
        The arguments of the prim function, including both static and dynamic shape arguments.
        Given in format [ ..., ((1, n, 128), "float32"), ... ].
        If not given, the arguments will be extracted from the PrimFunc.
    dym_var_dict: Optional[Dict[str, str]]
        The dictionary of dynamic shape variables. Given in format {"n": "int32", "m": "int32"}.
        If not given, the dictionary will be extracted from the PrimFunc.
    weight: int
        The weight of the prim function, by default 1.
    sample_number: int
        The number of times to sample dynamic shape variables, by default 5.
    target: Optional[Union[str, tvm.target.Target]]
        The target device to run the PrimFunc. If None, will use target from the context.

    Returns
    -------
    result : str
        The extracted PrimFunc test file content.
    """
    if target is None:
        target = tvm.target.Target.current()
        target_str = str(target)
        if target is None:
            raise ValueError("Target is not specified.")
    elif isinstance(target, str):
        target_str = target
        target = tvm.target.Target(target)
    elif isinstance(target, tvm.target.Target):
        target_str = str(target)
    else:
        raise TypeError("Unsupported target type: " + str(type(target)))

    return SKETCH.format(
        **{
            "model_name": model_name,
            "relax_func_name": relax_func_name,
            "prim_func_name": prim_func_name,
            "func_hash": tvm.ir.structural_hash(func),
            "weight": weight,
            "sample_number": sample_number,
            "dym_var_dict": f"pickle.loads({cloudpickle.dumps(dym_var_dict)})"
            if dym_var_dict is not None
            else "None",
            "input_args": f"pickle.loads({cloudpickle.dumps(func_args)})" if func_args else "None",
            "dym_var_sample_func": "pickle.loads("
            + f"{cloudpickle.dumps(default_dym_var_sample_func)}"
            + ")",
            "func_script": func.script(),
            "target": target_str,
        }
    )


def extract_from_relax(
    mod: tvm.ir.IRModule,
    model_name: str,
    file_path: str,
    target: Optional[Union[str, tvm.target.Target]] = None,
) -> None:
    """Extract self-contained PrimFunc test files from a Relax module.

    Parameters
    ----------
    mod: tvm.ir.IRModule
        The Relax module to be extracted.
    model_name: str
        The name of the model.
    file_path: str
        The path to store the extracted files.
    target: Optional[Union[str, tvm.target.Target]]
        The target device to run the PrimFunc. If None, will use target from the context.
    """
    relax_funcs, dym_var_dict = extract_all_func_info_from_relax(mod)
    Path(file_path).mkdir(parents=True, exist_ok=True)
    for relax_func_gv in relax_funcs:  # pylint: disable=consider-using-dict-items
        relax_func_name = get_func_name_from_gv(relax_func_gv)
        for prim_func_gv in relax_funcs[relax_func_gv]:
            prim_func_name = get_func_name_from_gv(prim_func_gv)
            for func_args, weight in relax_funcs[relax_func_gv][prim_func_gv]:
                with open(
                    f"{file_path}/{relax_func_name}_{prim_func_name}.py", "w", encoding="utf-8"
                ) as file:
                    print(
                        extract_prim_func(
                            model_name=model_name,
                            relax_func_name=relax_func_name,
                            prim_func_name=prim_func_name,
                            func=mod[prim_func_gv],
                            dym_var_dict=dym_var_dict[relax_func_gv],
                            func_args=func_args,
                            weight=weight,
                            target=target,
                        ),
                        file=file,
                    )
