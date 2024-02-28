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
import tvm
import os
from tvm.contrib.popen_pool import PopenPoolExecutor, StatusKind, MapResult
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import List, Tuple, Optional, Dict
from tvm import tir, IRModule
from tvm.runtime import Module
from tvm.tir import Schedule
from tvm import dlight as dl
from .analysis import get_root_block, get_reduction_blocks, find_var_from_func
from .roller.arch import Arch
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.gpu.matmul_analysis import get_tensorized_func_and_tags
from ..base.roller.rasterization import NoRasterization
import tempfile
import re
import itertools
from tvm.ir.supply import GlobalVarSupply


def match_global_kernel(source: str) -> int:
    pattern = r"__global__\s+void\s+[__launch_bounds__\(\d+\)\s+]\w+"
    matched = re.findall(pattern, source)
    assert len(matched) > 1  # may have statement before kernel
    return source.index(matched[0])


def get_rasterization_code(pannel_width: int = 8) -> str:
    return f"""
        const int MAX_BLOCK_N = {pannel_width};
        const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
        const auto totalPanel = (gridDim.x * gridDim.y +MAX_BLOCK_N * gridDim.x - 1) / (MAX_BLOCK_N * gridDim.x);
        const auto totalBlock = gridDim.x * gridDim.y;
        const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N *gridDim.x);
        const auto strideLd = panelIdx + 1 < totalPanel ?MAX_BLOCK_N : (totalBlock - panelIdx * (MAX_BLOCK_N *gridDim.x)) / gridDim.x;
        const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) / strideLd;
        const auto by = (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) % strideLd + panelIdx * MAX_BLOCK_N;
        const auto bz = blockIdx.z;
        const dim3 blockIdx(bx, by, bz);
    """


class CompileResult:
    """
    Class to store the result of compilation
    """

    def __init__(self, config, sch, mod: Module):
        self.config = config
        self.sch = sch
        self.mod = mod
        self.code = mod.imported_modules[0].get_source() if mod else None
        self.latency = 1e9
        self.profile_tensors = []
        self.time_evaluator = None

    def profile(self):
        return self.time_evaluator(*self.profile_tensors).mean


def _apply_config(
    func: tir.PrimFunc,
    config=None,  # todo(lei): update typing
) -> Optional[List[tir.Schedule]]:
    """
    find rules:
    case 1. if the main block has no reduce op, then use the Elementwise rule.
    case 2. if the config enabled tensorcore, then use the TensorCore rule.
    case 3. if any([t > 1 for t in config.reduce_thread]), we should use the InnerThread Reduction Rule.
    case 4. else we should use general reduction rule.
    """
    print("[FastDlight] Apply config ", config)

    sch = tir.Schedule(func)
    root_block = get_root_block(sch)
    blocks = sch.get_child_blocks(root_block)
    reduction_blocks = get_reduction_blocks(sch, blocks)
    try:
        if not reduction_blocks:
            return dl.gpu.ElementWise().apply_config(func, config)
        elif config.use_tc:
            if config.arch.sm_version >= 80:
                # For A100(sm_80) or more advanced gpu, use MMA tensorization.
                return dl.gpu.MatmulTensorizationMMA().apply_config(func, config)
            else:
                # For other GPUs, use WMMA tensorization.
                return dl.gpu.MatmulTensorizationWMMA().apply_config(func, config)
        else:
            _reduction_rules = []

            _reduction_rules.append(dl.gpu.GEMV())
            if not any([t > 1 for t in config.reduce_thread]):
                # Matrix multiplication template doesn't support inner thread reduction
                _reduction_rules.append(dl.gpu.Matmul())
            _reduction_rules.append(dl.gpu.GeneralReduction())

            for rule in _reduction_rules:
                try:
                    sch = rule.apply_config(func, config)
                except:
                    continue
                if sch is not None:
                    return sch
    except Exception as e_msg:
        print("[FastDlight] Apply config failed: ", e_msg)
    return None


def apply_and_build_parallel(func, configs, arch, num_repeats=5, max_workers=10) -> CompileResult:
    cpresults = []

    def var_warpper(v):
        if isinstance(v, tvm.tir.Var):
            assert "opt_shapes" in func.attrs
            assert v.name in func.attrs["opt_shapes"]
            return func.attrs["opt_shapes"][v.name].value
        elif isinstance(v, tvm.tir.IntImm):
            return v.value
        else:
            raise RuntimeError("Not supported type: ", type(v))

    profile_tensors = []
    for param in func.params:
        if param not in func.buffer_map:
            # in case of dynamic symbolic may in params
            continue
        arg = func.buffer_map[param]
        if arg.dtype == "int8":
            profile_tensors.append(
                tvm.nd.array(
                    np.random.randint(-127, 128, [var_warpper(i) for i in arg.shape]).astype(
                        arg.dtype
                    ),
                    device=arch.device,
                )
            )
        else:
            profile_tensors.append(
                tvm.nd.array(
                    np.random.uniform(0, 1, [var_warpper(i) for i in arg.shape]).astype(arg.dtype),
                    device=arch.device,
                )
            )

    max_workers = min(len(configs), os.cpu_count(), max_workers)

    # apply config in thread parallel
    _sched: List[Schedule] = []
    with ThreadPoolExecutor(max_workers=4) as schduler:
        futures = {
            schduler.submit(lambda f, c: _apply_config(f, c), func, config) for config in configs
        }
        for future in as_completed(futures):
            if future.result() is not None:
                _sched.append(future.result())

    builder = PopenPoolExecutor(max_workers=max_workers)

    # build in process parallel
    def _build(context) -> str:
        idx, mod, arch = context

        # TODO(lei):
        # this is a trick to implement rasteration, will be removed in the future
        # config = configs[idx]
        # @tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
        # def tvm_callback_cuda_postproc(code, _):
        #     index = code.index("{", match_global_kernel(code))
        #     if not isinstance(config.rasterization_plan, NoRasterization):
        #         factor = config.rasterization_plan.panel_width_
        #         rasterization_code = get_rasterization_code(factor)
        #         code = code[: index + 2] + rasterization_code + code[index + 2 :]
        #     return code

        with tvm.transform.PassContext(config={"tir.use_async_copy": True}):
            rt_mod = tvm.build(mod["main"], target=arch.target)

        from tvm.contrib.tar import tar  # pylint: disable=import-outside-toplevel

        artifact_path = os.path.join(tempfile.mkdtemp(), "tvm_tmp_mod." + tar.output_format)
        code = rt_mod.imported_modules[0].get_source()
        rt_mod.export_library(artifact_path, fcompile=tar)
        return idx, code, artifact_path

    for map_result in builder.map_with_error_catching(
        _build,
        [(i, sch.mod, arch) for i, sch in enumerate(_sched)],
    ):
        if map_result.status == StatusKind.TIMEOUT:
            print("[FastDlight] LocalBuilder: Timeout")
        elif map_result.status == StatusKind.EXCEPTION:
            # TODO(lei): redirect the exception to file if needed
            print("[FastDlight] LocalBuilder: An exception occurred ")
            continue
        elif map_result.status == StatusKind.COMPLETE:
            idx, code, artifact_path = map_result.value
            assert artifact_path is not None, "artifact_path is None"

            sch = _sched[idx]
            config = configs[idx]
            rt_mod = tvm.runtime.load_module(artifact_path)
            cpresult = CompileResult(config, sch, rt_mod)
            timer_cuda_mod = rt_mod.time_evaluator(
                rt_mod.entry_name, arch.device, number=num_repeats
            )
            cpresult.profile_tensors = profile_tensors
            cpresult.time_evaluator = timer_cuda_mod
            cpresult.code = code
            cpresults.append(cpresult)
        else:
            raise ValueError(f"Unreachable: unexpected result: {map_result}")

    del builder

    best = None
    best_latency = 1e9
    for cpresult in cpresults:
        config = cpresult.config
        try:
            latency = cpresult.profile()
        except Exception as e_mesg:
            print("[FastDlight] Evaluation with config failed: ", e_mesg)
            continue
        print("[FastDlight] Evaluation with config ", config)
        print("[FastDlight] Time cost of this config: {:.3f} ms".format(latency * 1e3))

        cpresult.latency = latency
        if latency < best_latency:
            best_latency = latency
            best = cpresult

    return cpresults, best


def apply_and_build(
    func,
    configs,
    arch,
    parallel_build=False,
) -> Tuple[List[CompileResult], CompileResult]:
    max_workers = 10 if parallel_build else 1
    return apply_and_build_parallel(func, configs, arch, max_workers)


def fast_tune(
    func: tir.PrimFunc,
    target: tvm.target.Target,
    topk: int = 10,
    parallel_build: bool = True,
):
    if target.kind.name != "cuda":
        print("[FastDlight] Only support CUDA target")
        return None, None
    specilized_func = func
    if func.attrs is not None and "opt_shapes" in func.attrs:
        opt_shapes = func.attrs["opt_shapes"]
        # should be int value
        if not all([isinstance(v.value, int) for v in opt_shapes.values()]):
            print("[FastDlight] The opt_shapes should be int value")
            return None, None
        # currently only support one dynmaic range
        if len(opt_shapes) > 1:
            print("[FastDlight] Currently only support one dynamic range")
            return None, None

        for buffer in func.buffer_map.values():
            for axis in buffer.shape:
                if isinstance(axis, tvm.tir.Var):
                    if axis.name not in opt_shapes:
                        raise NotImplementedError(
                            "Currently do not support fast tune with none-dynamic range set"
                        )
        if opt_shapes:
            for name, shape in opt_shapes.items():
                var = find_var_from_func(func, name)
                specilized_func = func.specialize({var: shape.astype(var.dtype)}).with_attr(
                    "is_specialized"
                )

    arch = CUDA(target)

    policy = DefaultPolicy(func=func, arch=arch)
    try:
        specilized_func, tags = get_tensorized_func_and_tags(specilized_func, arch.target)
    except:
        tags = None
    if tags:
        policy = TensorCorePolicy(func=specilized_func, arch=arch, tags=tags)

    configs = policy.emit_config(topk)
    cpresults, best = apply_and_build(func, configs, arch, parallel_build=parallel_build)

    return cpresults, best


# always use the first function as the base
def collect_buffers_to_declare(func):
    params = []
    # collect dynamic symbolic
    dyn_symbolic: List[tvm.tir.Var] = []
    buffers_to_declare = []
    for param in func.params:
        if param not in func.buffer_map:
            continue
        buffer = func.buffer_map[param]
        for axis in buffer.shape:
            if isinstance(axis, tvm.tir.Var) and axis not in dyn_symbolic:
                dyn_symbolic.append(axis)
        buffers_to_declare.append(buffer)
        params.append(buffer.data)

    # the args should be buffers + dynamic symbolic
    params += list(dyn_symbolic)

    return params, buffers_to_declare


def refactor_specialized_func(g_var, func, params, buffers_to_declare):
    body = func.body
    attrs = func.attrs
    global_symbol = g_var
    if func.attrs is not None and "opt_shapes" in func.attrs:
        opt_shapes = func.attrs["opt_shapes"]

    def serialize_name(opt_shapes: Dict):
        return "_opt_" + "_".join([f"{k}_{v}" for k, v in opt_shapes.items()])

    global_symbol += serialize_name(opt_shapes)
    ret_type = func.ret_type
    for buf in buffers_to_declare:
        body = tvm.tir.DeclBuffer(buf, body=body)

    # devide func must be private
    device_func = tvm.tir.PrimFunc(params, body, ret_type, attrs=attrs).without_attr(
        "global_symbol"
    )
    return global_symbol, device_func


def create_dispatch_func(g_var: str, func: tir.PrimFunc, refactored_funcs: List[str]):
    global_symbol = g_var
    attrs = func.attrs
    buffer_map = func.buffer_map
    params = func.params
    ret_type = func.ret_type

    # collect dynamic symbolic
    dyn_symbolic: List[tvm.tir.Var] = []
    _invoke_params = []
    for param in func.params:
        if param not in func.buffer_map:
            continue
        buffer = func.buffer_map[param]
        for axis in buffer.shape:
            if isinstance(axis, tvm.tir.Var) and axis not in dyn_symbolic:
                dyn_symbolic.append(axis)
        _invoke_params.append(buffer.data)
    _invoke_params += list(dyn_symbolic)

    func_range: List[int] = []
    global_symbols = []
    for g_var, refactor_func in refactored_funcs:
        opt_shapes = refactor_func.attrs["opt_shapes"]
        func_range.append(list(opt_shapes.values())[0])
        global_symbols.append(g_var)

    # TODO(lei): general the dispatch function to support multiple dynamic symbolics
    assert len(dyn_symbolic) == 1, "Only support one dyanmic symbolics currently"

    ib = tvm.tir.ir_builder.create()
    syb = list(dyn_symbolic)[-1]
    last_range = 0
    for i, (_range, g_var) in enumerate(zip(func_range, global_symbols)):
        if i == 0:
            with ib.if_scope(syb <= _range):
                ib.emit(tvm.tir.Call(None, g_var, _invoke_params))
        else:
            with ib.if_scope(tvm.tir.all(syb > last_range, syb <= _range)):
                ib.emit(tvm.tir.Call(None, g_var, _invoke_params))
        last_range = _range
    with ib.if_scope(syb > last_range):
        ib.emit(tvm.tir.Call(None, g_var, _invoke_params))
    stmt = ib.get()
    dispatch_func = tvm.tir.PrimFunc(params, stmt, ret_type, buffer_map, attrs).with_attrs(
        {"tir.is_global_func": True, "global_symbol": global_symbol}
    )
    return dispatch_func


def create_dispatch_mod(
    g_var: str, original_func: tir.PrimFunc, specialized_funcs: List[tir.PrimFunc]
) -> IRModule:
    dispatch_mod: IRModule = tvm.IRModule()
    g_var_supply = GlobalVarSupply(dispatch_mod)
    refactored_funcs = []
    for func in specialized_funcs:
        params, buffers_to_declare = collect_buffers_to_declare(func)
        global_symbol, device_func = refactor_specialized_func(
            g_var, func, params, buffers_to_declare
        )
        global_symbol = g_var_supply.fresh_global(global_symbol, add_prefix=False)
        dispatch_mod[global_symbol] = device_func
        refactored_funcs.append((global_symbol, device_func))
    dispatch_func = create_dispatch_func(g_var, original_func, refactored_funcs=refactored_funcs)
    dispatch_mod.update(tvm.IRModule.from_expr(dispatch_func))
    return dispatch_mod


def fast_tune_with_dynamic_range(
    func: tir.PrimFunc,
    target: tvm.target.Target,
    topk: int = 10,
    parallel_build: bool = True,
    global_symbol: Optional[str] = None,
    dynamic_range: Dict[str, List[int]] = {},
) -> IRModule:
    if target.kind.name != "cuda":
        print("[FastDlight] Only support CUDA target")
        return None
    if not global_symbol:
        global_symbol = func.attrs["global_symbol"]

    # set opt_shapes for the primfunc with dynamc symbolic
    opt_shapes: Dict[str, List[int]] = {}
    for buffer in func.buffer_map.values():
        for axis in buffer.shape:
            if isinstance(axis, tvm.tir.Var):
                if axis.name in dynamic_range:
                    opt_shapes[axis.name] = dynamic_range[axis.name]
                else:
                    raise ValueError(f"[FastDlight] The axis {axis.name} is not in dynamic_range")
    func = func.with_attr("opt_shapes", opt_shapes)

    if func.attrs is not None and "opt_shapes" not in func.attrs:
        print("[FastDlight] The primfunc has no opt_shapes, please set opt_shapes for the primfunc")
        return None
    else:
        # should be list value
        if not all([isinstance(v, tvm.ir.Array) for v in func.attrs["opt_shapes"].values()]):
            print("[FastDlight] The opt_shapes should be list value")
            return None

    print("[FastDlight] Start fast tuning with dynamic range")
    opt_shapes = func.attrs["opt_shapes"]

    # Step 1.Calculate the Cartesian product using itertools.product
    product_list = list(itertools.product(*(opt_shapes[key] for key in opt_shapes)))

    # Convert the Cartesian product to a list of dictionaries
    specialize_items: List[Dict] = [dict(zip(opt_shapes.keys(), values)) for values in product_list]

    specilized_tuned_funcs: List[tir.PrimFunc] = []
    for item in specialize_items:
        func = func.with_attr("opt_shapes", item)
        _, best = fast_tune(func, target, topk, parallel_build)
        if best is None:
            return None
        specilized_tuned_funcs.append(best.sch.mod["main"])

    return create_dispatch_mod(global_symbol, func, specilized_tuned_funcs)
