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
from typing import List, Tuple, Optional
from tvm import tir, IRModule
from tvm.runtime import Module
from tvm.tir import Schedule
from tvm import dlight as dl
from .analysis import get_root_block, get_reduction_blocks
from .roller.arch import Arch
from ..base.roller.rasterization import NoRasterization
import tempfile
import re

def match_global_kernel(source: str) -> int:
    pattern = r"__global__\s+void\s+[__launch_bounds__\(\d+\)\s+]\w+"
    matched = re.findall(pattern, source)
    assert len(matched) > 1 # may have statement before kernel
    return source.index(matched[0])

def get_rasterization_code(pannel_width:int = 8) -> str:
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
    
    ...
    
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
            assert v.name in config.opt_shapes
            return config.opt_shapes[v.name]
        elif isinstance(v, tvm.tir.IntImm):
            return v.value
        else:
            raise RuntimeError("Not supported type: ", type(v))

    profile_tensors = []
    for param in func.params:
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
            _sched.append(future.result())

    builder = PopenPoolExecutor(max_workers=max_workers)

    # build in process parallel
    def _build(context) -> str:
        idx, mod, arch = context
        config = configs[idx]
        # TODO(lei):
        # this is a trick to implement rasteration, will be removed in the future
        @tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
        def tvm_callback_cuda_postproc(code, _):
            index = code.index("{", match_global_kernel(code))
            if not isinstance(config.rasterization_plan, NoRasterization):
                factor = config.rasterization_plan.panel_width_
                rasterization_code = get_rasterization_code(factor)
                code = code[: index + 2] + rasterization_code + code[index + 2 :]
            return code

        with tvm.transform.PassContext(
            config={"tir.use_async_copy": True, "tir.merge_static_smem": False}
        ):
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
            print("[FastDlight] LocalBuilder: An exception occurred ", map_result.value)
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
