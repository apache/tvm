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
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import List, Tuple, Optional
from tvm import tir
from tvm import dlight as dl
from .analysis import get_root_block, get_reduction_blocks


class CompileResult:
    """
    Class to store the result of compilation
    """

    def __init__(self, config, sch, mod):
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
            return dl.gpu.MatmulTensorization().apply_config(func, config)
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


def _apply_and_build(
    func,
    config,
    arch,
) -> Tuple[Optional[List[tir.Schedule]], Optional[tvm.IRModule], Optional[tvm.runtime.Module]]:
    sch = _apply_config(func, config)
    if sch is None:
        return config, sch, None
    
    # TODO(@lei): is tvm.build thread safe?
    try:
        with tvm.transform.PassContext(config={"tir.use_async_copy": True}):
            mod = tvm.build(sch.mod["main"], target=arch.target)
    except:
        mod = None
    return config, sch, mod


def apply_and_build_parallel(
    func,
    configs,
    arch,
    num_repeats=3,
) -> CompileResult:
    cpresults = []

    profile_tensors = []
    for param in func.params:
        arg = func.buffer_map[param]
        if arg.dtype == "int8":
            profile_tensors.append(
                tvm.nd.array(
                    np.random.randint(-127, 128, [int(i) for i in arg.shape]).astype(arg.dtype),
                    device=arch.device,
                )
            )
        else:
            profile_tensors.append(
                tvm.nd.array(
                    np.random.uniform(0, 1, [int(i) for i in arg.shape]).astype(arg.dtype),
                    device=arch.device,
                )
            )

    num_procs = min(len(configs), os.cpu_count(), 10)
    with ThreadPoolExecutor(max_workers=num_procs) as executor:
        all_tasks = []
        for config in configs:
            t = executor.submit(
                _apply_and_build,
                func,
                config,
                arch,
            )
            all_tasks.append(t)

    for future in as_completed(all_tasks):
        config, sch, mod = future.result()
        if sch is None:
            continue

        if mod is None:
            continue
        cpresult = CompileResult(config, sch, mod)

        timer_cuda_mod = mod.time_evaluator(mod.entry_name, arch.device, number=num_repeats)
        cpresult.profile_tensors = profile_tensors
        cpresult.time_evaluator = timer_cuda_mod
        cpresults.append(cpresult)

    best = None
    best_latency = 1e9
    for cpresult in cpresults:
        config = cpresult.config
        try:
            latency = cpresult.profile()
        except:
            print("[FastDlight] Evaluation with config failed: ", config)
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
    if parallel_build:
        return apply_and_build_parallel(func, configs, arch)
    cpresults = []
    for config in configs:
        config, sch, mod = _apply_and_build(func, config, arch)

        cpresult = CompileResult(config, sch, mod)
        profile_tensors = []
        for param in func.params:
            arg = func.buffer_map[param]
            if arg.dtype == "int8":
                profile_tensors.append(
                    tvm.nd.array(
                        np.random.randint(-127, 128, [int(i) for i in arg.shape]).astype(arg.dtype),
                        device=arch.device,
                    )
                )
            else:
                profile_tensors.append(
                    tvm.nd.array(
                        np.random.uniform(0, 1, [int(i) for i in arg.shape]).astype(arg.dtype),
                        device=arch.device,
                    )
                )
        if mod:
            timer_cuda_mod = mod.time_evaluator(mod.entry_name, arch.device, number=5)
            cpresult.profile_tensors = profile_tensors
            cpresult.time_evaluator = timer_cuda_mod
            cpresults.append(cpresult)

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
