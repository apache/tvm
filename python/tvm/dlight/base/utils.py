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
from concurrent.futures import ThreadPoolExecutor
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
    # try:
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
            sch = rule.apply_config(func, config)
            if sch is not None:
                return sch
    # except Exception as e:
    #     print("[FastDlight] Apply config failed: ", e)
    #     return None
    return None


def _apply_and_build(
    func,
    config,
    arch,
) -> Tuple[Optional[List[tir.Schedule]], Optional[tvm.IRModule], Optional[tvm.runtime.Module]]:
    sch = _apply_config(func, config)
    if sch is None:
        return config, sch, None
    # todo(lei): should add exception handling
    with tvm.transform.PassContext(config={"tir.use_async_copy": True}):
        mod = tvm.build(sch.mod["main"], target=arch.target)
    return config, sch, mod


def apply_and_build_parallel(
    func,
    configs,
    arch,
    num_repeats=3,
) -> CompileResult:
    cpresults = []

    args = func.buffer_map.values()
    profile_tensors = []
    for arg in args:
        profile_tensors.append(
            tvm.nd.array(
                np.random.uniform(0, 1, [int(i) for i in arg.shape]).astype(arg.dtype),
                device=arch.device,
            )
        )

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_configs = executor.map(
            _apply_and_build,
            [func for _ in configs],
            configs,
            [arch for _ in configs],
        )

    for config, sch, mod in future_to_configs:
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

        latency = cpresult.profile()
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

        if mod is None:
            continue

        cpresult = CompileResult(config, sch, mod)
        args = func.buffer_map.values()
        profile_tensors = []
        for arg in args:
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
        latency = cpresult.profile()
        print("[FastDlight] Evaluation with config ", config)
        print("[FastDlight] Time cost of this config: {:.3f} ms".format(latency * 1e3))
        cpresult.latency = latency
        if latency < best_latency:
            best_latency = latency
            best = cpresult

    return cpresults, best
