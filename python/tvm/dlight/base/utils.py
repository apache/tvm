import tvm
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Tuple


class CompileResult:
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


def _apply_and_build(
    func,
    rule,
    config,
    arch,
):
    print("[FastDlight] Applying with config ", config)
    sch = rule.apply_config(func, config)
    if sch is None:
        return config, sch, None
    with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
        mod = tvm.build(sch.mod["main"], target=arch.target)
    return config, sch, mod


def apply_and_build_parallel(
    func,
    rule,
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
            [rule for _ in configs],
            configs,
            [arch for _ in configs],
        )

    for config, sch, mod in future_to_configs:
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
    rule,
    configs,
    arch,
    parallel_build=False,
) -> Tuple[List[CompileResult], CompileResult]:
    if parallel_build:
        return apply_and_build_parallel(func, rule, configs, arch)
    cpresults = []
    for config in configs:
        config, sch, mod = _apply_and_build(func, rule, config, arch)

        cpresult = CompileResult(config, sch, mod)
        args = func.buffer_map.values()
        profile_tensors = []
        for arg in args:
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
