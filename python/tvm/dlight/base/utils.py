import tvm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

class CompileResult:
    def __init__(self, config, sch, mod):
        self.config = config
        self.sch = sch
        self.mod = mod
        self.latency = None
        self.profile_tensors = []
        self.time_evaluator = None

    def profile(self):
        return self.time_evaluator(*self.profile_tensors).mean

def apply_and_build(
    func,
    rule,
    config,
    arch,
):
    sch = rule.apply_config(func, config)
    mod = tvm.build(sch.mod["main"], target=arch.target)
    return config, sch, mod

def apply_and_build_parallel(
    func,
    rule,
    configs,
    arch,
):
    cpresults = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_config = {executor.submit(apply_and_build, func, rule, config, arch): config for config in configs}
        
        for future in as_completed(future_to_config):
            config, sch, mod = future.result()
            cpresult = CompileResult(config, sch, mod)
            args = func.buffer_map.values()
        
            profile_tensors = []
            for arg in args:
                profile_tensors.append(tvm.nd.array(
                    np.random.uniform(0, 1, [int(i) for i in arg.shape]).astype(arg.dtype), device=arch.device)
                )
            timer_cuda_mod = mod.time_evaluator(
            mod.entry_name, arch.device, number=5)
            cpresult.profile_tensors = profile_tensors
            cpresult.time_evaluator = timer_cuda_mod
            cpresults.append(cpresult)

    best = None
    best_latency = 1e9
    for cpresult in cpresults:
        config = cpresult.config        
        
        latency = cpresult.profile()
        print("[FastDlight] Applying with config ", config)
        print("[FastDlight] Time cost of this config: {:.3f} ms".format(latency * 1e3))

        cpresult.latency = latency
        if latency < best_latency:
            best_latency = latency
            best = cpresult
            
    return best