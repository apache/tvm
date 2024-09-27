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
"""The auto-tune module for tl programs."""

import tvm
from tvm import tl
import inspect
from functools import wraps
from typing import Any, Callable, List, Literal
from tqdm import tqdm

class Autotuner:
    def __init__(
        self,
        fn: Callable,
        configs: Any,
        keys: List[str],
        warmup: int = 25,
        rep: int = 100,
    ):
        self.fn = fn
        self.configs = configs
        self.keys = keys
        self.warmup = warmup
        self.rep = rep
    
    def run(self, *args: Any, **kwds: Any) -> Any:
        sig = inspect.signature(self.fn)
        bound_args = sig.bind(*args, **kwds)
        bound_args.apply_defaults()
        # print("auto-tunner bound_args:")
        # for name, value in bound_args.arguments.items():
        #     print(f"{name} = {value}")
        best_latency = 1e8
        best_config = None
        for config in tqdm(self.configs, desc="Auto-tuning progress"):
            tqdm.write(f"Current config: {config}")
            new_args = []
            for name, value in bound_args.arguments.items():
                if name not in self.keys:
                    new_args.append(value)
                else:
                    new_args.append(config[name])
            new_args = tuple(new_args)
            # print("auto-tunner new_args:", new_args)
            try:
                latency, ref_latency = self.fn(*new_args, **kwds)
            except Exception as e:
                print("Fail on config ", config, " with error: ", e)
                latency = 1e8
            if latency < best_latency:
                best_latency = latency
                best_config = config
            tqdm.write(f"Latency: {latency}")
        return best_latency, best_config, ref_latency

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)

def autotune(configs: Any, keys: List[str], warmup: int = 25, rep: int = 100) -> Callable:
    """
    Decorator for tl program
    """
    def decorator(fn: Callable) -> Autotuner:
        return Autotuner(fn, configs=configs, keys=keys, warmup=warmup, rep=rep)
    return decorator

def jit(
    out_idx: List[int], 
    supply_type: tl.TensorSupplyType = tl.TensorSupplyType.Normal, 
    ref_prog: Callable = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    skip_check: bool = False, 
    profiler: Literal['torch', 'tvm']='torch'
    ) -> Callable:
    
    def wrapper(fn: Callable):
        ref_latency_cache = None
        @wraps(fn)
        def decorator(*args, **kwargs) -> float:
            nonlocal ref_latency_cache
            # Enabling Efficient Fusion
            with tvm.transform.PassContext(config={
                "tir.merge_static_smem": True
            }):
                mod, params = tl.lower(fn(*args, **kwargs))
            
            mod = tl.Profiler(mod, params, out_idx, supply_type)
            if (not skip_check) and (ref_prog is not None):
                mod.assert_allclose(ref_prog, rtol=rtol, atol=atol)
            
            latency = mod.do_bench(mod.func, warmup = 25, profiler = profiler)
            if ref_latency_cache is None and ref_prog is not None:
                ref_latency_cache = mod.do_bench(ref_prog, warmup = 25)
            return latency, ref_latency_cache
        return decorator
    return wrapper