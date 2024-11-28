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
from typing import Any, Callable, List, Any, Literal
import inspect
from tqdm import tqdm
import logging
from dataclasses import dataclass
import concurrent.futures

logging.basicConfig(filename='out.log', filemode='w', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')



@dataclass(frozen=True)
class JITContext:
    mod: tl.Profiler
    out_idx: List[int]
    supply_type: tl.TensorSupplyType
    ref_prog: Callable
    rtol: float
    atol: float
    skip_check: bool
    profiler: Literal['torch', 'tvm']
    target: Literal['cuda', 'hip']

class Autotuner:
    def __init__(
        self,
        fn: Callable,
        configs: Any,
        keys: List[str],
        warmup: int = 25,
        rep: int = 100,
        timeout: int = 30,
    ):
        self.fn = fn
        self.configs = configs
        self.keys = keys
        self.warmup = warmup
        self.rep = rep
        self.timeout = timeout
        
        # Precompute cached variables
        self.ref_latency_cache = None
        self.jit_input_tensors = None
        self.ref_input_tensors = None
    
    def run(self, *args: Any, **kwds: Any) -> Any:
        sig = inspect.signature(self.fn)
        bound_args = sig.bind(*args, **kwds)
        bound_args.apply_defaults()
        # print("auto-tunner bound_args:")
        # for name, value in bound_args.arguments.items():
        #     print(f"{name} = {value}")
        best_latency = 1e8
        best_config = None

        def target_fn(*new_args, **kwds):
            jit_context = self.fn(*new_args, **kwds)

            # Unpack the context
            mod = jit_context.mod
            profiler = jit_context.profiler
            skip_check = jit_context.skip_check
            ref_prog = jit_context.ref_prog
            rtol = jit_context.rtol
            atol = jit_context.atol

            self.jit_input_tensors = mod._get_inputs(
                with_output = profiler == "tvm"
            ) if self.jit_input_tensors is None else self.jit_input_tensors

            if (not skip_check) and (ref_prog is not None):
                mod.assert_allclose(ref_prog, rtol=rtol, atol=atol)

            latency = mod.do_bench(mod.func, n_warmup=self.warmup, n_repeat=self.rep, profiler=profiler, input_tensors=self.jit_input_tensors)
            if self.ref_latency_cache is None and ref_prog is not None:
                self.ref_input_tensors = mod._get_inputs(with_output=False) if self.ref_input_tensors is None else self.ref_input_tensors
                self.ref_latency_cache = mod.do_bench(ref_prog, n_warmup=self.warmup, n_repeat=self.rep, profiler="torch", input_tensors=self.ref_input_tensors)

            return latency, self.ref_latency_cache

        progress_bar = tqdm(self.configs, desc="Running configurations")
        for config in progress_bar:
            new_args = []
            for name, value in bound_args.arguments.items():
                if name not in self.keys:
                    new_args.append(value)
                else:
                    new_args.append(config[name])
            new_args = tuple(new_args)
            ref_latency = None
            try:
                # Use ThreadPoolExecutor to enforce timeout on target_fn execution
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(target_fn, *new_args, **kwds)
                    latency, ref_latency = future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                logging.error(f"Timeout exceeded for config {config}. Skipping this configuration.")
                continue
            except Exception as e:
                logging.error(f"An error occurred while testing config {config}: {e}")
                continue


            logging.info(f"Config {config} latency: {latency}")

            progress_bar.set_postfix({"best_latency": best_latency})

            if latency < best_latency:
                best_latency = latency
                best_config = config
            tqdm.write(f"Tuned Latency {latency} with config {config}")
        return best_latency, best_config, ref_latency

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)

def autotune(configs: Any, keys: List[str], warmup: int = 25, rep: int = 100, timeout: int = 100) -> Callable:
    """
    Decorator for tl program
    """
    def decorator(fn: Callable) -> Autotuner:
        return Autotuner(fn, configs=configs, keys=keys, warmup=warmup, rep=rep, timeout=timeout)
    return decorator

def jit(
    out_idx: List[int], 
    supply_type: tl.TensorSupplyType = tl.TensorSupplyType.Normal, 
    ref_prog: Callable = None,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    skip_check: bool = False, 
    profiler: Literal['torch', 'tvm']='torch',
    target: Literal['cuda', 'hip']='cuda'
) -> Callable:

    def wrapper(fn: Callable):

        @wraps(fn)
        def decorator(*args, **kwargs) -> float:
            # Enabling Efficient Fusion
            with tvm.transform.PassContext(config={
                "tir.merge_static_smem": True
            }):
                mod, params = tl.lower(fn(*args, **kwargs), target=target)

            mod = tl.Profiler(mod, params, out_idx, supply_type)
            
            return JITContext(
                mod=mod,
                out_idx=out_idx,
                supply_type=supply_type,
                ref_prog=ref_prog,
                rtol=rtol,
                atol=atol,
                skip_check=skip_check,
                profiler=profiler,
                target=target
            )

        return decorator
    return wrapper