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
"""The profiler and convert to torch utils"""

from typing import Any, List, Literal
from enum import Enum
from functools import partial
import torch

import tvm
from torch.utils.dlpack import to_dlpack
from tvm.runtime import ndarray
from tvm.relay import TensorType
from tvm.contrib.dlpack import to_pytorch_func
from torch.utils.dlpack import to_dlpack
from tvm.runtime import ndarray

from .engine import lower


class TensorSupplyType(Enum):
    Integer = 1
    Uniform = 2
    Normal = 3
    Randn = 4
    Zero = 5
    One = 6


def get_tensor_supply(supply_type: TensorSupplyType):
    def get_tensor(tensor: TensorType) -> torch.Tensor:
        dtype = torch.__getattribute__(str(tensor.dtype))
        device = torch.cuda.current_device()
        # torch.manual_seed(0)
        # torch.cuda.manual_seed(0)
        shape = list(map(int, tensor.shape))
        if dtype == torch.int8 and supply_type in [
            TensorSupplyType.Uniform,
            TensorSupplyType.Normal,
        ]:
            return torch.ones(*shape, device=device, dtype=dtype)

        if supply_type == TensorSupplyType.Integer:
            return torch.randint(low=-2, high=3, size=shape, device=device, dtype=dtype)
        elif supply_type == TensorSupplyType.Uniform:
            return torch.empty(*shape, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        elif supply_type == TensorSupplyType.Normal:
            return torch.empty(*shape, device=device, dtype=dtype).normal_(-1.0, 1.0)
        elif supply_type == TensorSupplyType.Randn:
            return torch.randn(*shape, device=device).to(dtype)
        elif supply_type == TensorSupplyType.Zero:
            return torch.zeros(*shape, device=device, dtype=dtype)
        elif supply_type == TensorSupplyType.One:
            return torch.ones(*shape, device=device, dtype=dtype)
        else:
            raise NotImplementedError(supply_type)

    return get_tensor


def torch_assert_close(tensor_a,
                       tensor_b,
                       rtol=1e-2,
                       atol=1e-3,
                       max_mismatched_ratio=0.001,
                       verbose=False):
    """
    Custom function to assert that two tensors are "close enough," allowing a specified 
    percentage of mismatched elements.

    Parameters:
    ----------
    tensor_a : torch.Tensor
        The first tensor to compare.
    tensor_b : torch.Tensor
        The second tensor to compare.
    rtol : float, optional
        Relative tolerance for comparison. Default is 1e-2.
    atol : float, optional
        Absolute tolerance for comparison. Default is 1e-3.
    max_mismatched_ratio : float, optional
        Maximum ratio of mismatched elements allowed (relative to the total number of elements). 
        Default is 0.001 (0.1% of total elements).

    Raises:
    -------
    AssertionError:
        If the ratio of mismatched elements exceeds `max_mismatched_ratio`.
    """
    import torch

    # Compute the absolute difference between the two tensors
    diff = torch.abs(tensor_a - tensor_b)

    # Compute the maximum allowable difference for each element
    max_diff = atol + rtol * torch.abs(tensor_b)

    # Identify elements where the difference exceeds the maximum allowable difference
    mismatched = diff > max_diff

    # Count the number of mismatched elements
    num_mismatched = mismatched.sum().item()

    # Calculate the total number of elements in the tensor
    total_elements = tensor_a.numel()

    # Compute the allowed mismatched elements based on the ratio
    max_allowed_mismatched = int(total_elements * max_mismatched_ratio)

    # Print debug information about the mismatch
    if verbose:
        print(f"Number of mismatched elements: {num_mismatched} / {total_elements} "
              f"(allowed: {max_allowed_mismatched})")

    # Check if the number of mismatched elements exceeds the allowed threshold
    if num_mismatched > max_allowed_mismatched:
        raise AssertionError(
            f"Too many mismatched elements: {num_mismatched} > {max_allowed_mismatched} "
            f"({max_mismatched_ratio * 100:.2f}% allowed). "
            f"Greatest absolute difference: {diff.max().item()}, "
            f"Greatest relative difference: {(diff / (torch.abs(tensor_b) + 1e-12)).max().item()}.")
    else:
        return True

class ConvertTorch:
    def __init__(self, mod, params: List[TensorType], result_idx: List[int]) -> None:
        self.mod = mod
        self.params = params
        self.result_idx = result_idx
        self.func = self._convert_torch_func()

    def _convert_torch_func(self) -> callable:
        torch_func = to_pytorch_func(self.mod)

        def func(*ins: List[torch.Tensor]):
            if len(ins) + len(self.result_idx) != len(self.params):
                raise ValueError(
                    f"Expected {len(self.params)} inputs, got {len(ins) + len(self.result_idx)} with {len(ins)} inputs and {len(self.result_idx)} outputs"
                )
            ins_idx = 0
            args = []
            device = torch.cuda.current_device()
            for i in range(len(self.params)):
                if i in self.result_idx:
                    dtype = torch.__getattribute__(str(self.params[i].dtype))
                    shape = list(map(int, self.params[i].shape))
                    tensor = torch.empty(*shape, dtype=dtype, device=device)
                else:
                    tensor = ins[ins_idx]
                    ins_idx += 1
                args.append(tensor)
            torch_func(*args)
            if len(self.result_idx) == 1:
                return args[self.result_idx[0]]
            else:
                return [args[i] for i in self.result_idx]

        return func

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)

    def get_kernel_source(self) -> str:
        return self.mod.imported_modules[0].get_source()


class Profiler(ConvertTorch):
    def __init__(
        self,
        mod,
        params: List[TensorType],
        result_idx: List[int],
        supply_type: TensorSupplyType = TensorSupplyType.Normal,
    ):
        super().__init__(mod, params, result_idx)
        self.supply = get_tensor_supply(supply_type)

    def _get_inputs(self, with_output=False):
        ins = []
        for i in range(len(self.params)):
            if with_output or i not in self.result_idx:
                ins.append(self.supply(self.params[i]))
        return ins

    def assert_allclose(self, reference_program: callable, atol: float = 1e-2, rtol: float = 1e-2, max_mismatched_ratio=0.01):
        ins = self._get_inputs()
        ref_outs = reference_program(*ins)
        torch.cuda.synchronize()
        lib_outs = self.func(*ins)
        torch.cuda.synchronize()

        if isinstance(lib_outs, torch.Tensor):
            lib_outs = [lib_outs]
        if isinstance(ref_outs, torch.Tensor):
            ref_outs = [ref_outs]
        assert len(lib_outs) == len(ref_outs)
        # torch.set_printoptions(edgeitems=torch.inf)
        for lhs, rhs in zip(lib_outs, ref_outs):
            # close_mask = torch.isclose(lhs, rhs, rtol=rtol, atol=atol)
            # total_elements = lhs.numel()
            # num_not_close = (~close_mask).sum().item()
            # percentage_not_close = (num_not_close / total_elements) * 100
            # print(f"{percentage_not_close:.2f}% of the elements are not close.")
            # print(f"Total elements: {total_elements}, Not close elements: {num_not_close}")
            torch_assert_close(lhs, rhs, rtol=rtol, atol=atol, max_mismatched_ratio=max_mismatched_ratio)

    def assert_consistent(self, repeat=10):
        # Used to check no race condition inside the kernel
        ins = self._get_inputs()
        ref_outs = self.func(*ins)

        for _ in range(repeat):
            lib_outs = self.func(*ins)
            for lhs, rhs in zip(lib_outs, ref_outs):
                assert torch.allclose(lhs, rhs), ["result is not consistent", lhs, rhs]

    def run_once(self, func=None):
        import ctypes
        libcuda = ctypes.CDLL("libcuda.so")
        
        ins = self._get_inputs()
        if not func:
            func = self.__call__
        return func(*ins)

    def do_bench(
        self,
        func: callable,
        warmup=25,
        rep=100,
        n_warmup=1,
        n_repeat=1,
        profiler: Literal["torch", "tvm", "auto"] = "auto",
        input_tensors: List[torch.Tensor] = None,
    ):
        if profiler == "torch":
            ins = self._get_inputs() if input_tensors is None else input_tensors
            bench_func = partial(func, *ins)
            return do_bench(
                bench_func, warmup=warmup, rep=rep, _n_warmup=n_warmup, _n_repeat=n_repeat
            )
        elif profiler == "tvm":
            ins = self._get_inputs(with_output=True) if input_tensors is None else input_tensors
            target = "cuda"
            try:
                target = self.mod.imported_modules[0].type_key
            except:
                pass
            
            assert target in ["cuda", "hip"], f"Unknown target: {target}"

            device = tvm.cuda(0) if target == "cuda" else tvm.rocm(0)
            time_evaluator = self.mod.time_evaluator(
                self.mod.entry_name, device, number=rep, repeat=n_repeat
            )
            tvm_inputs = [ndarray.from_dlpack(to_dlpack(inp)) for inp in ins]
            # Transform Latency to ms
            return time_evaluator(*tvm_inputs).mean * 1e3
        elif profiler == "auto":
            ins = self._get_inputs()
            bench_func = partial(func, *ins)
            torch_res = do_bench(
                bench_func, warmup=warmup, rep=rep, _n_warmup=n_warmup, _n_repeat=n_repeat
            )

            ins = self._get_inputs(with_output=True)
            time_evaluator = self.mod.time_evaluator(
                self.mod.entry_name, tvm.cuda(0), number=rep, repeat=n_repeat
            )
            tvm_inputs = [ndarray.from_dlpack(to_dlpack(inp)) for inp in ins]
            tvm_res = time_evaluator(*tvm_inputs).mean * 1e3
            return min(torch_res, tvm_res)
        else:
            raise ValueError(f"Unknown profiler: {profiler}")


def do_bench(
    fn,
    warmup=25,
    rep=100,
    _n_warmup=0,
    _n_repeat=0,
    grad_to_none=None,
    quantiles=None,
    fast_flush=True,
    return_mode="mean",
):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """
    assert return_mode in ["min", "max", "mean", "median"]
    fn()
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    if _n_warmup > 0:
        n_warmup = _n_warmup
    if _n_repeat > 0:
        n_repeat = _n_repeat
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float
    )
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()


_cached = {}


def cached(func, result_idx: List[int], *args):
    global _cached
    key = (func, tuple(result_idx), *args)
    if key not in _cached:
        program = func(*args)
        mod, params = lower(program)
        mod = ConvertTorch(mod, params, result_idx)
        _cached[key] = mod
    return _cached[key]
