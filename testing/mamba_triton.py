import torch
import triton
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.selective_state_update import selective_state_update

configs = [triton.testing.Benchmark(
            x_names=["batch", "seq_len", "nheads", "headdim", "ngroups", "dstate", "chunk_size"],
            x_vals=[
                (64, 4096, 64, 64, 8, 64, 256),
                ],
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=[""],
            line_names=[""],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="mamba2-performance-fp16",
            args={},
        )]

# @triton.testing.perf_report(configs)
# def benchmark(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider):
#     warmup = 25
#     rep = 100
#     x = torch.empty((batch, seq_len, nheads, headdim), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
#     dt = torch.empty((batch, seq_len, nheads), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
#     A = torch.empty((nheads), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
#     B = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
#     C = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
   
#     quantiles = [0.5, 0.2, 0.8]
#     ms, min_ms, max_ms = triton.testing.do_bench(lambda: mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size), warmup=warmup, rep=rep, quantiles=quantiles)
#     flops_chunk_cumsum_fwd = 0
#     flops_chunk_state_fwd =  2.0 * batch * seq_len * nheads * headdim * dstate
#     flops_state_passing_fwd = 0
#     flops_bmm_chunk_fwd = 2.0 * batch * ngroups * dstate * seq_len * chunk_size
#     flops_chunk_scan_fwd = 2.0 * batch * seq_len * chunk_size * nheads * headdim + 2.0 * batch * seq_len * nheads * headdim * dstate
#     total_flops = flops_chunk_cumsum_fwd + flops_chunk_state_fwd + flops_state_passing_fwd + flops_bmm_chunk_fwd + flops_chunk_scan_fwd
#     perf = lambda ms: total_flops * 1e-12 / (ms * 1e-3)
#     return perf(ms), perf(max_ms), perf(min_ms)


# benchmark.run(show_plots=True, print_data=True)


@triton.testing.perf_report(configs)
def benchmark(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider):
    warmup = 25
    rep = 100
    x = torch.empty((batch, seq_len, nheads, headdim), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    dt = torch.empty((batch, seq_len, nheads), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    A = torch.empty((nheads), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    B = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    C = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
   
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size), warmup=warmup, rep=rep, quantiles=quantiles)
    flops_chunk_cumsum_fwd = 0
    flops_chunk_state_fwd =  2.0 * batch * seq_len * nheads * headdim * dstate
    flops_state_passing_fwd = 0
    flops_bmm_chunk_fwd = 2.0 * batch * ngroups * dstate * seq_len * chunk_size
    flops_chunk_scan_fwd = 2.0 * batch * seq_len * chunk_size * nheads * headdim + 2.0 * batch * seq_len * nheads * headdim * dstate
    total_flops = flops_chunk_cumsum_fwd + flops_chunk_state_fwd + flops_state_passing_fwd + flops_bmm_chunk_fwd + flops_chunk_scan_fwd
    perf = lambda ms: total_flops * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


# benchmark.run(show_plots=True, print_data=True)
# benchmark(64, 4096, 64, 64, 8, 64, 256, "")

from transformers import AutoTokenizer, Mamba2Model
import torch

tokenizer = AutoTokenizer.from_pretrained("mistralai/mamba-codestral-7B-v0.1")
model = Mamba2Model.from_pretrained("mistralai/mamba-codestral-7B-v0.1")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state