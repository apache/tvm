# RK3588 NPU Performance Reference

This file is generated from `tools/rknpu_performance_reference.py`.

For the current architectural fusion comparison reference, see
[rknpu_fusion_matrix_reference.md](/home/user/code/tvm/docs/arch/rknpu_fusion_matrix_reference.md).
For the current submit/task overhead reference, see
[rknpu_overhead_reference.md](/home/user/code/tvm/docs/arch/rknpu_overhead_reference.md).

## Scope

- Measured on the current 2-D clean subset with `M in {1, 1500}`, `D=64`, `H=256`.
- Primary latency is `runtime_total_tail` after warmup mode `fixed` (minimum warmup `16` iteration(s)).
- Runtime DMA cache mode: `on`.
- Persistent DMA cache can materially change split-submit versus same-submit comparisons; use `--bridge-cache-dma off` for raw schedule studies and compare both modes when in doubt.
- Theoretical per-block TOPS are intentionally not claimed here unless they come directly from the TRM.

## Task Templates

| Name | Role | RegCmds | Data RegCfg | Enable Mask |
| --- | --- | ---: | ---: | --- |
| `cna_dpu` | matmul / conv without extra operand DMA | 112 | 108 | `0x000d` |
| `cna_dpu_dpu_rdma` | matmul / conv with bias or other extra operand stream | 130 | 126 | `0x001d` |
| `ew` | elementwise add / mul / relu style task | 74 | 70 | `0x0018` |
| `ppu` | pooling / planar post-processing task | 30 | 26 | `0x0060` |
| `lut_combined` | combined LUT upload + eval task | 1102 | 1098 | `0x0018` |

## Latency Atlas

| Family | Size | Shape | Cold Runtime (ns) | Tail Runtime (ns) | Tail HW (ns) | Tail Wall (ns) | Submits | Tasks | Stage IDs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `matmul` | `small` | `[1x64] x [64x256] -> [1x256]` | 132711 | 195129 | 74377 | 396092 | 1 | 1 | `[[1]]` |
| `matmul` | `large` | `[1500x64] x [64x256] -> [1500x256]` | 1335570 | 1349863 | 604055 | 1892374 | 1 | 2 | `[[1]]` |
| `matmul_bias` | `small` | `[1x64] x [64x256] + [256] -> [1x256]` | 200379 | 199504 | 74668 | 440135 | 1 | 1 | `[[11]]` |
| `matmul_bias` | `large` | `[1500x64] x [64x256] + [256] -> [1500x256]` | 1407031 | 1466532 | 612805 | 2064753 | 1 | 2 | `[[11]]` |
| `matmul_bias_relu` | `small` | `relu([1x64] x [64x256] + [256]) -> [1x256]` | 196004 | 193962 | 70876 | 431384 | 1 | 1 | `[[6]]` |
| `matmul_bias_relu` | `large` | `relu([1500x64] x [64x256] + [256]) -> [1500x256]` | 1416363 | 1445531 | 617763 | 2032085 | 1 | 2 | `[[6]]` |
| `add` | `small` | `[1x64] + [1x64] -> [1x64]` | 103544 | 108211 | 70293 | 326382 | 1 | 1 | `[[2]]` |
| `add` | `large` | `[1500x64] + [1500x64] -> [1500x64]` | 1233193 | 1246610 | 484469 | 1560742 | 1 | 1 | `[[2]]` |
| `mul` | `small` | `[1x64] * [1x64] -> [1x64]` | 96252 | 94793 | 67376 | 284964 | 1 | 1 | `[[9]]` |
| `mul` | `large` | `[1500x64] * [1500x64] -> [1500x64]` | 1209860 | 1236693 | 473385 | 1551408 | 1 | 1 | `[[9]]` |
| `residual_mlp_relu` | `small` | `relu([1x64] x [64x256] + [256]); [1x256] x [256x64] + [64] + residual` | 310923 | 311798 | 81669 | 668806 | 1 | 3 | `[[6, 11, 2]]` |
| `residual_mlp_relu` | `large` | `relu([1500x64] x [64x256] + [256]); [1500x256] x [256x64] + [64] + residual` | 3309613 | 3257987 | 1346946 | 3883042 | 1 | 6 | `[[6, 11, 2]]` |

## Fusion Penalty

| Size | Mode | Cold Runtime (ns) | Tail Runtime (ns) | Tail HW (ns) | Tail Wall (ns) | Ratio vs fused | Submits | Tasks | Stage IDs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `small` | `fused task` | 196004 | 193962 | 70876 | 431384 | 1.000 | 1 | 1 | `[[6]]` |
| `small` | `split tasks` | 210004 | 211171 | 79335 | 455301 | 1.089 | 1 | 3 | `[[1, 2, 3]]` |
| `small` | `split submits` | 182879 | 366049 | 207379 | 608138 | 1.887 | 3 | 3 | `[[1], [2], [3]]` |
| `large` | `fused task` | 1416363 | 1445531 | 617763 | 2032085 | 1.000 | 1 | 2 | `[[6]]` |
| `large` | `split tasks` | 6426722 | 6681060 | 3897333 | 7986879 | 4.622 | 1 | 4 | `[[1, 2, 3]]` |
| `large` | `split submits` | 3917166 | 3806915 | 1850081 | 4373344 | 2.634 | 3 | 4 | `[[1], [2], [3]]` |

## Effective Throughput

- Best measured effective TFLOP/s: `0.036413` from `matmul_large`.
- Best measured effective TMAC/s: `0.018206` from `matmul_large`.
- Best measured effective Gelem/s: `0.077626` from `mul_large`.

## Useful Additional Fields To Catalog

- actual NPU core clock during the run
- 1-core vs 3-core scaling for the same task template
- LUT and PPU throughput once those paths are fully signed off
- materialized-temp bytes and DRAM traffic proxies
- thermal state and clock governor settings
- shape cliffs where tiling or layout changes cause step-function latency jumps

## Unknowns

- The public TRM does not give precise per-block TOPS numbers for `DPU.BS`, `DPU.BN`, `DPU.EW`, `DPU.LUT`, or `PPU`.
- `MIPS` is not a useful metric for this accelerator; task latency and effective throughput are more actionable.
