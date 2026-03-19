# RK3588 NPU Hardware Block Guide

This is a compact reference for the RK3588 NPU block names, what they do,
and how our current TVM RKNPU backend maps compiler-visible ops onto them.

Two important scope notes:

- The top-level block names here are the TRM names: `PC`, `CNA`, `CORE`,
  `DPU`, `DPU_RDMA`, `PPU`, `PPU_RDMA`.
- Names like `CNA.matmul` or `DPU.relu` are project shorthand. They are useful
  for traces and pretty-printing, but they are not literal TRM object names.

Primary sources:

- `/home/user/code/00-rk3588-reverse-info/00-npu-chapter-overview-and-app-notes-take-2.md`
- `/home/user/code/00-rk3588-reverse-info/00-npu-chapter-register-descriptions.md`
- Current backend implementation under
  `python/tvm/relax/backend/contrib/rknpu/npu_core/`

## Top-Level Blocks

| Block | TRM role | Arithmetic or plumbing? | Current shorthand |
| --- | --- | --- | --- |
| `PC` | Program Counter. Fetches register lists for each task and starts a group of tasks. | Plumbing | `PC.task_group` |
| `CNA` | Convolution / Network Accelerator front-end. Feeds conv / matmul style work into the MAC pipeline. | Arithmetic/config | `CNA.conv2d`, `CNA.matmul` |
| `CORE` | Core-side output sizing / MAC-array-adjacent configuration. Needed for conv / matmul style tasks. | Mostly plumbing/config | `CORE` |
| `DPU` | Data Processing Unit. Post-processing and programmable per-element math. | Arithmetic | `DPU.BS`, `DPU.BN`, `DPU.EW`, `DPU.LUT` |
| `DPU_RDMA` | DPU-side DMA for bias / residual / EW operand reads and flying-mode inputs. | Plumbing/data movement | `DPU_RDMA.bias`, `DPU_RDMA.ew_operand` |
| `PPU` | Planar Processing Unit. Pooling and planar post-processing after DPU output. | Arithmetic | `PPU.pool` |
| `PPU_RDMA` | PPU-side DMA for PPU input reads. | Plumbing/data movement | `PPU_RDMA` |

TRM register-space mapping:

| Block | Base range |
| --- | --- |
| `PC` | `0x0000-0x0fff` |
| `CNA` | `0x1000-0x1fff` |
| `CORE` | `0x3000-0x3fff` |
| `DPU` | `0x4000-0x4fff` |
| `DPU_RDMA` | `0x5000-0x5fff` |
| `PPU` | `0x6000-0x6fff` |
| `PPU_RDMA` | `0x7000-0x7fff` |

## DPU Internal Sub-Blocks

The TRM does not expose these as separate top-level blocks, but the DPU
register file is clearly partitioned into several functional regions.

| DPU sub-block | TRM evidence | What it does | Useful shorthand |
| --- | --- | --- | --- |
| `BS` | `dpu_bs_*` registers | Bias/scale style post-processing. Has ALU, MUL, ReLU/ReluX controls. | `DPU.BS.bias`, `DPU.BS.relu` |
| `BN` | `dpu_bn_*` registers | Batch-norm-like / per-channel post-processing. Also has ALU, MUL, ReLU/ReluX controls. | `DPU.BN.*` |
| `EW` | `dpu_ew_*` registers | General elementwise pipeline. Supports add, min, max, minus, div, abs, neg, floor, ceil, plus ReLU/ReluX and converters. | `DPU.EW.add`, `DPU.EW.mul`, `DPU.EW.relu` |
| `LUT` | `dpu_lut_*` registers | LUT upload and LUT evaluation for activation-style approximations. | `DPU.LUT.exp`, `DPU.LUT.sigmoid_like`, `DPU.LUT.reciprocal` |
| output converter | `dpu_out_cvt_*` registers | Converts DPU output format / scale / shift on writeout. | `DPU.out_cvt` |

## What The TRM Says Each Area Is For

### PC

- PC mode fetches register config for each task.
- PC launches a group of tasks.
- `pc_task_con.task_number` sets the total task count.

This is orchestration only. It does not perform math.

### CNA

The overview chapter describes CNA as the main process unit for neural-network
arithmetic. It includes the convolution pre-process controller, internal
buffering, MAC array, and accumulator. In practice, this is the front-end for
conv / GEMM-like work.

Use this mental model:

- `CNA` handles conv / matmul style dataflow
- `CORE` supplies related output-shape / MAC pipeline configuration
- `DPU` does the post-processing that follows

### DPU

The overview chapter describes DPU as handling single-data calculations such as
`leaky_relu`, `relu`, `relux`, `sigmoid`, `tanh`, plus things like softmax,
transpose, and data-format conversion.

In register terms, the DPU is really a small programmable post-processing
pipeline with these main pieces:

- `BS`: bias/scale and related post-conv post-matmul work
- `BN`: another per-channel arithmetic stage
- `EW`: general elementwise arithmetic
- `LUT`: lookup-table evaluation
- output conversion

### DPU_RDMA

This is not a math block. It feeds extra operands into DPU-side arithmetic,
including:

- bias / scale operands
- EW operands
- flying-mode input streams

If a trace lights up `DPU_RDMA`, that usually means "this task needed an extra
operand stream" rather than "this task did a new semantic op."

### PPU

The overview chapter describes PPU as handling planar functions after DPU
output, especially:

- average pooling
- max pooling
- min pooling

The register descriptions also expose unpooling-related fields.

### PPU_RDMA

Again, plumbing rather than arithmetic. This is the DMA side of PPU input.

## Current Backend Mapping

This table is the most practical summary for day-to-day work. It describes what
our current TVM backend uses, not every possible hardware mode the TRM hints at.

| Compiler-visible op | Blocks used now | Notes |
| --- | --- | --- |
| `matmul` | `PC + CNA + CORE + DPU` | No extra operand DMA when unfused and bias-free. |
| `matmul + bias` | `PC + CNA + CORE + DPU + DPU_RDMA` | Bias is read through `DPU_RDMA`; post-op happens in DPU. |
| `matmul + bias + relu` | `PC + CNA + CORE + DPU + DPU_RDMA` | Same as above, with DPU-side ReLU enabled. |
| `conv2d` | `PC + CNA + CORE + DPU` | General mode-0 conv follows the same broad pipeline shape as matmul. |
| `conv2d + bias` | `PC + CNA + CORE + DPU + DPU_RDMA` | Bias path uses DPU-side operand DMA. |
| `conv2d + residual` | `PC + CNA + CORE + DPU + DPU_RDMA` | Residual/EW path also uses `DPU_RDMA`. |
| `conv2d + relu` | `PC + CNA + CORE + DPU` or `+ DPU_RDMA` | Depends on whether extra operands are fused. |
| `add` | `PC + DPU + DPU_RDMA` | Implemented through the EW pipeline. |
| `mul` | `PC + DPU + DPU_RDMA` | Implemented through the EW pipeline. |
| `relu` | `PC + DPU + DPU_RDMA` | Implemented via EW/BN-style post-op configuration, depending on task template. |
| `exp` | `PC + DPU + DPU_RDMA` | Implemented via combined LUT task. |
| `reciprocal` | `PC + DPU + DPU_RDMA` | Implemented via combined LUT task. |
| `gelu` | `PC + DPU + DPU_RDMA` | Current stage is LUT eval plus in-place EW multiply. |
| `avg/max/min pool` | `PC + PPU + PPU_RDMA` | PPU domain. Current pretty-print coverage for this is still basic. |

## DPU Sub-Block Mapping For Current Ops

This is the level we usually care about when reading traces.

| Shorthand | Likely meaning in current backend |
| --- | --- |
| `CNA.matmul` | Matmul / FC-style front-end setup in CNA |
| `CNA.conv2d` | Conv front-end setup in CNA |
| `DPU.BS.bias` | Bias / scale stage active |
| `DPU.BN.relu` | ReLU / ReluX style post-op in BN-style registers |
| `DPU.EW.add` | Elementwise add path |
| `DPU.EW.mul` | Elementwise multiply path |
| `DPU.EW.relu` | EW path with ReLU enabled |
| `DPU.LUT.exp` | LUT-configured exp approximation |
| `DPU.LUT.reciprocal` | LUT-configured reciprocal approximation |
| `DPU.LUT.gelu` | LUT-configured sigmoid-like GELU substep |

Important caveat:

- `DPU.<something>` in our pretty-printer is sometimes still a coarse label.
- The raw block membership is real.
- The exact sub-block label can still be improved by decoding more register
  fields directly, especially distinguishing `BS` vs `BN` vs `EW` vs `LUT`
  more precisely.

## Practical Reading Guide For Traces

When reading a task group:

- `PC` tells you there is a task chain.
- `CNA + CORE` tells you you are in conv/matmul territory.
- `DPU_RDMA` usually tells you there is an extra operand stream:
  bias, residual, or EW operand.
- `DPU` tells you some post-processing is happening.
- `PPU` tells you planar post-processing like pooling is happening.

Examples:

- `uses=[CNA.matmul, CORE, DPU_RDMA, DPU.BS.bias, DPU.BN.relu]`
  means: matmul-style compute plus bias plus ReLU in one task family.

- `uses=[DPU_RDMA, DPU.EW.add]`
  means: pure elementwise add, no CNA/CORE math block active.

- `uses=[DPU_RDMA, DPU.LUT.exp]`
  means: LUT-only activation-style task.

- `uses=[PPU, PPU_RDMA]`
  means: planar post-processing, typically pooling.

## Recommended Terminology Going Forward

For docs and traces, use:

- `submit` for the host/runtime submit
- `pc_task_group` for the hardware task group launched by PC
- `pc_task` for one hardware task
- `uses=[...]` with TRM block names and project shorthand

Good:

- `uses=[CNA.matmul, CORE, DPU_RDMA, DPU.EW.add]`

Avoid:

- pretending `CNA.matmul` is an official TRM object name
- treating `CORE` or `*_RDMA` as semantic ops by themselves

## What Is Still Missing

This guide is deliberately practical, not exhaustive.

It does not yet try to be a complete decode of:

- every DPU sub-block mode bit
- every conv mode
- every flying-mode path
- every PPU operating mode

If we need that later, the next step is a separate "register-field to semantic
sub-block" guide rather than bloating this file.
