# RKNPU MVP Hero Demo

This file is generated from `tools/rknpu_tir_mlp_mvp_demo.py`.

## Workload

- block: residual 2-D MLP with ReLU
- shape: `M=1500, D=64, H=256`
- cache mode: `off`
- warmup: `12`
- iters: `6`

Computation:

- `ff1 = matmul(x, w1) + b1`
- `act = relu(ff1)`
- `ff2 = matmul(act, w2) + b2`
- `out = ff2 + x`

The compare run holds the math fixed and lowers it three ways:

- `split_submits`: multi-submit non-fused
- `split_tasks`: one-submit non-fused
- `fused`: one-submit fused

## Reproduce

```bash
PYTHONPATH=python:. TVM_LIBRARY_PATH=build TVM_FFI_DISABLE_TORCH_C_DLPACK=1 python3 tools/rknpu_tir_mlp_mvp_demo.py --m 1500 --d-model 64 --hidden 256 --warmup 12 --iters 6 --real-submit --compare-variants --pretty --bridge-cache-dma off --bridge-debug-checks
```

## Strict Invariants

Every variant run in this artifact enforces:

- exact expected `submit_stage_ids`
- exact expected `num_submits`
- `blocked_boundary_count == 0`
- zero fallback and reloc mismatch counters
- embedded chain blob path
- `max_err <= 0.001` and zero non-finite outputs

## Summary

| Variant | Meaning | Stage IDs | Submits | Tasks | Runtime ms tail | HW ms tail | Sync To Bytes/Iter | Sync From Bytes/Iter | Chain Reuse Bytes/Iter | Ratio vs Fused |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `split_submits` | `multi-submit non-fused` | `[[1], [2], [3], [1], [2], [2]]` | 6 | 9 | 22.816 | 5.474 | 7745536 | 2880000 | 0 | 5.268 |
| `split_tasks` | `one-submit non-fused` | `[[1, 2, 3, 1, 2, 2]]` | 1 | 9 | 11.567 | 2.321 | 5057536 | 2880000 | 4992000 | 2.671 |
| `fused` | `fused` | `[[6, 11, 2]]` | 1 | 6 | 4.331 | 0.636 | 1602816 | 1152000 | 2496000 | 1.000 |

The architectural result for this hero shape is:

- `fused < one-submit non-fused < multi-submit non-fused`
- fewer submit boundaries reduce traffic
- fusion reduces both traffic and hardware work further

## Variant Details

### `split_submits`

- label: `multi-submit non-fused`
- `submit_stage_ids = [[1], [2], [3], [1], [2], [2]]`
- `num_submits = 6`
- `total_tasks = 9`
- `runtime_total_ms_tail = 22.816`
- `runtime_hw_ms_tail = 5.474`
- `data_sync_to_device_bytes = 46473216`
- `data_sync_from_device_bytes = 17280000`
- `chain_reuse_bytes = 0`
- `tir_max_err = 0.000488`

```text
locations(
  buf.x = materialized(name=x, dtype=float16, shape=[1500x64], bytes=192000)
  const.w1 = materialized(name=w1, dtype=float16, shape=[64x256], bytes=32768)
  const.b1 = materialized(name=b1, dtype=float16, shape=[256], bytes=512)
  const.w2 = materialized(name=w2, dtype=float16, shape=[256x64], bytes=32768)
  const.b2 = materialized(name=b2, dtype=float16, shape=[64], bytes=128)
  buf.y = materialized(name=y, dtype=float16, shape=[1500x64], bytes=192000)
  tmp.ff1m = materialized(name=ff1m, dtype=float16, shape=[1500x256], bytes=768000)
  tmp.ff1b = materialized(name=ff1b, dtype=float16, shape=[1500x256], bytes=768000)
  tmp.act = materialized(name=act, dtype=float16, shape=[1500x256], bytes=768000)
  tmp.ff2m = materialized(name=ff2m, dtype=float16, shape=[1500x64], bytes=192000)
  tmp.ff2b = materialized(name=ff2b, dtype=float16, shape=[1500x64], bytes=192000)
)

submit0(
  pc_task_group(
    pc_task[0:2)(
      logical_op = op.matmul,
      computes   = ff1m = x @ w1,
      uses_shorthand = [CNA.matmul, DPU.matmul, CORE],
      reads      = [x@buf.x(float16[1500x64]), w1@const.w1(float16[64x256])],
      writes     = [ff1m@tmp.ff1m(float16[1500x256])],
      split      = rows[[0:1020), [1020:1500)] x cols[[0:256)],
      task_slices = [
        pc_task#0(uses_shorthand=[CNA.matmul, DPU.matmul, CORE], reads=[x@buf.x[0:1020, 0:64](float16[1500x64]), w1@const.w1[0:64, 0:256](float16[64x256])], writes=[ff1m@tmp.ff1m[0:1020, 0:256](float16[1500x256])]),
        pc_task#1(uses_shorthand=[CNA.matmul, DPU.matmul, CORE], reads=[x@buf.x[1020:1500, 0:64](float16[1500x64]), w1@const.w1[0:64, 0:256](float16[64x256])], writes=[ff1m@tmp.ff1m[1020:1500, 0:256](float16[1500x256])]),
      ]
    )
  )
)

submit1(
  pc_task_group(
    pc_task[0:1)(
      logical_op = op.add,
      computes   = ff1b = ff1m + b1,
      uses_shorthand = [DPU.add, DPU_RDMA],
      reads      = [ff1m@tmp.ff1m(float16[1500x256]), b1@const.b1(float16[256])],
      writes     = [ff1b@tmp.ff1b(float16[1500x256])],
      split      = none,
    )
  )
)

submit2(
  pc_task_group(
    pc_task[0:1)(
      logical_op = op.relu,
      computes   = act = relu(ff1b),
      uses_shorthand = [DPU.relu, DPU_RDMA],
      reads      = [ff1b@tmp.ff1b(float16[1500x256])],
      writes     = [act@tmp.act(float16[1500x256])],
      split      = none,
    )
  )
)

submit3(
  pc_task_group(
    pc_task[0:3)(
      logical_op = op.matmul,
      computes   = ff2m = act @ w2,
      uses_shorthand = [CNA.matmul, DPU.matmul, CORE],
      reads      = [act@tmp.act(float16[1500x256]), w2@const.w2(float16[256x64])],
      writes     = [ff2m@tmp.ff2m(float16[1500x64])],
      split      = rows[[0:704), [704:1408), [1408:1500)] x cols[[0:64)],
      task_slices = [
        pc_task#0(uses_shorthand=[CNA.matmul, DPU.matmul, CORE], reads=[act@tmp.act[0:704, 0:256](float16[1500x256]), w2@const.w2[0:256, 0:64](float16[256x64])], writes=[ff2m@tmp.ff2m[0:704, 0:64](float16[1500x64])]),
        pc_task#1(uses_shorthand=[CNA.matmul, DPU.matmul, CORE], reads=[act@tmp.act[704:1408, 0:256](float16[1500x256]), w2@const.w2[0:256, 0:64](float16[256x64])], writes=[ff2m@tmp.ff2m[704:1408, 0:64](float16[1500x64])]),
        pc_task#2(uses_shorthand=[CNA.matmul, DPU.matmul, CORE], reads=[act@tmp.act[1408:1500, 0:256](float16[1500x256]), w2@const.w2[0:256, 0:64](float16[256x64])], writes=[ff2m@tmp.ff2m[1408:1500, 0:64](float16[1500x64])]),
      ]
    )
  )
)

submit4(
  pc_task_group(
    pc_task[0:1)(
      logical_op = op.add,
      computes   = ff2b = ff2m + b2,
      uses_shorthand = [DPU.add, DPU_RDMA],
      reads      = [ff2m@tmp.ff2m(float16[1500x64]), b2@const.b2(float16[64])],
      writes     = [ff2b@tmp.ff2b(float16[1500x64])],
      split      = none,
    )
  )
)

submit5(
  pc_task_group(
    pc_task[0:1)(
      logical_op = op.add,
      computes   = y = ff2b + x,
      uses_shorthand = [DPU.add, DPU_RDMA],
      reads      = [ff2b@tmp.ff2b(float16[1500x64]), x@buf.x(float16[1500x64])],
      writes     = [y@buf.y(float16[1500x64])],
      split      = none,
    )
  )
)

audit(submits=6, pc_tasks=9, blocked_boundaries=0)
```

### `split_tasks`

- label: `one-submit non-fused`
- `submit_stage_ids = [[1, 2, 3, 1, 2, 2]]`
- `num_submits = 1`
- `total_tasks = 9`
- `runtime_total_ms_tail = 11.567`
- `runtime_hw_ms_tail = 2.321`
- `data_sync_to_device_bytes = 30345216`
- `data_sync_from_device_bytes = 17280000`
- `chain_reuse_bytes = 29952000`
- `tir_max_err = 0.000488`

```text
locations(
  buf.x = materialized(name=x, dtype=float16, shape=[1500x64], bytes=192000)
  const.w1 = materialized(name=w1, dtype=float16, shape=[64x256], bytes=32768)
  const.b1 = materialized(name=b1, dtype=float16, shape=[256], bytes=512)
  const.w2 = materialized(name=w2, dtype=float16, shape=[256x64], bytes=32768)
  const.b2 = materialized(name=b2, dtype=float16, shape=[64], bytes=128)
  buf.y = materialized(name=y, dtype=float16, shape=[1500x64], bytes=192000)
  chain.ff1m = internal(name=ff1m, dtype=float16, shape=[1500x256])
  chain.ff1b = internal(name=ff1b, dtype=float16, shape=[1500x256])
  chain.act = internal(name=act, dtype=float16, shape=[1500x256])
  chain.ff2m = internal(name=ff2m, dtype=float16, shape=[1500x64])
  chain.ff2b = internal(name=ff2b, dtype=float16, shape=[1500x64])
)

submit0(
  pc_task_group(
    pc_task[0:2)(
      logical_op = op.matmul,
      computes   = ff1m = x @ w1,
      uses_shorthand = [CNA.matmul, DPU.matmul, CORE],
      reads      = [x@buf.x(float16[1500x64]), w1@const.w1(float16[64x256])],
      writes     = [ff1m@chain.ff1m(float16[1500x256])],
      split      = rows[[0:1020), [1020:1500)] x cols[[0:256)],
      task_slices = [
        pc_task#0(uses_shorthand=[CNA.matmul, DPU.matmul, CORE], reads=[x@buf.x[0:1020, 0:64](float16[1500x64]), w1@const.w1[0:64, 0:256](float16[64x256])], writes=[ff1m@chain.ff1m[0:1020, 0:256](float16[1500x256])]),
        pc_task#1(uses_shorthand=[CNA.matmul, DPU.matmul, CORE], reads=[x@buf.x[1020:1500, 0:64](float16[1500x64]), w1@const.w1[0:64, 0:256](float16[64x256])], writes=[ff1m@chain.ff1m[1020:1500, 0:256](float16[1500x256])]),
      ]
    )
    pc_task[2:3)(
      logical_op = op.add,
      computes   = ff1b = ff1m + b1,
      uses_shorthand = [DPU.add, DPU_RDMA],
      reads      = [ff1m@chain.ff1m(float16[1500x256]), b1@const.b1(float16[256])],
      writes     = [ff1b@chain.ff1b(float16[1500x256])],
      split      = none,
    )
    pc_task[3:4)(
      logical_op = op.relu,
      computes   = act = relu(ff1b),
      uses_shorthand = [DPU.relu, DPU_RDMA],
      reads      = [ff1b@chain.ff1b(float16[1500x256])],
      writes     = [act@chain.act(float16[1500x256])],
      split      = none,
    )
    pc_task[4:7)(
      logical_op = op.matmul,
      computes   = ff2m = act @ w2,
      uses_shorthand = [CNA.matmul, DPU.matmul, CORE],
      reads      = [act@chain.act(float16[1500x256]), w2@const.w2(float16[256x64])],
      writes     = [ff2m@chain.ff2m(float16[1500x64])],
      split      = rows[[0:704), [704:1408), [1408:1500)] x cols[[0:64)],
      task_slices = [
        pc_task#4(uses_shorthand=[CNA.matmul, DPU.matmul, CORE], reads=[act@chain.act[0:704, 0:256](float16[1500x256]), w2@const.w2[0:256, 0:64](float16[256x64])], writes=[ff2m@chain.ff2m[0:704, 0:64](float16[1500x64])]),
        pc_task#5(uses_shorthand=[CNA.matmul, DPU.matmul, CORE], reads=[act@chain.act[704:1408, 0:256](float16[1500x256]), w2@const.w2[0:256, 0:64](float16[256x64])], writes=[ff2m@chain.ff2m[704:1408, 0:64](float16[1500x64])]),
        pc_task#6(uses_shorthand=[CNA.matmul, DPU.matmul, CORE], reads=[act@chain.act[1408:1500, 0:256](float16[1500x256]), w2@const.w2[0:256, 0:64](float16[256x64])], writes=[ff2m@chain.ff2m[1408:1500, 0:64](float16[1500x64])]),
      ]
    )
    pc_task[7:8)(
      logical_op = op.add,
      computes   = ff2b = ff2m + b2,
      uses_shorthand = [DPU.add, DPU_RDMA],
      reads      = [ff2m@chain.ff2m(float16[1500x64]), b2@const.b2(float16[64])],
      writes     = [ff2b@chain.ff2b(float16[1500x64])],
      split      = none,
    )
    pc_task[8:9)(
      logical_op = op.add,
      computes   = y = ff2b + x,
      uses_shorthand = [DPU.add, DPU_RDMA],
      reads      = [ff2b@chain.ff2b(float16[1500x64]), x@buf.x(float16[1500x64])],
      writes     = [y@buf.y(float16[1500x64])],
      split      = none,
    )
  )
)

audit(submits=1, pc_tasks=9, blocked_boundaries=0)
```

### `fused`

- label: `fused`
- `submit_stage_ids = [[6, 11, 2]]`
- `num_submits = 1`
- `total_tasks = 6`
- `runtime_total_ms_tail = 4.331`
- `runtime_hw_ms_tail = 0.636`
- `data_sync_to_device_bytes = 9616896`
- `data_sync_from_device_bytes = 6912000`
- `chain_reuse_bytes = 14976000`
- `tir_max_err = 0.000488`

```text
locations(
  buf.x = materialized(name=x, dtype=float16, shape=[1500x64], bytes=192000)
  const.w1 = materialized(name=w1, dtype=float16, shape=[64x256], bytes=32768)
  const.b1 = materialized(name=b1, dtype=float16, shape=[256], bytes=512)
  const.w2 = materialized(name=w2, dtype=float16, shape=[256x64], bytes=32768)
  const.b2 = materialized(name=b2, dtype=float16, shape=[64], bytes=128)
  buf.y = materialized(name=y, dtype=float16, shape=[1500x64], bytes=192000)
  chain.h1 = internal(name=h1, dtype=float16, shape=[1500x256])
  chain.h2 = internal(name=h2, dtype=float16, shape=[1500x64])
)

submit0(
  pc_task_group(
    pc_task[0:2)(
      logical_op = op.matmul_bias_relu,
      computes   = h1 = relu(x @ w1 + b1),
      uses_shorthand = [CNA.matmul, DPU.matmul_bias_relu, DPU_RDMA, CORE],
      reads      = [x@buf.x(float16[1500x64]), w1@const.w1(float16[64x256]), b1@const.b1(float16[256])],
      writes     = [h1@chain.h1(float16[1500x256])],
      split      = rows[[0:1020), [1020:1500)] x cols[[0:256)],
      task_slices = [
        pc_task#0(uses_shorthand=[CNA.matmul, DPU.matmul_bias_relu, DPU_RDMA, CORE], reads=[x@buf.x[0:1020, 0:64](float16[1500x64]), w1@const.w1[0:64, 0:256](float16[64x256]), b1@const.b1[0:256](float16[256])], writes=[h1@chain.h1[0:1020, 0:256](float16[1500x256])]),
        pc_task#1(uses_shorthand=[CNA.matmul, DPU.matmul_bias_relu, DPU_RDMA, CORE], reads=[x@buf.x[1020:1500, 0:64](float16[1500x64]), w1@const.w1[0:64, 0:256](float16[64x256]), b1@const.b1[0:256](float16[256])], writes=[h1@chain.h1[1020:1500, 0:256](float16[1500x256])]),
      ]
    )
    pc_task[2:5)(
      logical_op = op.matmul_bias,
      computes   = h2 = h1 @ w2 + b2,
      uses_shorthand = [CNA.matmul, DPU.matmul_bias, DPU_RDMA, CORE],
      reads      = [h1@chain.h1(float16[1500x256]), w2@const.w2(float16[256x64]), b2@const.b2(float16[64])],
      writes     = [h2@chain.h2(float16[1500x64])],
      split      = rows[[0:704), [704:1408), [1408:1500)] x cols[[0:64)],
      task_slices = [
        pc_task#2(uses_shorthand=[CNA.matmul, DPU.matmul_bias, DPU_RDMA, CORE], reads=[h1@chain.h1[0:704, 0:256](float16[1500x256]), w2@const.w2[0:256, 0:64](float16[256x64]), b2@const.b2[0:64](float16[64])], writes=[h2@chain.h2[0:704, 0:64](float16[1500x64])]),
        pc_task#3(uses_shorthand=[CNA.matmul, DPU.matmul_bias, DPU_RDMA, CORE], reads=[h1@chain.h1[704:1408, 0:256](float16[1500x256]), w2@const.w2[0:256, 0:64](float16[256x64]), b2@const.b2[0:64](float16[64])], writes=[h2@chain.h2[704:1408, 0:64](float16[1500x64])]),
        pc_task#4(uses_shorthand=[CNA.matmul, DPU.matmul_bias, DPU_RDMA, CORE], reads=[h1@chain.h1[1408:1500, 0:256](float16[1500x256]), w2@const.w2[0:256, 0:64](float16[256x64]), b2@const.b2[0:64](float16[64])], writes=[h2@chain.h2[1408:1500, 0:64](float16[1500x64])]),
      ]
    )
    pc_task[5:6)(
      logical_op = op.add,
      computes   = y = h2 + x,
      uses_shorthand = [DPU.add, DPU_RDMA],
      reads      = [h2@chain.h2(float16[1500x64]), x@buf.x(float16[1500x64])],
      writes     = [y@buf.y(float16[1500x64])],
      split      = none,
    )
  )
)

audit(submits=1, pc_tasks=6, blocked_boundaries=0)
```

