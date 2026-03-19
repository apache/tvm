# RKNPU Submit/Task Overhead Reference

This document records the current overhead-oriented microbenchmarks for the
clean 2-D RKNPU path on RK3588.

The goal is to separate:

- one baseline task in one submit
- three tasks in one submit
- three tasks split across three submits

The microbench uses an add-zero identity chain:

- baseline: one `add`
- same-submit: `add -> add -> add` in one submit
- multi-submit: `add || add || add`

This keeps the math simple and stable while exercising real EW tasks.

## Measurement Protocol

Architectural baseline command:

```bash
PYTHONPATH=python:. TVM_LIBRARY_PATH=build TVM_FFI_DISABLE_TORCH_C_DLPACK=1 \
python3 tools/rknpu_performance_reference.py \
  --suite overhead_matrix \
  --host both \
  --warmup 12 \
  --iters 6 \
  --warmup-mode fixed \
  --bridge-cache-dma off \
  --json-out /tmp/rknpu_overhead_matrix_suite_off.json \
  --markdown-out /tmp/rknpu_overhead_matrix_suite_off.md
```

Supplementary runtime-behavior command:

```bash
PYTHONPATH=python:. TVM_LIBRARY_PATH=build TVM_FFI_DISABLE_TORCH_C_DLPACK=1 \
python3 tools/rknpu_performance_reference.py \
  --suite overhead_matrix \
  --host both \
  --warmup 12 \
  --iters 6 \
  --warmup-mode fixed \
  --bridge-cache-dma on \
  --json-out /tmp/rknpu_overhead_matrix_suite_on.json \
  --markdown-out /tmp/rknpu_overhead_matrix_suite_on.md
```

Primary metric:

- `runtime_total_tail`

Shape family:

- small: `[1x64]`
- large: `[1500x64]`

## Architectural Baseline

`cache_dma=off`

| Host | Size | Baseline ms | Same-submit 3-task ms | Multi-submit 3-task ms | Approx extra task ns | Approx extra submit ns |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `python` | `small` | 0.151 | 0.279 | 0.568 | 64,314 | 144,232 |
| `cpp` | `small` | 0.180 | 0.296 | 0.360 | 58,188 | 31,791 |
| `python` | `large` | 2.169 | 5.767 | 6.492 | 1,799,039 | 362,403 |
| `cpp` | `large` | 1.695 | 5.625 | 5.048 | 1,964,855 | -288,465 |

## Interpretation

Small-shape, off-cache numbers are the best current proxy for fixed overhead.

Most useful estimate:

- `cpp`, `small`, `cache_dma=off`
  - extra task in same submit: about `58 us`
  - extra submit boundary: about `32 us`

Python-hosted measurements are larger, especially for extra submits:

- `python`, `small`, `cache_dma=off`
  - extra task in same submit: about `64 us`
  - extra submit boundary: about `144 us`

That is exactly why the native C++ host is a better architectural timing
reference than Python wall-time loops.

## Important Caveat

The `large` cases are not pure fixed-overhead estimates.

Each extra add task still runs real full-tensor EW work on `[1500x64]`, so the
derived deltas include:

- fixed task/submit overhead
- full-tensor EW compute
- data movement for the added task

So use the large-shape rows as:

- cost of adding a real extra full-tensor EW stage

not as:

- pure host/runtime constant overhead

This explains why `cpp`, `large`, `cache_dma=off` shows the three-submit case
slightly faster than the one-submit three-task case. That does not mean “extra
submits are free”; it means the hardware/runtime behavior for a chained 3-task
EW submit is not a pure additive overhead model on large tensors.

## Runtime Behavior View

`cache_dma=on`

| Host | Size | Baseline ms | Same-submit 3-task ms | Multi-submit 3-task ms | Approx extra task ns | Approx extra submit ns |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `python` | `small` | 0.096 | 0.102 | 0.080 | 2,917 | -10,646 |
| `cpp` | `small` | 0.025 | 0.084 | 0.225 | 29,604 | 70,585 |
| `python` | `large` | 1.280 | 3.465 | 3.635 | 1,092,315 | 84,876 |
| `cpp` | `large` | 1.071 | 3.342 | 2.574 | 1,135,629 | -384,133 |

These numbers are useful for current end-to-end runtime behavior, but not for
the architectural schedule reference. Persistent DMA/cache effects are large
enough to distort the “extra submit” estimate, especially on tiny cases.

## Practical Takeaway

Use:

- `fusion_matrix` + `cache_dma=off` for schedule truth
- `overhead_matrix` + `cache_dma=off`, especially `cpp` small, for rough fixed-overhead estimates
- `cache_dma=on` only when you want current runtime behavior rather than raw architectural cost
