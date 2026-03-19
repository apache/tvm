# RKNPU Fusion Matrix Reference

This document records the current architectural fusion benchmark for the clean
2-D RKNPU subset on RK3588.

For the current human-readable hero artifact for the MVP residual MLP block,
see [rknpu_mvp_hero_demo.md](/home/user/code/tvm/docs/arch/rknpu_mvp_hero_demo.md).

Use this as the schedule reference for:

- `fused`
- `one-submit non-fused`
- `multi-submit non-fused`

The primary architectural reference is `cache_dma=off`. That mode removes a
large amount of runtime-side DMA cache behavior and is the best proxy we
currently have for raw schedule cost.

## Measurement Protocol

Command used for the architectural reference:

```bash
PYTHONPATH=python:. TVM_LIBRARY_PATH=build TVM_FFI_DISABLE_TORCH_C_DLPACK=1 \
python3 tools/rknpu_performance_reference.py \
  --suite fusion_matrix \
  --host both \
  --comparison-group all \
  --warmup 12 \
  --iters 6 \
  --warmup-mode fixed \
  --bridge-cache-dma off \
  --json-out /tmp/rknpu_fusion_matrix_suite_off.json \
  --markdown-out /tmp/rknpu_fusion_matrix_suite_off.md
```

Runtime metric used below:

- `runtime_total_tail`

Frozen shape family:

- `M in {1, 1500}`
- `D = 64`
- `H = 256`

Compared groups:

- `matmul_bias_relu`
- `residual_mlp_relu`

## Architectural Reference

`cache_dma=off`

| Group | Host | Size | Fused ms | One-submit non-fused ms | Multi-submit non-fused ms | one/fused | multi/fused |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `matmul_bias_relu` | `python` | `small` | 0.380 | 0.479 | 0.716 | 1.260 | 1.885 |
| `matmul_bias_relu` | `python` | `large` | 3.096 | 8.629 | 16.540 | 2.787 | 5.342 |
| `matmul_bias_relu` | `cpp` | `small` | 0.337 | 0.379 | 0.685 | 1.125 | 2.033 |
| `matmul_bias_relu` | `cpp` | `large` | 2.202 | 8.566 | 12.130 | 3.890 | 5.508 |
| `residual_mlp_relu` | `python` | `small` | 0.690 | 0.857 | 1.049 | 1.242 | 1.521 |
| `residual_mlp_relu` | `python` | `large` | 6.063 | 16.849 | 16.807 | 2.779 | 2.772 |
| `residual_mlp_relu` | `cpp` | `small` | 0.678 | 0.792 | 1.261 | 1.167 | 1.859 |
| `residual_mlp_relu` | `cpp` | `large` | 4.352 | 11.364 | 16.294 | 2.611 | 3.744 |

## Current Conclusion

For meaningful shapes, the answer is now stable:

- fused is best
- one-submit non-fused is worse
- multi-submit non-fused is worst or tied-worst

That conclusion is strongest in the `cpp` host measurements with
`cache_dma=off`.

For the current MVP-relevant residual MLP block:

- fused path: `[[6, 11, 2]]`
- one-submit non-fused path: `[[1, 2, 3, 1, 2, 2]]`
- multi-submit non-fused path: `[[1], [2], [3], [1], [2], [2]]`

Large-shape residual MLP result:

- `cpp`, `cache_dma=off`: `4.352 ms < 11.364 ms < 16.294 ms`
- `python`, `cache_dma=off`: `6.063 ms < 16.849 ms ~= 16.807 ms`

So the fused residual MLP schedule is materially better than both unfused
variants, which is the scheduling result we needed for the MVP path.

## Runtime Behavior Note

The same suite was also run with `cache_dma=on`. That mode is useful for
end-to-end runtime behavior, but it should not be the architectural reference.

It can change the relative ordering of the two unfused variants, especially on
small or boundary cases, because persistent DMA/cache behavior becomes part of
the measured cost model.

Even there, the fused path remains clearly best on the large shapes that matter
for the current MVP slice.

## Regeneration

To regenerate both the architectural and runtime views:

```bash
PYTHONPATH=python:. TVM_LIBRARY_PATH=build TVM_FFI_DISABLE_TORCH_C_DLPACK=1 \
python3 tools/rknpu_performance_reference.py \
  --suite fusion_matrix \
  --host both \
  --comparison-group all \
  --warmup 12 \
  --iters 6 \
  --warmup-mode fixed \
  --bridge-cache-dma off \
  --json-out /tmp/rknpu_fusion_matrix_suite_off.json \
  --markdown-out /tmp/rknpu_fusion_matrix_suite_off.md

PYTHONPATH=python:. TVM_LIBRARY_PATH=build TVM_FFI_DISABLE_TORCH_C_DLPACK=1 \
python3 tools/rknpu_performance_reference.py \
  --suite fusion_matrix \
  --host both \
  --comparison-group all \
  --warmup 12 \
  --iters 6 \
  --warmup-mode fixed \
  --bridge-cache-dma on \
  --json-out /tmp/rknpu_fusion_matrix_suite_on.json \
  --markdown-out /tmp/rknpu_fusion_matrix_suite_on.md
```
