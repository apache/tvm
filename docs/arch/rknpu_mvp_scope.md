# RKNPU MVP Scope

## Goal

Publish one narrow, defensible result for the TVM TIR RKNPU path:

- real submit on RK3588
- no CPU fallback in the measured path
- exact NumPy correctness checks
- Relax params bound into the compiled artifact; runtime entrypoint is `main(x)`
- native C++ runner used for publishable latency numbers
- a human-auditable near-optimal schedule

This MVP is intentionally narrower than the full backend.

For the current publishable hero artifact for this slice, see
[rknpu_mvp_hero_demo.md](/home/user/code/tvm/docs/arch/rknpu_mvp_hero_demo.md).

## Included Subset

The MVP target is a residual 2-D feed-forward block:

- `ff1 = matmul(x, w1) + b1`
- `act = relu(ff1)`
- `ff2 = matmul(act, w2) + b2`
- `out = ff2 + x`

Tensor shapes:

- `x`: `[M, D]`
- `w1`: `[D, H]`
- `b1`: `[H]`
- `w2`: `[H, D]`
- `b2`: `[D]`

Frozen width family:

- `D = 64`
- `H = 256`

Required sequence/batch family:

- `M in {1, 15, 16, 63, 64, 65, 127, 128, 129, 257, 511, 512, 1024, 1500}`

## Expected Lowering Shape

The intended schedule is one submit with this stage sequence:

- `[[6, 11, 2]]`

Meaning:

- `6`: fused `matmul_bias_relu`
- `11`: fused `matmul_bias`
- `2`: residual tensor add

This is the human-auditable optimality claim for the MVP. If the trace shows
extra submits or extra stage splits for this block, the MVP does not pass.

## Required Correctness Conditions

For every declared `M`:

- NumPy reference match within threshold
- no non-finite outputs
- zero repeat drift beyond threshold
- zero runtime fallback counters
- zero reloc semantic/range mismatches
- exact expected submit-stage sequence
- `blocked_boundary_count == 0`

## Explicitly Out Of Scope

These are not part of the MVP claim:

- softmax
- attention
- layernorm
- conv / `relu_4d`
- 4-D layouts
- arbitrary shape claims outside the frozen family
- unsupported fused stage `7` (`add_relu`)

## Why This Scope

This slice is large enough to be real and useful, but small enough to prove
end-to-end without caveats. It exercises:

- TVM Relax graph construction
- TIR-first RKNPU lowering
- stage fusion
- PC chaining
- runtime bridge submit path
- compile-time schedule reporting
- NumPy-backed correctness validation
