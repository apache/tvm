# RKNPU Runtime Reuse Reference

This note records the current runtime-side reuse mechanisms and the concrete
findings from the driver-probe and DMA-debug work.

The point is to separate:

- raw schedule structure
- runtime DMA-buffer reuse
- transform-cache reuse
- true chain-internal intermediate reuse

Those are different mechanisms and they affect benchmark interpretation
differently.

## Tools

- Driver/kernel probe:
  [rknpu_driver_probe.py](/home/user/code/tvm/tools/rknpu_driver_probe.py)
- Main benchmark harness:
  [rknpu_performance_reference.py](/home/user/code/tvm/tools/rknpu_performance_reference.py)

Useful switches:

- `--capture-driver-probe`
- `--capture-devfreq`
- `TVM_RKNPU_BRIDGE_CACHE_DMA=0|1`
- `TVM_RKNPU_BRIDGE_ASSUME_INPUTS_IMMUTABLE=0|1`
- `TVM_RKNPU_BRIDGE_DEBUG_DMA=1`

## What `cache_dma` Actually Means

`TVM_RKNPU_BRIDGE_CACHE_DMA` is not a kernel DMA subsystem feature.

It is a userspace runtime cache in
[rknpu_runtime.cc](/home/user/code/tvm/src/runtime/contrib/rknpu/rknpu_runtime.cc:2763)
that maps:

- `host_ptr -> DMABuffer`

where `DMABuffer` is the allocated kernel-backed NPU buffer object described in
[rknpu_device.h](/home/user/code/tvm/src/runtime/contrib/rknpu/rknpu_device.h:170).

So `cache_dma=on` means:

- reuse previously allocated DMA buffer objects
- reuse their DMA addresses
- avoid repeated alloc/free/mmap churn

It does **not** mean:

- the runtime proved tensor contents are unchanged
- uploads are skipped automatically
- the kernel understands graph-level tensor identity

The allocation/free path being avoided is:

- alloc: [rknpu_device.cc](/home/user/code/tvm/src/runtime/contrib/rknpu/rknpu_device.cc:165)
- free: [rknpu_device.cc](/home/user/code/tvm/src/runtime/contrib/rknpu/rknpu_device.cc:224)

## Upload Skipping Is Separate

Upload skipping is controlled by a different flag:

- `TVM_RKNPU_BRIDGE_ASSUME_INPUTS_IMMUTABLE`

in
[rknpu_runtime.cc](/home/user/code/tvm/src/runtime/contrib/rknpu/rknpu_runtime.cc:2740).

With:

- `cache_dma=on`
- `assume_inputs_immutable=off`

the runtime still reuses DMA buffers but continues to upload/sync contents.

With:

- `cache_dma=on`
- `assume_inputs_immutable=on`

the runtime may skip upload for already-uploaded host pointers.

## Transform Cache Is Also Separate

`TVM_RKNPU_BRIDGE_CACHE_TRANSFORMS` caches CPU-side transformed blobs, such as:

- scattered inputs
- packed weights
- expanded EW/bias layouts

This is separate from DMA-buffer reuse and lives in:

- [rknpu_runtime.cc](/home/user/code/tvm/src/runtime/contrib/rknpu/rknpu_runtime.cc:2699)

Unlike persistent DMA cache, the persistent transform cache includes a checksum
of the source bytes in its key.

## Chain Reuse Is The Important One For Scheduling

Inside one chained submit, if a produced writeback buffer already has the
device layout needed by the next stage, the runtime can reuse that same device
buffer instead of materializing back to host layout and re-uploading.

That logic is in:

- [rknpu_runtime.cc](/home/user/code/tvm/src/runtime/contrib/rknpu/rknpu_runtime.cc:4007)

This is what drives:

- `chain_reuse_hits`
- `chain_reuse_bytes`

This is the reuse mechanism most directly connected to good fusion/chaining.

## Driver Probe Findings

The driver probe was added in:

- [rknpu_driver_probe.py](/home/user/code/tvm/tools/rknpu_driver_probe.py)

and can be attached to benchmark runs through:

- [rknpu_performance_reference.py](/home/user/code/tvm/tools/rknpu_performance_reference.py:1922)

Current finding from the fusion-matrix runs:

- driver freq was flat at `1000000000`
- volt was flat at `800000`
- devfreq governor stayed `rknpu_ondemand`
- devfreq load stayed `100@1000000000Hz`
- no before/after probe drift was observed across cases

So the driver probe is useful as an environment sanity check, but it did not
explain the fused-vs-split differences. Those differences are not coming from
simple clock/governor drift.

## DMA Debug Dump

`TVM_RKNPU_BRIDGE_DEBUG_DMA=1` adds `host_dma_debug` entries to
`get_runtime_bridge_stats()`.

Relevant implementation points:

- env flag:
  [rknpu_runtime.cc](/home/user/code/tvm/src/runtime/contrib/rknpu/rknpu_runtime.cc:2753)
- JSON emission:
  [rknpu_runtime.cc](/home/user/code/tvm/src/runtime/contrib/rknpu/rknpu_runtime.cc:3131)
- capture of reuse flags:
  [rknpu_runtime.cc](/home/user/code/tvm/src/runtime/contrib/rknpu/rknpu_runtime.cc:3853)
  [rknpu_runtime.cc](/home/user/code/tvm/src/runtime/contrib/rknpu/rknpu_runtime.cc:3875)
  [rknpu_runtime.cc](/home/user/code/tvm/src/runtime/contrib/rknpu/rknpu_runtime.cc:4028)
  [rknpu_runtime.cc](/home/user/code/tvm/src/runtime/contrib/rknpu/rknpu_runtime.cc:4470)

Each debug entry records:

- `host_ptr`
- `dma_addr`
- `bytes`
- `write_back`
- `persistent_cached`
- `persistent_cache_hit`
- `upload_skipped`
- `sync_to_device_requested`
- `chain_reused`

## Concrete Finding: Fused vs Split-Submit

Measured case:

- `residual_mlp_relu_large`
- `cache_dma=on`
- Python host

Observed:

- fused path:
  - one submit: `[[6, 11, 2]]`
  - `chain_reused_entries = 2`
  - `chain_reuse_bytes = 4,992,000`
  - `data_sync_to_device_bytes = 3,205,632`
  - `data_sync_from_device_bytes = 2,304,000`
  - tail latency about `3.93 ms`

- split-submit path:
  - six submits: `[[1], [2], [3], [1], [2], [2]]`
  - `chain_reused_entries = 0`
  - `chain_reuse_bytes = 0`
  - `data_sync_to_device_bytes = 15,491,072`
  - `data_sync_from_device_bytes = 5,760,000`
  - tail latency about `6.13 ms`

Important interpretation:

- persistent DMA cache hits occurred in both cases
- upload skipping was still `0` in both cases
- the fused win was therefore not “contents cache magic”
- it was mainly:
  - less materialization
  - less host/device traffic
  - real chain reuse across internal stage boundaries

## Practical Benchmark Rules

Use:

- `cache_dma=off` for schedule truth
- `cache_dma=on` for current runtime behavior
- driver probe as an environment sanity check
- DMA debug dump when a runtime-side reuse explanation is needed

Do not interpret:

- `cache_dma=on`

as proof that extra submits are architecturally cheap. It can hide some buffer
management cost, but it does not remove the materialization and traffic
penalties that show up clearly in the debug dump.
