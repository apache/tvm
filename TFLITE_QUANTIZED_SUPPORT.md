# Importing a full-integer-quantized TFLite model into Relax

Notes for TVM developers, from getting a stock `tf.lite` int8 CNN through
`tvm.relax.frontend.tflite` and running it on a CPU backend.

There are three findings. Two are fixed in this branch; one had already landed
upstream. The second fix is the important one: it is a **numerical correctness
bug that silently produced wrong results** rather than failing to import.

    apps/tflite_quantized/verify_quantized_tflite.py

reproduces all of it against the TFLite interpreter, using TVM's default
lowering with no BYOC. On the model below:

| | isolated `AVERAGE_POOL_2D` | whole model |
|---|---|---|
| before | **510 / 1024 elements wrong (50%)** | 312 / 1280 logits, max 106 |
| after | **0 / 1024 — exact** | 22 / 1280, max 23 |

Reproducer used throughout: the MLCommons Tiny image-classification benchmark
model, a CIFAR-10 ResNet exported with `tf.lite.Optimize.DEFAULT` and an int8
representative dataset, so every tensor is int8 and every op is quantized.

<https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite>

```python
import tflite
from tvm.relax.frontend.tflite import from_tflite
buf = open("pretrainedResnet_quant.tflite", "rb").read()
mod = from_tflite(tflite.Model.GetRootAsModel(buf, 0))
```

---

## 1. `AVERAGE_POOL_2D` is implemented but not listed as quantized-capable  (fixed here)

### Symptom

```
OpNotImplemented: The following quantized TFLite operators are not supported in
frontend TFLite yet: 'AVERAGE_POOL_2D'.
```

### Cause

`OperatorConverter._SUPPORTED_QUANTIZED_OPS` is the allowlist checked *before*
dispatching, and `AVERAGE_POOL_2D` is missing from it — even though
`convert_pool2d` already has a complete quantized branch for
`pool_type="average"`. The op is implemented and unreachable at the same time.

The set already contains `MEAN`, `REDUCE_MAX`, `RESIZE_BILINEAR` and the other
pooling-adjacent ops, so this reads as an omission rather than a decision.
`MAX_POOL_2D` is absent for what looks like the same reason; we did not need it,
so we neither added nor tested it.

### Fix (this branch)

```diff
             "ABS",
             "ADD",
             "ATAN2",
+            "AVERAGE_POOL_2D",
             "CEIL",
```

### Why it matters

Global average pooling is the classifier head of essentially every
MobileNet / ResNet / EfficientNet variant, so this rejects most quantized image
classifiers at import.

---

## 2. `relax.op.cast` does not exist — already fixed upstream

Recorded only so the history is clear. The quantized average-pool branch used
to call `relax.op.cast`, which is the *Relay* spelling; `relax.op` exports
`astype`. Against `v0.26.dev0-198-g67bd1ea1a` this raised

```
AttributeError: module 'tvm.relax.op' has no attribute 'cast'
```

as soon as finding 1 made the branch reachable — the two bugs hid each other.
Current `main` already uses `astype`, so nothing is needed here.

---

## 3. The quantized average pool computed the wrong values  (fixed here)

This one imported and ran cleanly and gave wrong numbers.

### The arithmetic

With finding 1 applied, the frontend emits — correctly, in the integer domain,
with no float round trip:

```python
out = relax.op.astype(in_expr, "int32")
out = relax.op.nn.avg_pool2d(out, **params)
out = relax.op.astype(out, output_tensor_type_str)
```

`nn.avg_pool2d` on an integer tensor divides the window sum with a **truncating**
division. TFLite's quantized `AveragePool` rounds **half away from zero**
(`tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h`):

```c
acc = acc > 0 ? (acc + filter_count / 2) / filter_count
               : (acc - filter_count / 2) / filter_count;
```

Dropping the `±filter_count/2` biases every pooled value toward zero by up to
half an LSB. It is a systematic bias, not a rounding tie: it is wrong about half
the time.

### Measured

Against the TFLite interpreter on this model's 8x8 -> 1x1 pool, feeding it the
interpreter's own input tensor so nothing else can contribute:

| model of the divide | pooled values differing |
|---|---|
| TFLite round-half-away | **0 / 1024** |
| **truncate toward zero** (what relax does) | **537 / 1024  (52%)** |
| floor | 428 / 1024 |

End to end it is much louder than that ratio suggests, because the graph ends in
an int8 `SOFTMAX`: half an LSB on a pooled feature reaches the logits as tens of
counts. Substituting only this op into an otherwise bit-exact execution of the
whole graph:

| pool implementation | output logits differing vs the TFLite interpreter |
|---|---|
| truncating divide | **82 / 320, max error 91** |
| round half away from zero | **0 / 320** |

The 82/320 is also what TVM produces end to end for this model, so this single
op accounts for essentially all of the divergence.

### The fix

`nn.avg_pool2d` is left alone -- changing the rounding of a general operator
would change semantics for every integer user, which is a separate discussion.
Instead the frontend now takes the window SUM and does TFLite's division
explicitly:

```python
window = filter_h * filter_w
acc = relax.op.astype(in_expr, "int32")
acc = relax.op.multiply(acc, relax.const(window, "int32"))
acc = relax.op.nn.avg_pool2d(acc, count_include_pad=True, **params)
acc = relax.op.astype(acc, "int32")
half = relax.const(counts // 2, "int32")
out = relax.op.where(relax.op.greater(acc, relax.const(0, "int32")),
                     relax.op.add(acc, half),
                     relax.op.subtract(acc, half))
out = relax.op.divide(out, relax.const(counts, "int32"))
```

Four things make this work, each verified rather than assumed:

* **The sum is exact.** `avg_pool2d` divides by the window size, so pre-scaling
  the input by that size makes its division exact and leaves the sum behind.
  `|acc| <= 255 * window^2` for 8-bit input, which the code asserts fits int32.
* **`count_include_pad=True` keeps that divisor constant.** The padded taps are
  zeros, so the sum over the padded window is the sum over the valid taps.
* **`relax.op.divide` on int32 truncates toward zero**, which is the semantics
  TFLite's `(acc ± count/2) / count` is written against. Confirmed by probing
  it on negative operands.
* **`counts` is the number of NON-padded taps**, which varies per output
  position under SAME padding. Shapes are static, so it is folded to a constant
  `[1, OH, OW, 1]` array at import time instead of being computed in the graph.

One subtlety worth knowing if you touch this: legalization widens the pooling
accumulator (TOPI uses int64 for integer pools), so the result is pinned back
to int32 with an `astype` before it meets the int32 rounding constants —
without it the import fails with a binary-op dtype mismatch.

### Verifying it

```
pip install ai-edge-litert tflite
python3 apps/tflite_quantized/verify_quantized_tflite.py --model pretrainedResnet_quant.tflite
```

The script does two things. It runs the whole model through TVM's default
lowering against the interpreter, and it slices each `AVERAGE_POOL_2D` out into
a standalone one-op model, imports that, and compares it on its own — which is
the exact test, because nothing else can contribute to it. Its exit status
follows the per-operator result.

Both comparisons are **exact, with no tolerance**, on purpose: the error this
catches is 1 LSB per pooled value, which any tolerance would hide.

**The whole-model number is not zero, and that is a different issue.** TVM's
QDQ lowering dequantizes and accumulates the convolutions in float32, which
costs a count or two by itself; a backend that keeps the convolution in int32
gets the model bit-exact. That is why the per-operator line is the one that
carries the claim here.

Note that TFLite's int8 average pool requires input and output to share a scale
and zero point (the frontend already asserts this), so the reference is a pure
integer average of the raw quantized bytes with no rescale.

---

## What this does NOT need

Worth stating because it is the obvious guess and it is wrong: **no layout work
is required.** ONNX forces NCHW, and `relax.transform.ConvertLayout` cannot
convert a QDQ graph anyway because `relax.quantize` / `relax.dequantize` carry
no `FRelaxInferLayout`. The TFLite frontend sidesteps that by emitting NHWC
natively, which is what a CPU backend wants. With finding 1 applied the import
produces exactly the QDQ shape a quantized conv should have:

```
lv  = R.dequantize(x,      scale_x, zp_x)     # int8   -> float32
lv1 = R.dequantize(weight, scale_w, zp_w)     # int8   -> float32, per-channel
lv2 = R.nn.conv2d(lv, lv1, data_layout="NHWC", kernel_layout="HWIO")
lv3 = R.dequantize(bias,   scale_b, zp_b)     # int32  -> float32
lv4 = R.add(lv2, lv3)
lv5 = R.quantize(lv4, scale_o, zp_o)          # float32 -> int8
```

Every scale and zero point is a compile-time constant, so a backend can fold all
three into a single per-channel requantization scale at compile time.

---

## Unrelated, recorded because it costs time: `export_library` picks the wrong triple

`Module.export_library` takes the LLVM target for its packed-imports object
(`devc.o`) from the first LLVM module it finds, and falls back to
`fcompile.get_target_triple()` when there is none. A graph fully offloaded to a
BYOC backend leaves no TIR and hence no LLVM module, so the default
`create_shared` reports the *build host's* triple and cross-compilation fails:

```
ld: unknown architecture of input file `.../devc.o' is incompatible with aarch64 output
```

Caller-side workaround:

```python
ex.export_library(so, fcompile=_cc.cross_compiler(
    CXX, options=[...], get_target_triple=_cc.get_target_by_dump_machine(CXX)))
```

A model that keeps *any* TIR hides this, which is why it shows up on a
single-conv test and not on a full network.
