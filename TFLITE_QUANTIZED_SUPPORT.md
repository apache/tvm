# Importing a full-integer-quantized TFLite model into Relax

Notes for TVM developers, from getting a stock `tf.lite` int8 CNN through
`tvm.relax.frontend.tflite` and running it on a CPU backend.

There are three findings. **One is a one-line fix and is included in this
branch. One has already landed upstream. The third is a numerical correctness
bug that is still open, is not fixed here, and is the one worth your attention**
— it silently produces wrong results rather than failing to import.

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

## 3. The quantized average pool computes the wrong values  (OPEN — not fixed here)

This one imports and runs cleanly and gives wrong numbers.

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

### Why it is not fixed in this branch

The obvious fix — make integer `nn.avg_pool2d` round half away from zero —
changes the semantics of a general operator for every integer user, which is a
decision for TVM, not for us. The alternative is to keep `nn.avg_pool2d`
untouched and have the TFLite frontend emit the window sum and the rounded
divide explicitly, which is contained but needs a sum-pooling path the frontend
does not have today.

We took neither: our backend claims the op and implements TFLite's rounding
itself. With that in place the ResNet is **bit-exact against the TFLite
interpreter over 128 random inputs, 0/1280 logits differing**, which is what
establishes that the rounding really is the whole story.

### Suggested regression test

A `from_tflite` round trip on a two-op int8 graph (`CONV_2D` then
`AVERAGE_POOL_2D`) asserting the output matches the TFLite interpreter exactly.
A tolerance-based test will pass while the bug is present — the error is 1 LSB
per pooled value — so the assertion has to be exact.

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
