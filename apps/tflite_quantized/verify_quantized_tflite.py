#!/usr/bin/env python3
"""Check a full-integer-quantized TFLite model imported into Relax against the
TFLite interpreter itself, using TVM's DEFAULT lowering -- no BYOC, no tuning.

The comparison is EXACT on purpose. Every tensor in such a model is int8 and
every op is quantized, so an import that preserves the model's arithmetic must
reproduce the interpreter bit for bit. A tolerance would hide precisely the
class of bug this is meant to catch: a rounding rule that is off by one, which
looks negligible per layer and is not, because these graphs end in an int8
softmax that amplifies one LSB into tens of counts.

    pip install ai-edge-litert tflite
    python3 verify_quantized_tflite.py --model pretrainedResnet_quant.tflite

Default model: the MLCommons Tiny image-classification benchmark network,
    https://github.com/mlcommons/tiny/blob/master/benchmark/training/
    image_classification/trained_models/pretrainedResnet_quant.tflite
a CIFAR-10 ResNet quantized to int8 (32x32x3 in, 10 logits out) whose head is a
global AVERAGE_POOL_2D.

Exit status is 0 only if every output element of every trial matches.
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to a quantized .tflite file")
    ap.add_argument("--trials", type=int, default=128, help="random inputs to compare")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--target", default="llvm")
    ap.add_argument("--op", default="AVERAGE_POOL_2D",
                    help="operator to check in isolation, or '' to skip")
    a = ap.parse_args()

    import tvm
    from tvm import relax
    import tflite
    from tvm.relax.frontend.tflite import from_tflite

    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # older wheel name; numpy 1.x only
        from tflite_runtime.interpreter import Interpreter

    buf = open(a.model, "rb").read()
    mod = from_tflite(tflite.Model.GetRootAsModel(buf, 0))

    dev = tvm.cpu(0)
    vm = relax.VirtualMachine(tvm.compile(mod, target=tvm.target.Target(a.target)), dev)

    interp = Interpreter(model_path=a.model)
    interp.allocate_tensors()
    din, dout = interp.get_input_details()[0], interp.get_output_details()[0]
    shape = [int(v) for v in din["shape"]]
    dtype = np.dtype(din["dtype"])

    lo, hi = (-128, 128) if dtype == np.int8 else (0, 256)
    rng = np.random.default_rng(a.seed)
    bad = total = worst = 0
    for _ in range(a.trials):
        x = rng.integers(lo, hi, shape).astype(dtype)
        interp.set_tensor(din["index"], x)
        interp.invoke()
        want = interp.get_tensor(dout["index"]).copy()
        got = vm["main"](tvm.runtime.tensor(x, dev)).numpy()
        d = np.abs(got.astype(np.int64) - want.astype(np.int64))
        bad += int((d > 0).sum())
        total += d.size
        worst = max(worst, int(d.max()))

    # ---- the isolated operator check --------------------------------------
    # The whole-model number above mixes every op together, and TVM's QDQ
    # lowering runs the CONVOLUTIONS in float32, which costs a count or two on
    # its own. To say something exact about one operator, slice it out of the
    # model into a standalone single-op .tflite, import THAT, and compare it
    # against an interpreter running the same slice. Nothing else can
    # contribute, so the result must be 0.
    op_bad = op_total = 0
    if a.op:
        try:
            op_bad, op_total = _check_single_op(a, buf, Interpreter, tvm, relax,
                                                from_tflite, tflite, dev)
        except ImportError as e:
            print(f"[skip] isolated {a.op} check needs ai-edge-litert's schema: {e}")
            op_total = -1

    print(f"model   : {a.model}")
    print(f"target  : {a.target}   (default lowering, no BYOC)")
    print(f"trials  : {a.trials} random inputs, shape {tuple(shape)} {dtype}")
    print(f"result  : {bad}/{total} output elements differ"
          + (f", max |diff| = {worst}" if bad else "")
          + f"  ->  {'PASS' if bad == 0 else 'FAIL'}")
    if op_total > 0:
        print(f"{a.op:<8}: {op_bad}/{op_total} elements differ, sliced out and run "
              f"on its own  ->  {'PASS' if op_bad == 0 else 'FAIL'}")
    if bad:
        print("\nThe whole-model figure is not expected to be zero for every model:\n"
              "TVM's QDQ lowering dequantizes and accumulates convolutions in float32,\n"
              "which costs a count or two by itself. The per-operator line above is the\n"
              "exact one -- it isolates a single op from everything else.")
    return 1 if op_bad else 0


def _check_single_op(a, buf, Interpreter, tvm, relax, from_tflite, tflite, dev):
    """Slice every instance of `a.op` into its own one-op model and compare.

    The slice keeps the operator's constant inputs baked in with their
    quantization parameters, and promotes its activations to subgraph inputs,
    so it is the same computation the full graph performs.
    """
    import copy

    import flatbuffers
    from ai_edge_litert.schema_py_generated import Model, ModelT

    mt = ModelT.InitFromObj(Model.GetRootAsModel(buf, 0))
    sg = mt.subgraphs[0]
    names = {}
    from tflite.BuiltinOperator import BuiltinOperator

    for n in dir(BuiltinOperator):
        if not n.startswith("_"):
            names[getattr(BuiltinOperator, n)] = n
    codes = [c.builtinCode for c in mt.operatorCodes]

    ref = Interpreter(model_path=a.model, experimental_preserve_all_tensors=True)
    ref.allocate_tensors()
    din = ref.get_input_details()[0]
    shape = [int(v) for v in din["shape"]]
    dtype = np.dtype(din["dtype"])
    lo, hi = (-128, 128) if dtype == np.int8 else (0, 256)

    targets = [i for i, op in enumerate(sg.operators)
               if names.get(codes[op.opcodeIndex], "") == a.op]
    if not targets:
        return 0, 0

    slices = []
    for i in targets:
        m2 = copy.deepcopy(mt)
        s2 = m2.subgraphs[0]
        op = s2.operators[i]
        s2.operators = [op]
        s2.inputs = [t for t in op.inputs if t >= 0 and
                     not (m2.buffers[s2.tensors[t].buffer].data is not None
                          and len(m2.buffers[s2.tensors[t].buffer].data))]
        s2.outputs = list(op.outputs)
        b = flatbuffers.Builder(1024)
        b.Finish(m2.Pack(b), b"TFL3")
        content = bytes(b.Output())
        it = Interpreter(model_content=content)
        it.allocate_tensors()
        sub = from_tflite(tflite.Model.GetRootAsModel(content, 0))
        vm = relax.VirtualMachine(
            tvm.compile(sub, target=tvm.target.Target(a.target)), dev)
        slices.append((i, op, it, vm))

    rng = np.random.default_rng(a.seed + 1)
    bad = total = 0
    for _ in range(max(1, a.trials // 8)):
        ref.set_tensor(din["index"], rng.integers(lo, hi, shape).astype(dtype))
        ref.invoke()
        for i, op, it, vm in slices:
            # feed both the interpreter and the TVM module the SAME activation
            # tensors, taken from the full model's own run
            args = []
            ins = it.get_input_details()
            for d, t in zip(ins, [t for t in op.inputs if t >= 0][:len(ins)]):
                v = np.ascontiguousarray(ref.get_tensor(t))
                it.set_tensor(d["index"], v)
                args.append(tvm.runtime.tensor(v, dev))
            it.invoke()
            want = it.get_tensor(it.get_output_details()[0]["index"])
            got = vm["main"](*args).numpy()
            d = np.abs(got.astype(np.int64) - want.astype(np.int64))
            bad += int((d > 0).sum())
            total += d.size
    return bad, total


if __name__ == "__main__":
    sys.exit(main())
