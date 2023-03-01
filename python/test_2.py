import onnx

import tvm.relay
from tvm import meta_schedule as ms

mod, params = tvm.relay.frontend.from_onnx(
    onnx.load("resnet50-v1-12-int8.onnx"), shape={"data": [1, 3, 224, 224]}, freeze_params=True
)


def apply_relay_passes(
    mod: tvm.IRModule,
):
    """Applies relay passes to the input IRModule.

    :param mod: The input IRModule
    :return: The IRModule after all the relays passes have been applied
    """
    # N.B. Defer the import so as not to unconditionally require other runtimes.
    from tvm import relay, transform
    from tvm.relay.op.contrib.dnnl import rewrite_layer_norm

    mod = rewrite_layer_norm(mod)

    passes = []

    # If the inputs are static, run DynamicToStatic to remove
    # any residual dynamism in the model.
    # If the inputs are dynamic, this pass is much more expensive
    # and will not remove dynamism from the model, so we skip it.
    passes.append(relay.transform.DynamicToStatic())

    # Infer types prior to the quantization pass below as some
    # transforms might need them.
    passes.append(relay.transform.InferType())

    # Transform fake quantized sub-graphs to actual integer ops.
    # Should have no effect on graphs without the relevant patterns.
    passes.append(relay.transform.FakeQuantizationToInteger())

    passes.append(relay.transform.FastMath())

    # Fold constants after FQ2I becuase some weights are stored in FP32.
    passes.append(relay.transform.FoldConstant())

    # Use sequential to solve for dependent passes
    seq = transform.Sequential(passes)

    with tvm.transform.PassContext(opt_level=4):
        mod = seq(mod)

    mod = relay.transform.InferType()(mod)
    # mod["main"] = rewrite(relay.qnn.transform.LayerNormQuantizedRewrite(), mod["main"])

    return mod


mod = apply_relay_passes(mod)

print(mod)
target = tvm.target.Target("nvidia/geforce-rtx-3070")
work_dir = "resnet_tune"

db = ms.relay_integration.tune_relay(mod, params, target, work_dir, 1000)

"""
Works:
Does not:
57de9e7f3d2711582368903ce95f08b91216b7b5    Mon Nov 28 21:28:37 2022 -0800

"""
