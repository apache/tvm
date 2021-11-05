#!/usr/bin/env python3
import os

import numpy as np
import tflite
import tvm
from tvm.contrib import graph_executor
from tvm import relay
import argparse

INPUT_NAME = "input_1"


def run_model(mod, params):
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(mod, target=target, params=params)

    rt_mod = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))
    x = np.load("test/test_xiaoai.npy")
    for i in range(len(x)):
        rt_mod.set_input(INPUT_NAME, x[i : i + 1])
        rt_mod.run()
        tvm_res = rt_mod.get_output(0).numpy()
        print(tvm_res)
    print("------------")
    x = np.load("test/test_unknown.npy")
    for i in range(len(x)):
        rt_mod.set_input(INPUT_NAME, x[i : i + 1])
        rt_mod.run()
        tvm_res = rt_mod.get_output(0).numpy()
        print(tvm_res)


def get_model(mode):
    def load_tflite(file):
        tflite_model = tflite.Model.GetRootAsModel(open(file, "rb").read(), 0)
        return relay.frontend.from_tflite(
            tflite_model,
            shape_dict={INPUT_NAME: (1, 99, 12)},
            dtype_dict={INPUT_NAME: "float32"},
        )

    if mode == "float":
        return load_tflite("kws.tflite")

    if mode == "tflite_quant":
        return load_tflite("kws_quant.tflite")

    if mode == "tvm_quant":

        def calibrate_dataset():
            calib_data = np.load("calib.npy")
            for data in calib_data:
                data = np.expand_dims(data, 0)
                yield {INPUT_NAME: data}

        mod, params = load_tflite("kws.tflite")
        with relay.quantize.qconfig(calibrate_mode="kl_divergence", weight_scale="max"):
            mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset())
        return mod, None

    raise Exception(f"unsupported mode: `{mode}`")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--mode", choices=["tflite_quant", "tvm_quant", "float"], required=True
    )
    args = args_parser.parse_args()

    run_model(*get_model(args.mode))
