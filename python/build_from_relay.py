import argparse
import os

from loguru import logger
import numpy as np
import tvm
import tvm.relay as relay
from tvm import rpc
from tvm import autotvm
import tvm.relay.testing
import pickle

from tvm.relay.function import Function
from tvm.relay.expr import Call, Constant, Tuple, GlobalVar, Var, TupleGetItem
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
import tvm.contrib.graph_executor as runtime
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch


def make_parser():
    parser = argparse.ArgumentParser("trt_onnx")
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-d", "--device", type=str, default='cuda')
    parser.add_argument("-n", "--relay_only", type=bool, default=False)
    parser.add_argument("-ep", "--export_path", type=str, default="./relays")
    parser.add_argument("-e", "--eval", type=bool, default=False)
    parser.add_argument("--opt_log", type=str, default=None)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    target = ""
    fmt = ".so"
    if args.device == 'cuda':
        target = tvm.target.Target("cuda", host="llvm")
    elif args.device == 'opencl':
        target = tvm.target.Target("opencl", host="llvm")
    elif args.device == 'arm':
        target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
    elif args.device == 'arm_cuda':
        target = tvm.target.Target("cuda -arch=sm_62", host="llvm -mtriple=aarch64-linux-gnu")
        # set_cuda_target_arch('sm_62')
        os.environ["TVM_NVCC_PATH"] = "/usr/local/cuda-10.1/bin/nvcc"
    elif args.device == "win32":
        #  -mattr=+sse2
        target = "llvm -mtriple=i386-unknown-windows-msvc -mcpu=core-avx2"
        # change -mtriple to -target can compile using clang on linux
        fmt = ".tar"
    else:
        target = "llvm -mcpu=skylake"
    print(target)

    mod_path = os.path.join(args.path, "mod.dat")
    params_path = os.path.join(args.path, "params.dat")

    mod = None
    params = None
    dtype = "float32"
    filename = "yolov7_tiny_"+args.device+fmt
    if not args.eval:
        with open(mod_path, "rb") as mod_fn:
            mod_raw = mod_fn.read()
            mod = pickle.loads(mod_raw)
            print(mod)
        with open(params_path, "rb") as params_fn:
            params_raw = params_fn.read()
            params_array = bytearray(params_raw)
            params = relay.load_param_dict(params_array)
        if args.opt_log is None:
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(mod, target=target,
                                               params=params)
        else:
            with autotvm.apply_history_best(args.opt_log):
                print("opt compiling...")
                with tvm.transform.PassContext(opt_level=3):
                    lib = relay.build_module.build(mod,
                                                   target=target,
                                                   params=params)

        if args.device == "arm_cuda":
            lib.export_library(os.path.join("./", filename), cc="aarch64-linux-gnu-g++")
        else:
            lib.export_library(os.path.join("./", filename))

    if fmt == ".tar":
        remote = autotvm.measure.request_remote("rtx3070-win-x86",
                                                "192.168.6.252", 9190, timeout=10000)
        remote.upload(os.path.join("./", filename))
        rlib = remote.load_module(filename)

        dev = remote.device(str(target), 0)
        module = runtime.GraphModule(rlib["default"](dev))
        data_tvm = tvm.nd.array((np.random.uniform(size=(1, 3, 640, 640))).astype(dtype), dev)
        module.set_input("images", data_tvm)
        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, number=1, repeat=30)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )
        exit(0)
    elif fmt == ".so" and args.device == "arm_cuda":
        remote = autotvm.measure.request_remote("tx2", "192.168.6.252", 9190, timeout=10000)
        remote.upload(os.path.join("./", filename))
        rlib = remote.load_module(filename)

        dev = remote.device(str(target), 0)
        module = runtime.GraphModule(rlib["default"](dev))
        data_tvm = tvm.nd.array((np.random.uniform(size=(1, 3, 640, 640))).astype(dtype), dev)
        module.set_input("images", data_tvm)
        module.run()
        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, number=1, repeat=300)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )
    else:
        lib = tvm.runtime.load_module(os.path.join("./", filename))
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
        data_tvm = tvm.nd.array((np.zeros(shape=(1, 3, 640, 640))).astype(dtype))
        module.set_input("images", data_tvm)
        module.run()
        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )




