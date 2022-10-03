import argparse
import os
import pickle

from loguru import logger
import numpy as np
import tvm
import tvm.relay as relay
from tvm import rpc
from tvm import autotvm
import tvm.relay.testing

from tvm.relay.function import Function
from tvm.relay.expr import Call, Constant, Tuple, GlobalVar, Var, TupleGetItem
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
from tvm.ir import IRModule
import tvm.contrib.graph_executor as runtime
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch


def make_parser():
    parser = argparse.ArgumentParser("trt_onnx")
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-d", "--device", type=str, default='cuda')
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("-ep", "--export_path", type=str, default="./relays")
    parser.add_argument("--input_size", nargs='+', type=int, default=[1, 3, 112, 112],
                        help="input size in list")
    parser.add_argument("--input_name", type=str, default="images",
                        help="input node name")
    parser.add_argument("--input_img", type=str, default="random",
                        help="input data from image or random generated")
    parser.add_argument("--model_name", type=str, default="face_det")
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--use_relay_text", type=bool, default=False)
    parser.add_argument("--opt_log", type=str, default=None)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    target = ""
    fmt = ".so"
    dtype = "float32"
    if args.fp16:
        dtype = "float16"
    export_path = args.export_path
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
    elif args.device == "arm_opencl":
        target = tvm.target.Target("opencl -model=mali", host="llvm -mtriple=aarch64-linux-gnu")
    elif args.device == "win32":
        #  -mattr=+sse2
        target = "llvm -mtriple=i386-unknown-windows-msvc -mcpu=core-avx2"
        # change -mtriple to -target can compile using clang on linux
        # fmt = ".tar"
    else:
        target = "llvm -mcpu=skylake"
    print(target)


    # mod_path = os.path.join(args.path, "mod.dat")
    # params_path = os.path.join(args.path, "params.dat")
    mod_path = os.path.join(args.path, args.model_name + "_"+dtype+".txt")
    params_path = os.path.join(args.path, args.model_name + "_"+dtype+".params")

    mod = None
    params = None

    filename = args.model_name + "_" + args.device+"_"+dtype + fmt
    if not args.eval:
        if args.use_relay_text:
            with open(mod_path, "r") as mod_fn:
                mod_raw = mod_fn.read()

                mod = tvm.parser.fromtext(mod_raw)
                # nmod = IRModule(mod)
                vars = mod.get_global_vars()
                print(mod)
        else:
            pickle_path = os.path.join(args.path, args.model_name+"_"+dtype + ".pickle")
            with open(pickle_path, "rb") as pickle_fn:
                mod_bytes = pickle_fn.read()
                mod = pickle.loads(mod_bytes)
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

        if str(args.device).startswith("arm"):
            lib.export_library(os.path.join(export_path, filename), cc="aarch64-linux-gnu-g++")
        elif str(args.device) == "win32":
            lib.export_library(os.path.join(export_path, filename), options=["-m32"])
        else:
            lib.export_library(os.path.join(export_path, filename))

    shape_dict = tuple(args.input_size)
    input_name = args.input_name
    if str(args.device) == "win32":
        remote = autotvm.measure.request_remote("rtx3070-win-x86",
                                                "192.168.6.252", 9190, timeout=10000)
        remote.upload(os.path.join(export_path, filename))
        rlib = remote.load_module(filename)

        dev = remote.device(str(target), 0)
        module = runtime.GraphModule(rlib["default"](dev))
        data_tvm = tvm.nd.array((np.random.uniform(size=shape_dict)).astype(dtype), dev)
        module.set_input(input_name, data_tvm)
        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, number=1, repeat=30)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )
        exit(0)
    elif fmt == ".so" and (args.device == "arm_cuda" or args.device == "arm_opencl" or args.device == "arm"):
        if args.device == "arm_cuda":
            remote = autotvm.measure.request_remote("tx2", "192.168.6.69", 9190, timeout=10000)
        else:
            remote = autotvm.measure.request_remote("rk3588", "192.168.6.252", 9190, timeout=10000)
        remote.upload(os.path.join(export_path, filename))
        rlib = remote.load_module(filename)

        dev = remote.device(str(target), 0)
        module = runtime.GraphModule(rlib["default"](dev))

        data_tvm = tvm.nd.array((np.random.uniform(size=shape_dict)).astype(dtype), dev)
        module.set_input(input_name, data_tvm)
        module.run()
        num_outs = module.get_num_outputs()
        for i in range(0, num_outs):
            oval = module.get_output(i).numpy()
            print(oval)
        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, number=1, repeat=300)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )
    else:
        lib = tvm.runtime.load_module(os.path.join(export_path, filename))
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
        data_tvm = tvm.nd.array((np.zeros(shape=shape_dict)).astype(dtype))
        module.set_input(input_name, data_tvm)
        module.run()
        num_outs = module.get_num_outputs()
        for i in range(0, num_outs):
            oval = module.get_output(i).numpy()
            print(oval)
        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )
