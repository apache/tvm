import argparse
import os

import numpy as np
import torch

import tvm
import tvm.relay as relay
from tvm import rpc
from tvm import autotvm
import tvm.relay.testing
import pickle
import cv2

from tvm.relay.function import Function
from tvm.ir import IRModule
from tvm.relay.expr import Call, Constant, Tuple, GlobalVar, Var, TupleGetItem
from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
import tvm.contrib.graph_executor as runtime
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
import onnx
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms


def make_parser():
    parser = argparse.ArgumentParser("trt_onnx")
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-d", "--device", type=str, default='cuda')
    parser.add_argument("-n", "--relay_only", type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("-ep", "--export_path", type=str, default="./relays")
    parser.add_argument("--input_size", nargs='+', type=int, default=[1, 3, 112, 112],
                        help="input size in list")
    parser.add_argument("--input_name", type=str, default="images",
                        help="input node name")
    parser.add_argument("--input_img", type=str, default="random",
                        help="input data from image or random generated")
    parser.add_argument("--model_name", type=str, default="face_det")
    return parser


def load_graph_exec(device, fname):
    module_path = os.path.join("./", fname)
    if device == "arm_cuda":
        remote = autotvm.measure.request_remote("tx2", "192.168.6.252",
                                                9190, timeout=10000)
        remote.upload(module_path)
        rlib = remote.load_module(fname)
        rdev = remote.device(str(target), 0)
        rmodule = runtime.GraphModule(rlib["default"](rdev))
        return rmodule, rdev
    else:
        lib = tvm.runtime.load_module(module_path)
        dev = tvm.device((str(target)), 0)
        module = runtime.GraphModule(lib["default"](dev))
        return module, dev


if __name__ == '__main__':
    args = make_parser().parse_args()
    target = ""
    cfg = None
    cfg = cfg_mnet

    if args.device == 'cuda':
        target = tvm.target.Target("cuda", host="llvm")
        # set_cuda_target_arch('sm_62')
        # os.environ["TVM_NVCC_PATH"] = "/usr/local/cuda-11.1/bin/nvcc"
        # os.environ["TVM_NVCC_PATH"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin/nvcc.exe"
    elif args.device == 'opencl':
        target = tvm.target.Target("opencl", host="llvm")
    elif args.device == "arm_opencl":
        target = tvm.target.Target("opencl -arch=3.0", host="llvm -mtriple=aarch64-linux-gnu")
    elif args.device == 'arm':
        target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
    elif args.device == 'arm_cuda':
        target = tvm.target.Target("cuda -arch=sm_62", host="llvm -mtriple=aarch64-linux-gnu")
        # set_cuda_target_arch('sm_62')
        os.environ["TVM_NVCC_PATH"] = "/usr/local/cuda-10.1/bin/nvcc"
    elif args.device == "win32":
        #  -mattr=+sse2
        target = "llvm -mtriple=i386-unknown-windows-msvc -mcpu=core-avx2"
    else:
        target = "llvm -mcpu=skylake"
    print(target)

    dso_name = args.model_name + "_" + args.device + ".so"
    path = os.path.join("./", dso_name)
    dtype = "float32"
    input_shape = tuple(args.input_size)
    input_name = args.input_name
    export_path = args.export_path
    if not args.eval:
        onnx_model = onnx.load(args.path)

        # img = np.zeros((1, 3, 480, 640), dtype=np.float)
        img = np.zeros(tuple(args.input_size), dtype=np.float)
        shape_dict = {input_name: img.shape}

        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        # nmod = IRModule(mod)
        # nmod.astext(show_meta_data=False)
        ir_text = mod.astext(show_meta_data=True)
        print(ir_text)

        with open(os.path.join(export_path, args.model_name + ".txt"), "w") as irf:
            irf.write(ir_text)
        mod_bytes = pickle.dumps(mod)
        with open(os.path.join(export_path, args.model_name+".pickle"), "wb") as pick_fn:
            pick_fn.write(mod_bytes)
        params_bytes = relay.save_param_dict(params)
        with open(os.path.join(export_path, args.model_name+".params"), "wb") as pf:
            pf.write(params_bytes)
        # mod = tvm.parser.fromtext(ir_text)
        print(mod)

        if args.relay_only:
            exit(0)

        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)
        # lib.export_library(path, {"cc": "aarch64-linux-gnu-g++"})
        if args.device == "arm_cuda":
            lib.export_library(path, cc="aarch64-linux-gnu-g++")
        else:
            lib.export_library(path)

    # lib = tvm.runtime.load_module(path)
    # dev = tvm.device((str(target)), 0)
    # module = runtime.GraphModule(lib["default"](dev))
    module, dev = load_graph_exec(args.device, dso_name)
    # load image from cv2
    img_raw = cv2.imread(args.input_img)
    face_img = np.float32(img_raw)
    scale = torch.Tensor([face_img.shape[1], face_img.shape[0], face_img.shape[1], face_img.shape[0]])
    face_img = cv2.resize(face_img, (input_shape[3], input_shape[2]))
    face_img = np.transpose(face_img, (2, 0, 1))
    face_img = face_img.astype(dtype)
    # face_img /= 255.0
    face_img = np.expand_dims(face_img, 0)

    # data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype), dev)
    data_tvm = tvm.nd.array(face_img, dev)
    module.set_input(input_name, data_tvm)
    module.run()
    module.run()
    num_inps = module.get_num_outputs()
    loc = module.get_output(0).numpy()
    conf = module.get_output(1).numpy()
    landms = module.get_output(2).numpy()

    priorbox = PriorBox(cfg, image_size=(480, 640))
    priors = priorbox.forward()
    # priors = priors.to(torch.device("cpu"))
    prior_data = priors

    boxes = decode(torch.tensor(loc.squeeze(0)), prior_data,
                   cfg["variance"])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0)[:, 1]
    landms = decode_landm(torch.tensor(landms.squeeze(0)),
                          prior_data,
                          cfg["variance"])
    # scale1 = torch.Tensor([face_img.shape[3], face_img.shape[2], face_img.shape[3], face_img.shape[2],
    #                        face_img.shape[3], face_img.shape[2], face_img.shape[3], face_img.shape[2],
    #                        face_img.shape[3], face_img.shape[2]])

    scale1 = torch.Tensor([img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0],
                           img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0],
                           img_raw.shape[1], img_raw.shape[0]])

    landms = landms * scale1
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > 0.5)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    top_k = 1000
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    nms_threshold = 0.4
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    keep_top_k = 750
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    for b in dets:
        if b[4] < 0.6:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

    name = "test.jpg"
    cv2.imwrite(name, img_raw)

    # for i in range(0, num_inps):
    #     out = module.get_output(i)
    #     np_out = out.numpy()
    #     if i == 1:
    #         cls = np_out[:, :, 1] > np_out[:, :, 0]
    #         # cls = np.array(cls.astype(np.int))
    #         idxs = np.where(cls == True)[1]
    #         for idx in idxs:
    #             print(np_out[0, idx, :])
    #         total_faces = np.sum(cls)
    #         print("total faces " + str(total_faces))
    #     print(out.shape)
    # evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", dev, number=1, repeat=600)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print(
        "Mean inference time (std dev): %.2f ms (%.2f ms)"
        % (np.mean(prof_res), np.std(prof_res))
    )

    exit(0)
