import os.path

import torch
import numpy as np
import argparse
import pickle


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
    parser.add_argument("--fp16", type=bool, default=False)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    input_shape = tuple(args.input_size)
    input_name = args.input_name
    img = np.zeros(input_shape, dtype=np.float32)
    ts_model = torch.jit.load(args.path)
    shape_list = [(input_name, img.shape)]
    mod = None
    params = None
    export_path = args.export_path
    dtype = "float32"
    if args.fp16:
        dtype = "float16"
    with open(os.path.join(export_path, args.model_name+"_"+dtype+".txt"), "w") as f:
        import tvm
        import tvm.relay as relay

        mod, params = relay.frontend.from_pytorch(ts_model,
                                                  shape_list)
        print(mod)

        with open(os.path.join(export_path, args.model_name+"_"+dtype+".pickle"), "wb") as pf:
            mod_bytes = pickle.dumps(mod)
            pf.write(mod_bytes)

        ir_text = mod.astext()
        f.write(ir_text)

        param_bytes = relay.save_param_dict(params)
        with open(os.path.join(export_path, args.model_name+"_"+dtype+".params"), "wb") as pf:
            pf.write(param_bytes)
        print("mod and params saved to " + os.path.join(export_path, args.model_name+".txt"))
