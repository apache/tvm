import torch
import numpy as np
import argparse
import pickle


def make_parser():
    parser = argparse.ArgumentParser("trt_onnx")
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-d", "--device", type=str, default='cuda')
    parser.add_argument("-n", "--relay_only", type=bool, default=False)
    parser.add_argument("-ep", "--export_path", type=str, default="./relays")
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    img = np.zeros((1, 1024), dtype=np.float)
    ts_model = torch.jit.load(args.path)
    shape_list = [("stage", img.shape)]
    with open("log.log", "wb") as f:
        import tvm
        import tvm.relay as relay

        mod, params = relay.frontend.from_pytorch(ts_model,
                                                  shape_list)
        print(mod)

        raw_mod = pickle.dumps(mod)
        f.write(raw_mod)
