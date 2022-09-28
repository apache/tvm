import argparse
import os

import numpy as np

import tvm
import tvm.relay as relay
from tvm import rpc
from tvm import autotvm
import tvm.relay.testing
import cv2

from tvm.relay.function import Function
from tvm.ir import IRModule
import tvm.contrib.graph_executor as runtime
import paddle

if __name__ == '__main__':
    img = np.zeros(shape=(1, 3, 48, 48), dtype=np.float32)
    shape_dict = {"0": img.shape}
    # pmodel = paddle.load("/home/share/data/workspace/project/nn_compiler/tvm/python/models/ppocr/models/ch_PP-OCRv3_rec_infer")
    # paddle.jit.save(pmodel, "/home/share/data/workspace/project/nn_compiler/tvm/python/models/ppocr/models/ch_PP-OCRv3_rec_infer_2")
    model = paddle.jit.load("/home/share/data/workspace/project/nn_compiler/tvm/python/models/ppocr/models/ch_PP-OCRv3_rec_infer")
    mod, params = relay.frontend.from_paddle(model, shape_dict)

    ir_text = mod.astext()
    params_bytes = relay.save_param_dict(params)
    with open("./ocrv3_rec.txt", "w") as ir_fn:
        ir_fn.write(ir_text)
    with open("./ocrv3_rec.params", "wb") as param_fn:
        param_fn.write(params_bytes)
    exit(0)
