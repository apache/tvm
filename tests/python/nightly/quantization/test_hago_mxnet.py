import tvm
from tvm import relay

import numpy as np
import argparse
import os

import mxnet as mx
from tvm import hago
from mxnet import gluon

from common_utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet18_v1", help="model to quantize")
parser.add_argument("--soundness_check", default=False, action='store_true')
parser.add_argument("--skip_fp32", default=False, action='store_true')
parser.add_argument("--run_all", default=False, action='store_true')
args = parser.parse_args()

batch_size = 32
target = 'llvm -mcpu=core-avx2'
ctx = tvm.context(target)

#####################
# Dataset prepartions
#####################

def get_val_data(img_size,
                 rec_val,
                 batch_size,
                 num_workers=4):
    rec_val = os.path.expanduser(rec_val)
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        preprocess_threads  = num_workers,
        shuffle             = True,
        seed                = 0,
        batch_size          = batch_size,
        resize              = 256,
        data_shape          = (3, img_size, img_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return val_data, batch_fn

###############################################################################
# Load the model
# ----------------
def get_model(model_name):
    gluon_model = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
    img_size = 299 if model_name == 'inceptionv3' else 224
    data_shape = (batch_size, 3, img_size, img_size)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})
    return mod, params

def main():
    # val_path = '/home/ubuntu/tensorflow_datasets/downloads/manual/imagenet2012/val.rec'
    val_path = '/home/ziheng/datasets1/imagenet/rec/val.rec'
    if args.run_all:
        models = ['squeezenet1.1', 'resnet50_v1', 'inceptionv3', 'mobilenetv2_1.0', 'mobilenet1.0', 'resnet18_v1',
                  'vgg16', 'densenet161']
    else:
        models = [args.model]
    for model_name in models:
        img_size = 299 if model_name == 'inceptionv3' else 224
        val_data, batch_fn = get_val_data(img_size, val_path, batch_size)

        if not args.skip_fp32:
            fp32_mod, params = get_model(model_name)
            func = hago.prerequisite_optimize(fp32_mod['main'], params=params)
            acc = eval_acc(func, val_data, batch_fn, args, var_name='data', target=target, ctx=ctx)
            print("fp32_accuracy", model_name, acc, sep=',')

        for is_per_channel in [False, True]:
            # Quantize
            calib_dataset = get_calibration_dataset(val_data, batch_fn, var_name='data')
            fp32_mod, params = get_model(model_name)
            qconfig = hago.qconfig(use_channel_quantize=is_per_channel, log_file='temp.log')
            quantized_func = quantize_hago(fp32_mod, params, calib_dataset, qconfig)
            acc = eval_acc(quantized_func, val_data, batch_fn, args, var_name='data', target=target, ctx=ctx)
            channel_or_tensor = "per_channel" if is_per_channel else "per_tensor"
            print("quantized_accuracy", model_name, channel_or_tensor, acc, sep=',')


if __name__ == '__main__':
    main()
