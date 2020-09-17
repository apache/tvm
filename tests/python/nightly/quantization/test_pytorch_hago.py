import tvm
from tvm import relay

import numpy as np
import argparse

import torch
from torch.nn import Module
import torchvision
from torchvision import transforms
import os

import mxnet as mx
from tvm import hago
from mxnet import gluon

from common_hago import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet50_v1", help="model to quantize")
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
    mean_rgb = [255 * x for x in [0.485, 0.456, 0.406]]
    std_rgb = [255 * x for x in [0.229, 0.224, 0.225]]
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
# Load the model from torchvision
# ----------------
def load_model(model_name):
    """Given a model name, returns a model as well as an example input."""
    if hasattr(torchvision.models, model_name):
        with torch.no_grad():
            if model_name.startswith("inception"):
                height = width = 299
                mean = [0.5, 0.5, 0.5]
                std = [0.5, 0.5, 0.5]
            else:
                height = width = 224
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            input_shape = [batch_size, 3, height, width]
            input_data = torch.randn(input_shape).float()
            for channel in range(3):
                input_data[:, channel] -= mean[channel]
                input_data[:, channel] /= std[channel]
            model = getattr(torchvision.models, model_name)(pretrained=True)
            model = model.float().eval()
            return model, [input_data]
    try:
        import pretrainedmodels
        if hasattr(pretrainedmodels, model_name):
            return load_pretrainedmodels(model_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please install pretrainedmodels.pytorch")
    raise RuntimeError("Model not supported")

def get_model(model_name):
    torch.set_grad_enabled(False)
    baseline_model, baseline_input = load_model(model_name)

    trace = torch.jit.trace(baseline_model, baseline_input)
    if isinstance(baseline_model, torch.nn.Module):
        trace = trace.float().eval()
        trace = trace.cpu()

    global input_names
    input_names = ["input{}".format(idx) for idx, inp in enumerate(baseline_input)]
    input_shapes = list(zip(input_names,
                            [inp.shape for inp in baseline_input]))
    mod, params = relay.frontend.from_pytorch(trace, input_shapes)
    return mod, params


#############
# Test models
#############
def main():
    val_path = '/home/ubuntu/tensorflow_datasets/downloads/manual/imagenet2012/val.rec'
    if args.run_all:
        models = ['resnet50', 'inception_v3', 'mobilenet_v2', 'resnet18',
                  'densenet161', 'googlenet', 'vgg16']
    else:
        models = [args.model]
    for model_name in models:
        height = 224
        if model_name.startswith("inception"):
            height = 299

        val_data, batch_fn = get_val_data(height, val_path, batch_size)

        # Original 
        if not args.skip_fp32:
            fp32_mod, params = get_model(model_name)
            func = hago.prerequisite_optimize(fp32_mod['main'], params=params)
            acc = eval_acc(func, val_data, batch_fn, args, var_name=input_names[0], target=target, ctx=ctx)
            print("fp32_accuracy", model_name, acc, sep=',')
        
        # Quantize 
        calib_dataset = get_calibration_dataset(val_data, batch_fn, var_name=input_names[0])
        fp32_mod, params = get_model(model_name)
        quantized_func = quantize_hago(fp32_mod, params, calib_dataset)
        acc = eval_acc(quantized_func, val_data, batch_fn, args, var_name=input_names[0], target=target, ctx=ctx)
        print("quantized_accuracy", model_name, acc, sep=',')


if __name__ == '__main__':
    main()
