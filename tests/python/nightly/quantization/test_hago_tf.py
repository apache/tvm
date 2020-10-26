import tvm
from tvm import relay

import numpy as np
import argparse
import os

import mxnet as mx
from tvm import hago
from mxnet import gluon

from common_utils import *

try:
  # %tensorflow_version only exists in Colab.
  import tensorflow.compat.v2 as tf
except Exception:
  pass
# tf.enable_v2_behavior()
import tensorflow_hub as hub


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet50", help="model to quantize")
parser.add_argument("--soundness_check", default=False, action='store_true')
parser.add_argument("--skip_fp32", default=False, action='store_true')
parser.add_argument("--run_all", default=False, action='store_true')
args = parser.parse_args()

batch_size = 32
target = 'llvm -mcpu=core-avx2'
ctx = tvm.context(target)


##############################
# Original FP32 TF/Keras model
##############################
tf_hub_links = {
    "resnet50"             : "https://tfhub.dev/tensorflow/resnet_50/classification/1",
    "resnet_v2_50"          : "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/4",
    "mobilenet_v1"          : "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/4",
    "mobilenet_v2"          : "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4",
    "inception_v1"          : "https://tfhub.dev/google/imagenet/inception_v1/classification/4",
    "inception_v2"          : "https://tfhub.dev/google/imagenet/inception_v2/classification/4",
    "inception_v3"          : "https://tfhub.dev/google/imagenet/inception_v3/classification/4",
}


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
        data0 = data[0].asnumpy()
        data0 = np.transpose(data0, axes=[0, 2, 3, 1])
        data = [mx.nd.array(data0)]
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
        scale               = 1.0/255.0,
        # mean_r              = mean_rgb[0],
        # mean_g              = mean_rgb[1],
        # mean_b              = mean_rgb[2],
        # std_r               = std_rgb[0],
        # std_g               = std_rgb[1],
        # std_b               = std_rgb[2],
    )
    return val_data, batch_fn

###############################################################################
# Load the model
# ----------------
def get_model(model_name):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    model = tf.keras.Sequential([
        hub.KerasLayer(tf_hub_links[model_name], output_shape=[1001])
    ])
    img_size = 299 if model_name == 'inceptionv3' else 224
    np_image = np.random.rand(batch_size, img_size, img_size, 3).astype('float32')
    model._set_inputs(np_image)


    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="data"))

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./.tf_saved_model/" + model_name,
                      name="frozen_graph.pb",
                      as_text=False)

    parser = tvm.relay.frontend.TFParser("./.tf_saved_model/"
                                         + model_name +  "/frozen_graph.pb")
    graph_def = parser.parse()
    mod, params = relay.frontend.from_tensorflow(graph_def,
                                                 shape={"data": (batch_size, img_size, img_size, 3)})

    # We assume our model's heavily-layout sensitive operators only consist of nn.conv2d
    desired_layouts = {'nn.conv2d': ['NCHW', 'default']}

    # Convert the layout to NCHW
    # RemoveUnunsedFunctions is used to clean up the graph.
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                    relay.transform.ConvertLayout(desired_layouts)])
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod, params

def ignore_first(tensor):
    if tensor.shape[1] == 1001:
        tensor = tensor[:, 1:]
    return tensor

def main():
    val_path = '/home/ubuntu/tensorflow_datasets/downloads/manual/imagenet2012/val.rec'
    if args.run_all:
        models = tf_hub_links.keys()
    else:
        models = [args.model]
    for model_name in models:
        img_size = 299 if model_name == 'inceptionv3' else 224
        postprocess = ignore_first if 'resnet50' not in model_name else None
        val_data, batch_fn = get_val_data(img_size, val_path, batch_size)

        # Original
        if not args.skip_fp32:
            fp32_mod, params = get_model(model_name)
            func = hago.prerequisite_optimize(fp32_mod['main'], params=params)
            acc = eval_acc(func, val_data, batch_fn, args, var_name='data', target=target, ctx=ctx,
                           postprocess=postprocess)
            print("fp32_accuracy", model_name, acc, sep=',')

        for is_per_channel in [False, True]:
            # Quantize
            calib_dataset = get_calibration_dataset(val_data, batch_fn, var_name='data')
            fp32_mod, params = get_model(model_name)
            qconfig = hago.qconfig(use_channel_quantize=is_per_channel, log_file='temp.log')
            quantized_func = quantize_hago(fp32_mod, params, calib_dataset, qconfig)
            acc = eval_acc(quantized_func, val_data, batch_fn, args, var_name='data', target=target,
                           ctx=ctx, postprocess=postprocess)
            channel_or_tensor = "per_channel" if is_per_channel else "per_tensor"
            print("quantized_accuracy", model_name, channel_or_tensor, acc, sep=',')


if __name__ == '__main__':
    main()
