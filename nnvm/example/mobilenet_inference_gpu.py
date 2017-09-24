"""Forward propagation of MobileNet on GPU."""
import numpy as np
import time
import os

import tvm
import topi
import nnvm.symbol as sym
import nnvm.compiler
import nnvm.runtime
from tvm.contrib import nvcc

TASK="mobilenet"

target = 'cuda'
ctx = tvm.gpu(0)

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx", options=["-arch=sm_60"])
    return ptx

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    return code

dtype = 'float32'
epsilon = 1e-10 + 1e-5

def conv_block(data, name, channels, kernel_size=(3,3), strides=(1,1), padding=(1,1)):
    # convolution + bn + relu
    conv = sym.conv2d(data=data, channels=channels, kernel_size=kernel_size, strides=strides,
        padding=padding, use_bias=False, layout='NCHW', name=name + '_conv')
    bn = sym.batch_norm(data=conv, epsilon=epsilon, name=name + '_bn')
    act = sym.relu(data=bn, name=name + '_relu')
    return act

def separable_conv_block(data, name, depthwise_channels, pointwise_channels, kernel_size=(3,3), downsample=False, padding=(1,1)):
    if downsample:
        strides = (2,2)
    else:
        strides = (1,1)
    # depthwise convolution + bn + relu
    conv1 = sym.conv2d(data=data, channels=depthwise_channels, groups=depthwise_channels, kernel_size=kernel_size, strides=strides,
        padding=padding, use_bias=False, layout='NCHW', name=name + '_conv1')
    bn1 = sym.batch_norm(data=conv1, epsilon=epsilon, name=name + '_bn1')
    act1 = sym.relu(data=bn1, name=name + '_relu1')
    # pointwise convolution + bn + relu
    conv2 = sym.conv2d(data=act1, channels=pointwise_channels, kernel_size=(1,1), strides=(1,1),
        padding=(0,0), use_bias=False, layout='NCHW', name=name + '_conv2')
    bn2 = sym.batch_norm(data=conv2, epsilon=epsilon, name=name + '_bn2')
    act2 = sym.relu(data=bn2, name=name + '_relu2')
    return act2

def mobile_net(num_classes=1000, alpha=1.0, is_shallow=False):
    data = sym.Variable("data")
    body = conv_block(data, 'conv_block_1', int(32*alpha), strides=(2,2))
    body = separable_conv_block(body, 'separable_conv_block_1', int(32*alpha), int(64*alpha))
    body = separable_conv_block(body, 'separable_conv_block_2', int(64*alpha), int(128*alpha), downsample=True)
    body = separable_conv_block(body, 'separable_conv_block_3', int(128*alpha), int(128*alpha))
    body = separable_conv_block(body, 'separable_conv_block_4', int(128*alpha), int(256*alpha), downsample=True)
    body = separable_conv_block(body, 'separable_conv_block_5', int(256*alpha), int(256*alpha))
    body = separable_conv_block(body, 'separable_conv_block_6', int(256*alpha), int(512*alpha), downsample=True)
    if is_shallow:
        body = separable_conv_block(body, 'separable_conv_block_7', int(512*alpha), int(1024*alpha), downsample=True)
        body = separable_conv_block(body, 'separable_conv_block_8', int(1024*alpha), int(1024*alpha))
    else:
        for i in range(7, 12):
            body = separable_conv_block(body, 'separable_conv_block_%d' % i, int(512*alpha), int(512*alpha))
        body = separable_conv_block(body, 'separable_conv_block_12', int(512*alpha), int(1024*alpha), downsample=True)
        body = separable_conv_block(body, 'separable_conv_block_13', int(1024*alpha), int(1024*alpha))
    pool = sym.global_avg_pool2d(data=body, name='pool')
    flatten = sym.flatten(data=pool, name='flatten')
    fc = sym.dense(data=flatten, units=num_classes, use_bias=False, name='fc')
    softmax = sym.softmax(data=fc, name='softmax')
    return softmax


batch_size = 1
num_classes = 1000
image_shape = (3,224,224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_classes)

net = mobile_net(num_classes=num_classes, alpha=1.0, is_shallow=False)

# build graph
with nnvm.compiler.build_config(opt_level=2):
    graph, lib, _ = nnvm.compiler.build(net, target, {'data': data_shape})
# prepare params
params = {}
names = graph.index.input_names
shapes = [graph.json_attr("shape")[graph.index.entry_id(x)] for x in names]
for i in range(len(names)):
    params[names[i]] = tvm.nd.array(np.random.uniform(-0.1, 0.1, size=shapes[i]).astype(dtype), ctx=ctx)
# create runtime module
module = nnvm.runtime.create(graph, lib, ctx)
# set input
module.set_input(**params)
# run
print("run")
module.run()
ctx.sync()
start = time.time()
for i in range(1000):
    module.run()
    ctx.sync()
print("average time cost of 1000 runs = %g ms" % ((time.time() - start)))
# get output
out = module.get_output(0, tvm.nd.empty(out_shape, dtype))
