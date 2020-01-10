# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile Darknet Models
=====================
This article is a test script to test darknet models with NNVM.
All the required models and libraries will be downloaded from the internet
by the script.
"""
import numpy as np
import tvm
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
download_testdata.__test__ = False
from nnvm import frontend
from tvm.relay.testing.darknet import LAYERTYPE
from tvm.relay.testing.darknet import __darknetffi__
import nnvm.compiler

DARKNET_LIB = 'libdarknet2.0.so'
DARKNETLIB_URL = 'https://github.com/siju-samuel/darknet/blob/master/lib/' \
                                    + DARKNET_LIB + '?raw=true'
LIB = __darknetffi__.dlopen(download_testdata(DARKNETLIB_URL, DARKNET_LIB, module='darknet'))

DARKNET_TEST_IMAGE_NAME = 'dog.jpg'
DARKNET_TEST_IMAGE_URL = 'https://github.com/siju-samuel/darknet/blob/master/data/' + DARKNET_TEST_IMAGE_NAME +'?raw=true'
DARKNET_TEST_IMAGE_PATH = download_testdata(DARKNET_TEST_IMAGE_URL, DARKNET_TEST_IMAGE_NAME, module='data')

def _read_memory_buffer(shape, data, dtype='float32'):
    length = 1
    for x in shape:
        length *= x
    data_np = np.zeros(length, dtype=dtype)
    for i in range(length):
        data_np[i] = data[i]
    return data_np.reshape(shape)

def _get_tvm_output(net, data, build_dtype='float32'):
    '''Compute TVM output'''
    dtype = 'float32'
    sym, params = frontend.darknet.from_darknet(net, dtype)

    target = 'llvm'
    shape_dict = {'data': data.shape}
    graph, library, params = nnvm.compiler.build(sym, target, shape_dict,
                                                 build_dtype, params=params)
    # Execute on TVM
    ctx = tvm.cpu(0)
    m = graph_runtime.create(graph, library, ctx)
    # set inputs
    m.set_input('data', tvm.nd.array(data.astype(dtype)))
    m.set_input(**params)
    m.run()
    # get outputs
    tvm_out = []
    for i in range(m.get_num_outputs()):
        tvm_out.append(m.get_output(i).asnumpy())
    return tvm_out

def _load_net(cfg_url, cfg_name, weights_url, weights_name):
    cfg_path = download_testdata(cfg_url, cfg_name, module='darknet')
    weights_path = download_testdata(weights_url, weights_name, module='darknet')
    net = LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
    return net

def verify_darknet_frontend(net, build_dtype='float32'):
    '''Test network with given input image on both darknet and tvm'''
    def get_darknet_output(net, img):
        LIB.network_predict_image(net, img)
        out = []
        for i in range(net.n):
            layer = net.layers[i]
            if layer.type == LAYERTYPE.REGION:
                attributes = np.array([layer.n, layer.out_c, layer.out_h,
                                       layer.out_w, layer.classes,
                                       layer.coords, layer.background],
                                      dtype=np.int32)
                out.insert(0, attributes)
                out.insert(0, _read_memory_buffer((layer.n*2, ), layer.biases))
                layer_outshape = (layer.batch, layer.out_c,
                                  layer.out_h, layer.out_w)
                out.insert(0, _read_memory_buffer(layer_outshape, layer.output))
            elif layer.type == LAYERTYPE.YOLO:
                attributes = np.array([layer.n, layer.out_c, layer.out_h,
                                       layer.out_w, layer.classes,
                                       layer.total],
                                      dtype=np.int32)
                out.insert(0, attributes)
                out.insert(0, _read_memory_buffer((layer.total*2, ), layer.biases))
                out.insert(0, _read_memory_buffer((layer.n, ), layer.mask, dtype='int32'))
                layer_outshape = (layer.batch, layer.out_c,
                                  layer.out_h, layer.out_w)
                out.insert(0, _read_memory_buffer(layer_outshape, layer.output))
            elif i == net.n-1:
                if layer.type == LAYERTYPE.CONNECTED:
                    darknet_outshape = (layer.batch, layer.out_c)
                elif layer.type in [LAYERTYPE.SOFTMAX]:
                    darknet_outshape = (layer.batch, layer.outputs)
                else:
                    darknet_outshape = (layer.batch, layer.out_c,
                                        layer.out_h, layer.out_w)
                out.insert(0, _read_memory_buffer(darknet_outshape, layer.output))
        return out

    dtype = 'float32'

    img = LIB.letterbox_image(LIB.load_image_color(DARKNET_TEST_IMAGE_PATH.encode('utf-8'), 0, 0), net.w, net.h)
    darknet_output = get_darknet_output(net, img)
    batch_size = 1
    data = np.empty([batch_size, img.c, img.h, img.w], dtype)
    i = 0
    for c in range(img.c):
        for h in range(img.h):
            for k in range(img.w):
                data[0][c][h][k] = img.data[i]
                i = i + 1

    tvm_out = _get_tvm_output(net, data, build_dtype)
    for tvm_outs, darknet_out in zip(tvm_out, darknet_output):
        tvm.testing.assert_allclose(darknet_out, tvm_outs, rtol=1e-3, atol=1e-3)

def verify_rnn_forward(net):
    '''Test network with given input data on both darknet and tvm'''
    def get_darknet_network_predict(net, data):
        return LIB.network_predict(net, data)
    from cffi import FFI
    ffi = FFI()
    np_arr = np.zeros([1, net.inputs], dtype='float32')
    np_arr[0, 84] = 1
    cffi_arr = ffi.cast('float*', np_arr.ctypes.data)
    tvm_out = _get_tvm_output(net, np_arr)[0]
    darknet_output = get_darknet_network_predict(net, cffi_arr)
    darknet_out = np.zeros(net.outputs, dtype='float32')
    for i in range(net.outputs):
        darknet_out[i] = darknet_output[i]
    last_layer = net.layers[net.n-1]
    darknet_outshape = (last_layer.batch, last_layer.outputs)
    darknet_out = darknet_out.reshape(darknet_outshape)
    tvm.testing.assert_allclose(darknet_out, tvm_out, rtol=1e-4, atol=1e-4)

def test_forward_extraction():
    '''test extraction model'''
    model_name = 'extraction'
    cfg_name = model_name + '.cfg'
    weights_name = model_name + '.weights'
    cfg_url = 'https://github.com/pjreddie/darknet/blob/master/cfg/' + cfg_name + '?raw=true'
    weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'
    net = _load_net(cfg_url, cfg_name, weights_url, weights_name)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_alexnet():
    '''test alexnet model'''
    model_name = 'alexnet'
    cfg_name = model_name + '.cfg'
    weights_name = model_name + '.weights'
    cfg_url = 'https://github.com/pjreddie/darknet/blob/master/cfg/' + cfg_name + '?raw=true'
    weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'
    net = _load_net(cfg_url, cfg_name, weights_url, weights_name)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_resnet50():
    '''test resnet50 model'''
    model_name = 'resnet50'
    cfg_name = model_name + '.cfg'
    weights_name = model_name + '.weights'
    cfg_url = 'https://github.com/pjreddie/darknet/blob/master/cfg/' + cfg_name + '?raw=true'
    weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'
    net = _load_net(cfg_url, cfg_name, weights_url, weights_name)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_yolov2():
    '''test yolov2 model'''
    model_name = 'yolov2'
    cfg_name = model_name + '.cfg'
    weights_name = model_name + '.weights'
    cfg_url = 'https://github.com/pjreddie/darknet/blob/master/cfg/' + cfg_name + '?raw=true'
    weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'
    net = _load_net(cfg_url, cfg_name, weights_url, weights_name)
    build_dtype = {}
    verify_darknet_frontend(net, build_dtype)
    LIB.free_network(net)

def test_forward_yolov3():
    '''test yolov3 model'''
    model_name = 'yolov3'
    cfg_name = model_name + '.cfg'
    weights_name = model_name + '.weights'
    cfg_url = 'https://github.com/pjreddie/darknet/blob/master/cfg/' + cfg_name + '?raw=true'
    weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'
    net = _load_net(cfg_url, cfg_name, weights_url, weights_name)
    build_dtype = {}
    verify_darknet_frontend(net, build_dtype)
    LIB.free_network(net)

def test_forward_convolutional():
    '''test convolutional layer'''
    net = LIB.make_network(1)
    layer = LIB.make_convolutional_layer(1, 224, 224, 3, 32, 1, 3, 2, 0, 1, 0, 0, 0, 0)
    net.layers[0] = layer
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_dense():
    '''test fully connected layer'''
    net = LIB.make_network(1)
    layer = LIB.make_connected_layer(1, 75, 20, 1, 0, 0)
    net.layers[0] = layer
    net.w = net.h = 5
    LIB.resize_network(net, 5, 5)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_dense_batchnorm():
    '''test fully connected layer with batchnorm'''
    net = LIB.make_network(1)
    layer = LIB.make_connected_layer(1, 12, 2, 1, 1, 0)
    for i in range(5):
        layer.rolling_mean[i] = np.random.rand(1)
        layer.rolling_variance[i] = np.random.rand(1)
        layer.scales[i] = np.random.rand(1)
    net.layers[0] = layer
    net.w = net.h = 2
    LIB.resize_network(net, 2, 2)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_maxpooling():
    '''test maxpooling layer'''
    net = LIB.make_network(1)
    layer = LIB.make_maxpool_layer(1, 224, 224, 3, 2, 2, 0)
    net.layers[0] = layer
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_avgpooling():
    '''test avgerage pooling layer'''
    net = LIB.make_network(1)
    layer = LIB.make_avgpool_layer(1, 224, 224, 3)
    net.layers[0] = layer
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_batch_norm():
    '''test batch normalization layer'''
    net = LIB.make_network(1)
    layer = LIB.make_convolutional_layer(1, 224, 224, 3, 32, 1, 3, 2, 0, 1, 1, 0, 0, 0)
    for i in range(32):
        layer.rolling_mean[i] = np.random.rand(1)
        layer.rolling_variance[i] = np.random.rand(1)
    net.layers[0] = layer
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_shortcut():
    '''test shortcut layer'''
    net = LIB.make_network(3)
    layer_1 = LIB.make_convolutional_layer(1, 224, 224, 3, 32, 1, 3, 2, 0, 1, 0, 0, 0, 0)
    layer_2 = LIB.make_convolutional_layer(1, 111, 111, 32, 32, 1, 1, 1, 0, 1, 0, 0, 0, 0)
    layer_3 = LIB.make_shortcut_layer(1, 0, 111, 111, 32, 111, 111, 32)
    layer_3.activation = 1
    layer_3.alpha = 1
    layer_3.beta = 1
    net.layers[0] = layer_1
    net.layers[1] = layer_2
    net.layers[2] = layer_3
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_reorg():
    '''test reorg layer'''
    net = LIB.make_network(2)
    layer_1 = LIB.make_convolutional_layer(1, 222, 222, 3, 32, 1, 3, 2, 0, 1, 0, 0, 0, 0)
    layer_2 = LIB.make_reorg_layer(1, 110, 110, 32, 2, 0, 0, 0)
    net.layers[0] = layer_1
    net.layers[1] = layer_2
    net.w = net.h = 222
    LIB.resize_network(net, 222, 222)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_region():
    '''test region layer'''
    net = LIB.make_network(2)
    layer_1 = LIB.make_convolutional_layer(1, 19, 19, 3, 425, 1, 1, 1, 0, 1, 0, 0, 0, 0)
    layer_2 = LIB.make_region_layer(1, 19, 19, 5, 80, 4)
    layer_2.softmax = 1
    net.layers[0] = layer_1
    net.layers[1] = layer_2
    net.w = net.h = 19
    LIB.resize_network(net, 19, 19)
    build_dtype = {}
    verify_darknet_frontend(net, build_dtype)
    LIB.free_network(net)

def test_forward_yolo_op():
    '''test yolo layer'''
    net = LIB.make_network(2)
    layer_1 = LIB.make_convolutional_layer(1, 224, 224, 3, 14, 1, 3, 2, 0, 1, 0, 0, 0, 0)
    layer_2 = LIB.make_yolo_layer(1, 111, 111, 2, 9, __darknetffi__.NULL, 2)
    net.layers[0] = layer_1
    net.layers[1] = layer_2
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    build_dtype = {}
    verify_darknet_frontend(net, build_dtype)
    LIB.free_network(net)

def test_forward_upsample():
    '''test upsample layer'''
    net = LIB.make_network(1)
    layer = LIB.make_upsample_layer(1, 19, 19, 3, 3)
    layer.scale = 1
    net.layers[0] = layer
    net.w = net.h = 19
    LIB.resize_network(net, 19, 19)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_l2normalize():
    '''test l2 normalization layer'''
    net = LIB.make_network(1)
    layer = LIB.make_l2norm_layer(1, 224*224*3)
    layer.c = layer.out_c = 3
    layer.h = layer.out_h = 224
    layer.w = layer.out_w = 224
    net.layers[0] = layer
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_elu():
    '''test elu activation layer'''
    net = LIB.make_network(1)
    layer_1 = LIB.make_convolutional_layer(1, 224, 224, 3, 32, 1, 3, 2, 0, 1, 0, 0, 0, 0)
    layer_1.activation = 8
    net.layers[0] = layer_1
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_softmax():
    '''test softmax layer'''
    net = LIB.make_network(1)
    layer_1 = LIB.make_softmax_layer(1, 75, 1)
    layer_1.temperature = 1
    net.layers[0] = layer_1
    net.w = net.h = 5
    LIB.resize_network(net, net.w, net.h)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_softmax_temperature():
    '''test softmax layer'''
    net = LIB.make_network(1)
    layer_1 = LIB.make_softmax_layer(1, 75, 1)
    layer_1.temperature = 0.8
    net.layers[0] = layer_1
    net.w = net.h = 5
    LIB.resize_network(net, net.w, net.h)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_rnn():
    '''test RNN layer'''
    net = LIB.make_network(1)
    batch = 1
    inputs = 256
    outputs = 256
    steps = 1
    activation = 1
    batch_normalize = 0
    adam = 0
    layer_1 = LIB.make_rnn_layer(batch, inputs, outputs, steps, activation, batch_normalize, adam)
    net.layers[0] = layer_1
    net.inputs = inputs
    net.outputs = outputs
    net.w = net.h = 0
    LIB.resize_network(net, net.w, net.h)
    verify_rnn_forward(net)
    LIB.free_network(net)

def _test_forward_crnn():
    '''test CRNN layer'''
    net = LIB.make_network(1)
    batch = 1
    c = 3
    h = 224
    w = 224
    hidden_filters = c
    output_filters = c
    steps = 1
    activation = 0
    batch_normalize = 0
    inputs = 256
    outputs = 256
    layer_1 = LIB.make_crnn_layer(batch, h, w, c, hidden_filters, output_filters,
                                  steps, activation, batch_normalize)
    net.layers[0] = layer_1
    net.inputs = inputs
    net.outputs = output_filters * h * w
    net.w = w
    net.h = h
    LIB.resize_network(net, net.w, net.h)
    verify_darknet_frontend(net)
    LIB.free_network(net)

def test_forward_lstm():
    '''test LSTM layer'''
    net = LIB.make_network(1)
    batch = 1
    inputs = 256
    outputs = 256
    steps = 1
    batch_normalize = 0
    adam = 0
    layer_1 = LIB.make_lstm_layer(batch, inputs, outputs, steps, batch_normalize, adam)
    net.layers[0] = layer_1
    net.inputs = inputs
    net.outputs = outputs
    net.w = net.h = 0
    LIB.resize_network(net, net.w, net.h)
    verify_rnn_forward(net)
    LIB.free_network(net)

def test_forward_gru():
    '''test GRU layer'''
    net = LIB.make_network(1)
    batch = 1
    inputs = 256
    outputs = 256
    steps = 1
    batch_normalize = 0
    adam = 0
    layer_1 = LIB.make_gru_layer(batch, inputs, outputs, steps, batch_normalize, adam)
    net.layers[0] = layer_1
    net.inputs = inputs
    net.outputs = outputs
    net.w = net.h = 0
    LIB.resize_network(net, net.w, net.h)
    verify_rnn_forward(net)
    LIB.free_network(net)

def test_forward_activation_logistic():
    '''test logistic activation layer'''
    net = LIB.make_network(1)
    batch = 1
    h = 224
    w = 224
    c = 3
    n = 32
    groups = 1
    size = 3
    stride = 2
    padding = 0
    activation = 0
    batch_normalize = 0
    binary = 0
    xnor = 0
    adam = 0
    layer_1 = LIB.make_convolutional_layer(batch, h, w, c, n, groups, size, stride, padding,
                                           activation, batch_normalize, binary, xnor, adam)
    net.layers[0] = layer_1
    net.w = w
    net.h = h
    LIB.resize_network(net, net.w, net.h)
    verify_darknet_frontend(net)
    LIB.free_network(net)

if __name__ == '__main__':
    test_forward_resnet50()
    test_forward_alexnet()
    test_forward_extraction()
    test_forward_yolov2()
    test_forward_yolov3()
    test_forward_convolutional()
    test_forward_maxpooling()
    test_forward_avgpooling()
    test_forward_batch_norm()
    test_forward_shortcut()
    test_forward_dense()
    test_forward_dense_batchnorm()
    test_forward_softmax()
    test_forward_softmax_temperature()
    test_forward_rnn()
    test_forward_reorg()
    test_forward_region()
    test_forward_yolo_op()
    test_forward_upsample()
    test_forward_l2normalize()
    test_forward_elu()
    test_forward_rnn()
# FIXME: Skip CRNN test since it causes segfault in libdarknet2.0.so
#    _test_forward_crnn()
    test_forward_lstm()
    test_forward_gru()
    test_forward_activation_logistic()
