"""
Compile Darknet Models
=====================
This article is a test script to test darknet models with NNVM.
All the required models and libraries will be downloaded from the internet
by the script.
"""
import os
import requests
import sys
import urllib
import numpy as np
import tvm
from tvm.contrib import graph_runtime
from nnvm import frontend
from nnvm.testing.darknet import __darknetffi__
import nnvm.compiler
if sys.version_info >= (3,):
    import urllib.request as urllib2
else:
    import urllib2


def _download(url, path, overwrite=False, sizecompare=False):
    ''' Download from internet'''
    if os.path.isfile(path) and not overwrite:
        if sizecompare:
            file_size = os.path.getsize(path)
            res_head = requests.head(url)
            res_get = requests.get(url, stream=True)
            if 'Content-Length' not in res_head.headers:
                res_get = urllib2.urlopen(url)
            urlfile_size = int(res_get.headers['Content-Length'])
            if urlfile_size != file_size:
                print("exist file got corrupted, downloading", path, " file freshly")
                _download(url, path, True, False)
                return
        print('File {} exists, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    try:
        urllib.request.urlretrieve(url, path)
        print('')
    except:
        urllib.urlretrieve(url, path)

DARKNET_LIB = 'libdarknet2.0.so'
DARKNETLIB_URL = 'https://github.com/siju-samuel/darknet/blob/master/lib/' \
                                    + DARKNET_LIB + '?raw=true'
_download(DARKNETLIB_URL, DARKNET_LIB)
LIB = __darknetffi__.dlopen('./' + DARKNET_LIB)

def _get_tvm_output(net, data):
    '''Compute TVM output'''
    dtype = 'float32'
    sym, params = frontend.darknet.from_darknet(net, dtype)

    target = 'llvm'
    shape_dict = {'data': data.shape}
    graph, library, params = nnvm.compiler.build(sym, target, shape_dict, dtype, params=params)
    # Execute on TVM
    ctx = tvm.cpu(0)
    m = graph_runtime.create(graph, library, ctx)
    # set inputs
    m.set_input('data', tvm.nd.array(data.astype(dtype)))
    m.set_input(**params)
    m.run()
    # get outputs
    out_shape = (net.outputs,)
    tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()
    return tvm_out

def test_forward(net):
    '''Test network with given input image on both darknet and tvm'''
    def get_darknet_output(net, img):
        return LIB.network_predict_image(net, img)
    dtype = 'float32'

    test_image = 'dog.jpg'
    img_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + test_image   +'?raw=true'
    _download(img_url, test_image)
    img = LIB.letterbox_image(LIB.load_image_color(test_image.encode('utf-8'), 0, 0), net.w, net.h)
    darknet_output = get_darknet_output(net, img)
    darknet_out = np.zeros(net.outputs, dtype='float32')
    for i in range(net.outputs):
        darknet_out[i] = darknet_output[i]
    batch_size = 1

    data = np.empty([batch_size, img.c, img.h, img.w], dtype)
    i = 0
    for c in range(img.c):
        for h in range(img.h):
            for k in range(img.w):
                data[0][c][h][k] = img.data[i]
                i = i + 1

    tvm_out = _get_tvm_output(net, data)
    np.testing.assert_allclose(darknet_out, tvm_out, rtol=1e-3, atol=1e-3)

def test_rnn_forward(net):
    '''Test network with given input data on both darknet and tvm'''
    def get_darknet_network_predict(net, data):
        return LIB.network_predict(net, data)
    from cffi import FFI
    ffi = FFI()
    np_arr = np.zeros([1, net.inputs], dtype='float32')
    np_arr[0, 84] = 1
    cffi_arr = ffi.cast('float*', np_arr.ctypes.data)
    tvm_out = _get_tvm_output(net, np_arr)
    darknet_output = get_darknet_network_predict(net, cffi_arr)
    darknet_out = np.zeros(net.outputs, dtype='float32')
    for i in range(net.outputs):
        darknet_out[i] = darknet_output[i]
    np.testing.assert_allclose(darknet_out, tvm_out, rtol=1e-4, atol=1e-4)

def test_forward_extraction():
    '''test extraction model'''
    model_name = 'extraction'
    cfg_name = model_name + '.cfg'
    weights_name = model_name + '.weights'
    cfg_url = 'https://github.com/pjreddie/darknet/blob/master/cfg/' + cfg_name + '?raw=true'
    weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'
    _download(cfg_url, cfg_name)
    _download(weights_url, weights_name)
    net = LIB.load_network(cfg_name.encode('utf-8'), weights_name.encode('utf-8'), 0)
    test_forward(net)
    LIB.free_network(net)

def test_forward_alexnet():
    '''test alexnet model'''
    model_name = 'alexnet'
    cfg_name = model_name + '.cfg'
    weights_name = model_name + '.weights'
    cfg_url = 'https://github.com/pjreddie/darknet/blob/master/cfg/' + cfg_name + '?raw=true'
    weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'
    _download(cfg_url, cfg_name)
    _download(weights_url, weights_name)
    net = LIB.load_network(cfg_name.encode('utf-8'), weights_name.encode('utf-8'), 0)
    test_forward(net)
    LIB.free_network(net)

def test_forward_resnet50():
    '''test resnet50 model'''
    model_name = 'resnet50'
    cfg_name = model_name + '.cfg'
    weights_name = model_name + '.weights'
    cfg_url = 'https://github.com/pjreddie/darknet/blob/master/cfg/' + cfg_name + '?raw=true'
    weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'
    _download(cfg_url, cfg_name)
    _download(weights_url, weights_name)
    net = LIB.load_network(cfg_name.encode('utf-8'), weights_name.encode('utf-8'), 0)
    test_forward(net)
    LIB.free_network(net)

def test_forward_yolo():
    '''test yolo model'''
    model_name = 'yolov2'
    cfg_name = model_name + '.cfg'
    weights_name = model_name + '.weights'
    cfg_url = 'https://github.com/pjreddie/darknet/blob/master/cfg/' + cfg_name + '?raw=true'
    weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'
    _download(cfg_url, cfg_name)
    _download(weights_url, weights_name)
    net = LIB.load_network(cfg_name.encode('utf-8'), weights_name.encode('utf-8'), 0)
    test_forward(net)
    LIB.free_network(net)

def test_forward_convolutional():
    '''test convolutional layer'''
    net = LIB.make_network(1)
    layer = LIB.make_convolutional_layer(1, 224, 224, 3, 32, 1, 3, 2, 0, 1, 0, 0, 0, 0)
    net.layers[0] = layer
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    test_forward(net)
    LIB.free_network(net)

def test_forward_dense():
    '''test fully connected layer'''
    net = LIB.make_network(1)
    layer = LIB.make_connected_layer(1, 75, 20, 1, 0, 0)
    net.layers[0] = layer
    net.w = net.h = 5
    LIB.resize_network(net, 5, 5)
    test_forward(net)
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
    test_forward(net)
    LIB.free_network(net)

def test_forward_maxpooling():
    '''test maxpooling layer'''
    net = LIB.make_network(1)
    layer = LIB.make_maxpool_layer(1, 224, 224, 3, 2, 2, 0)
    net.layers[0] = layer
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    test_forward(net)
    LIB.free_network(net)

def test_forward_avgpooling():
    '''test avgerage pooling layer'''
    net = LIB.make_network(1)
    layer = LIB.make_avgpool_layer(1, 224, 224, 3)
    net.layers[0] = layer
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    test_forward(net)
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
    test_forward(net)
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
    test_forward(net)
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
    test_forward(net)
    LIB.free_network(net)

def test_forward_region():
    '''test region layer'''
    net = LIB.make_network(2)
    layer_1 = LIB.make_convolutional_layer(1, 224, 224, 3, 8, 1, 3, 2, 0, 1, 0, 0, 0, 0)
    layer_2 = LIB.make_region_layer(1, 111, 111, 2, 2, 1)
    layer_2.softmax = 1
    net.layers[0] = layer_1
    net.layers[1] = layer_2
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    test_forward(net)
    LIB.free_network(net)

def test_forward_yolo_op():
    '''test yolo layer'''
    net = LIB.make_network(2)
    layer_1 = LIB.make_convolutional_layer(1, 224, 224, 3, 14, 1, 3, 2, 0, 1, 0, 0, 0, 0)
    a = []
    layer_2 = LIB.make_yolo_layer(1, 111, 111, 2, 0, a, 2)
    net.layers[0] = layer_1
    net.layers[1] = layer_2
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    test_forward(net)
    LIB.free_network(net)

def test_forward_upsample():
    '''test upsample layer'''
    net = LIB.make_network(1)
    layer = LIB.make_upsample_layer(1, 19, 19, 3, 3)
    layer.scale = 1
    net.layers[0] = layer
    net.w = net.h = 19
    LIB.resize_network(net, 19, 19)
    test_forward(net)
    LIB.free_network(net)

def test_forward_elu():
    '''test elu activation layer'''
    net = LIB.make_network(1)
    layer_1 = LIB.make_convolutional_layer(1, 224, 224, 3, 32, 1, 3, 2, 0, 1, 0, 0, 0, 0)
    layer_1.activation = 8
    net.layers[0] = layer_1
    net.w = net.h = 224
    LIB.resize_network(net, 224, 224)
    test_forward(net)
    LIB.free_network(net)

def test_forward_softmax():
    '''test softmax layer'''
    net = LIB.make_network(1)
    layer_1 = LIB.make_softmax_layer(1, 75, 1)
    layer_1.temperature=1
    net.layers[0] = layer_1
    net.w = net.h = 5
    LIB.resize_network(net, net.w, net.h)
    test_forward(net)
    LIB.free_network(net)

def test_forward_softmax_temperature():
    '''test softmax layer'''
    net = LIB.make_network(1)
    layer_1 = LIB.make_softmax_layer(1, 75, 1)
    layer_1.temperature=0.8
    net.layers[0] = layer_1
    net.w = net.h = 5
    LIB.resize_network(net, net.w, net.h)
    test_forward(net)
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
    test_rnn_forward(net)
    LIB.free_network(net)

def test_forward_crnn():
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
    test_forward(net)
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
    test_rnn_forward(net)
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
    test_rnn_forward(net)
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
    test_forward(net)
    LIB.free_network(net)

if __name__ == '__main__':
    test_forward_resnet50()
    test_forward_alexnet()
    test_forward_extraction()
    test_forward_yolo()
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
    test_forward_elu()
    test_forward_rnn()
    test_forward_crnn()
    test_forward_lstm()
    test_forward_gru()
    test_forward_activation_logistic()
