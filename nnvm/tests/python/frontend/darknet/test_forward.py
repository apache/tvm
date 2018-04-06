"""
Compile Darknet Models
=====================
This article is a test script to test darknet models with NNVM.
All the required models and libraries will be downloaded from the internet
by the script.
"""
import os
import requests
import numpy as np
from nnvm import frontend
from nnvm.testing.darknet import __darknetffi__
import nnvm.compiler
import tvm
import sys
import urllib
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

DARKNET_LIB = 'libdarknet.so'
DARKNETLIB_URL = 'https://github.com/siju-samuel/darknet/blob/master/lib/' \
                                    + DARKNET_LIB + '?raw=true'
_download(DARKNETLIB_URL, DARKNET_LIB)
LIB = __darknetffi__.dlopen('./' + DARKNET_LIB)

def test_forward(net):
    '''Test network with given input image on both darknet and tvm'''
    def get_darknet_output(net, img):
        return LIB.network_predict_image(net, img)

    def get_tvm_output(net, img):
        '''Compute TVM output'''
        dtype = 'float32'
        batch_size = 1
        sym, params = frontend.darknet.from_darknet(net, dtype)
        data = np.empty([batch_size, img.c, img.h, img.w], dtype)
        i = 0
        for c in range(img.c):
            for h in range(img.h):
                for k in range(img.w):
                    data[0][c][h][k] = img.data[i]
                    i = i + 1

        target = 'llvm'
        shape_dict = {'data': data.shape}
        #with nnvm.compiler.build_config(opt_level=2):
        graph, library, params = nnvm.compiler.build(sym, target, shape_dict, dtype, params=params)
        ######################################################################
        # Execute on TVM
        # ---------------
        # The process is no different from other examples.
        from tvm.contrib import graph_runtime
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

    test_image = 'dog.jpg'
    img_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + test_image   +'?raw=true'
    _download(img_url, test_image)
    img = LIB.letterbox_image(LIB.load_image_color(test_image.encode('utf-8'), 0, 0), net.w, net.h)
    darknet_output = get_darknet_output(net, img)
    darknet_out = np.zeros(net.outputs, dtype='float32')
    for i in range(net.outputs):
        darknet_out[i] = darknet_output[i]
    tvm_out = get_tvm_output(net, img)
    np.testing.assert_allclose(darknet_out, tvm_out, rtol=1e-3, atol=1e-3)

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
    model_name = 'yolo'
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
    test_forward_reorg()
    test_forward_region()
