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
import numpy as np
import nnvm
import tvm
from tvm.contrib import graph_runtime
from nnvm.testing.config import ctx_list
import keras

# prevent keras from using up all gpu memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


def verify_keras_frontend(keras_model, need_transpose=True):
    # Keras frontend currently supports tensorflow backend only.
    assert(keras.backend.backend() == 'tensorflow')

    in_shapes = []
    for layer in keras_model._input_layers:
        in_shapes.append(tuple(dim.value if dim.value is not None else 1 for dim in layer.input.shape))

    def get_keras_output(xs, dtype='float32'):
        return keras_model.predict(xs)

    def get_tvm_output(xs, target, ctx, dtype='float32'):
        sym, params = nnvm.frontend.from_keras(keras_model)
        shape_dict = {name: x.shape for (name, x) in zip(keras_model.input_names, xs)}
        with nnvm.compiler.build_config(opt_level=2):
            graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        for name, x in zip(keras_model.input_names, xs):
            m.set_input(name, tvm.nd.array(x.astype(dtype)))
        m.set_input(**params)
        m.run()

        return [m.get_output(i).asnumpy() for i in range(m.get_num_outputs())]

    def to_channels_first(arr):
        return arr.transpose([0, -1] + list(range(1, arr.ndim - 1)))

    def to_channels_last(arr):
        return arr.transpose([0] + list(range(2, arr.ndim)) + [1])

    xs = [np.random.uniform(size=shape, low=-1.0, high=1.0) for shape in in_shapes]
    keras_out = get_keras_output(xs)

    keras_out = keras_out if isinstance(keras_out, list) else [keras_out]
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output([to_channels_first(x) for x in xs] if need_transpose else xs, target, ctx)
        for kout, tout in zip(keras_out, tvm_out):
            if need_transpose:
                tout = to_channels_last(tout)
            tvm.testing.assert_allclose(kout, tout, rtol=1e-5, atol=1e-5)

def test_forward_elemwise_add():
    r = []
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.Conv2D(8, (3, 3), padding="same")(data)
    r.append(x)
    x = keras.layers.Conv2D(8, (3, 3), padding="same")(x)
    r.append(x)
    x = keras.layers.Conv2D(8, (3, 3), padding="same")(x)
    # add two symbols
    y = keras.layers.add([keras.layers.add([x, r[0]]), r[1]])
    y = keras.layers.GlobalAveragePooling2D()(y)
    keras_model = keras.models.Model(data, y)
    verify_keras_frontend(keras_model)
    # add three symbols
    y = keras.layers.add([x, r[0], r[1]])
    y = keras.layers.GlobalAveragePooling2D()(y)
    keras_model = keras.models.Model(data, y)
    verify_keras_frontend(keras_model)


def _test_forward_dense():
    data = keras.layers.Input(shape=(32,32,1))
    x = keras.layers.Flatten()(data)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(10, activation='relu', kernel_initializer='uniform')(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)

def _test_forward_dense_with_3d_inp():
    data = keras.layers.Input(shape=(1, 20))
    x = keras.layers.Dense(10, activation='relu', kernel_initializer='uniform')(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model, need_transpose=False)

def test_forward_dense():
    _test_forward_dense()
    _test_forward_dense_with_3d_inp()

def test_forward_pool():
    data = keras.layers.Input(shape=(32,32,1))
    # maxpool
    x = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)
    # avgpool
    y = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(data)
    keras_model = keras.models.Model(data, y)
    verify_keras_frontend(keras_model)


def test_forward_conv():
    data = keras.layers.Input(shape=(32,32,3))
    conv_funcs = [keras.layers.Conv2D(filters=10, kernel_size=(3,3),
                                      strides=(2,2), padding='same'),
                  keras.layers.Conv2D(filters=10, kernel_size=(3,3),
                                      dilation_rate=(2,2), padding='same'),
                  keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same'),
                  keras.layers.Conv2DTranspose(filters=10, kernel_size=(3,3), padding='valid'),
                  keras.layers.SeparableConv2D(filters=10, kernel_size=(3,3), padding='same')]
    for conv_func in conv_funcs:
        x = conv_func(data)
        keras_model = keras.models.Model(data, x)
        verify_keras_frontend(keras_model)


def test_forward_upsample():
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.UpSampling2D(size=(3,3))(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_reshape():
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.Reshape(target_shape=(32,32,3))(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_crop():
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.Cropping2D(cropping=((1, 1), (1, 1)))(data)
    x = keras.layers.Cropping2D(cropping=(1, 1))(x)
    x = keras.layers.Cropping2D(cropping=1)(x)
    x = keras.layers.Cropping2D(cropping=((0, 1), (1, 0)))(x)
    x = keras.layers.Cropping2D(cropping=(1, 0))(x)
    x = keras.layers.Cropping2D(cropping=0)(x)
    x = keras.layers.Add()([x, x])
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_vgg16():
    keras_model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet',
        input_shape=(224,224,3), classes=1000)
    verify_keras_frontend(keras_model)


def test_forward_xception():
    keras_model = keras.applications.xception.Xception(include_top=True, weights='imagenet',
        input_shape=(299,299,3), classes=1000)
    verify_keras_frontend(keras_model)


def test_forward_resnet50():
    keras_model = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet',
        input_shape=(224,224,3), classes=1000)
    verify_keras_frontend(keras_model)


def test_forward_mobilenet():
    keras_model = keras.applications.mobilenet.MobileNet(include_top=True, weights='imagenet',
        input_shape=(224,224,3), classes=1000)
    verify_keras_frontend(keras_model)


def test_forward_activations():
    data = keras.layers.Input(shape=(32,32,3))
    weights = np.random.rand(1, 32, 32, 3)
    act_funcs = [keras.layers.Activation('softmax'),
                 keras.layers.Activation('softplus'),
                 keras.layers.ReLU(),
                 keras.layers.ReLU(max_value=6.),
                 keras.layers.LeakyReLU(alpha=0.3),
                 keras.layers.PReLU(weights=weights, alpha_initializer="zero"),
                 keras.layers.ELU(alpha=0.5),
                 keras.layers.Activation('selu'),
                 keras.layers.ThresholdedReLU(theta=0.5),
                 keras.layers.Activation('softsign'),
                 keras.layers.Activation('hard_sigmoid'),
                 keras.layers.Activation('sigmoid'),
                 keras.layers.Activation('tanh'),
                 keras.layers.Activation('linear')]
    for act_func in act_funcs:
        x = act_func(data)
        keras_model = keras.models.Model(data, x)
        verify_keras_frontend(keras_model)


def test_forward_multi_inputs():
    data1 = keras.layers.Input(shape=(32,32,3))
    data2 = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.Conv2D(8, (3, 3), padding="same")(data1)
    y = keras.layers.Conv2D(8, (3, 3), padding="same")(data2)
    z = keras.layers.add([x, y])
    z = keras.layers.GlobalAveragePooling2D()(z)
    keras_model = keras.models.Model([data1, data2], z)
    verify_keras_frontend(keras_model)


def test_forward_multi_outputs():
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.Conv2D(8, (3, 3), padding="same")(data)
    x = keras.layers.GlobalAveragePooling2D()(x)
    y = keras.layers.Conv2D(8, (3, 3), padding="same")(data)
    y = keras.layers.GlobalAveragePooling2D()(y)
    keras_model = keras.models.Model(data, [x, y])
    verify_keras_frontend(keras_model)


def test_forward_reuse_layers():
    # reuse conv2d
    data = keras.layers.Input(shape=(32,32,3))
    conv2d = keras.layers.Conv2D(8, (3, 3), padding="same")
    x = conv2d(data)
    y = conv2d(data)
    z = keras.layers.add([x, y])
    z = keras.layers.GlobalAveragePooling2D()(z)
    keras_model = keras.models.Model(data, z)
    verify_keras_frontend(keras_model)

    # reuse add
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.Conv2D(8, (3, 3), padding="same")(data)
    add = keras.layers.Add()
    x = add([x, x])
    x = add([x, x])
    z = keras.layers.GlobalAveragePooling2D()(x)
    keras_model = keras.models.Model(data, z)
    verify_keras_frontend(keras_model)

def _test_LSTM(time_steps, inputs, hidden, return_state=True):
    data = keras.layers.Input(shape=(time_steps, inputs))
    lstm_out = keras.layers.LSTM(hidden,
                                 return_state=return_state,
                                 recurrent_activation='sigmoid',
                                 activation='tanh')
    x = lstm_out(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model, need_transpose=False)

def _test_LSTM_MultiLayer(inputs, hidden):
    inputs = keras.layers.Input(shape=(1, inputs))
    layer = keras.layers.LSTM(hidden, return_state=True, return_sequences=True,
                                 recurrent_activation='sigmoid',
                                 activation='tanh')
    outputs = layer(inputs)
    output, state = outputs[0], outputs[1:]
    output = keras.layers.LSTM(hidden, recurrent_activation='sigmoid',
                               activation='tanh')(output, initial_state=state)
    keras_model = keras.models.Model(inputs, output)
    verify_keras_frontend(keras_model, need_transpose=False)


def test_forward_LSTM():
    _test_LSTM(1, 8, 8, return_state=True)
    _test_LSTM(1, 4, 4, return_state=False)
    _test_LSTM(20, 16, 256, return_state=False)
    _test_LSTM_MultiLayer(4, 4)

def _test_RNN(inputs, units):
    data = keras.layers.Input(shape=(1, inputs))
    rnn_out = keras.layers.SimpleRNN(units, return_state=True,
                                 activation='tanh')
    x = rnn_out(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model, need_transpose=False)

def _test_RNN_MultiLayer(inputs, units):
    inputs = keras.layers.Input(shape=(1, inputs))
    layer = keras.layers.SimpleRNN(units, return_state=True, return_sequences=True,
                                   activation='tanh')
    outputs = layer(inputs)
    output, state = outputs[0], outputs[1:]
    output = keras.layers.SimpleRNN(units, activation='tanh')(output, initial_state=state)
    keras_model = keras.models.Model(inputs, output)
    verify_keras_frontend(keras_model, need_transpose=False)

def test_forward_RNN():
    _test_RNN(2, 4)
    _test_RNN(4, 3)
    _test_RNN_MultiLayer(4, 12)

def _test_GRU(inputs, units):
    data = keras.layers.Input(shape=(1, inputs))
    gru_out = keras.layers.GRU(units,
                               return_state=True,
                               recurrent_activation='sigmoid',
                               activation='tanh')
    x = gru_out(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model, need_transpose=False)

def _test_GRU_MultiLayer(inputs, units):
    inputs = keras.layers.Input(shape=(1, inputs))
    layer = keras.layers.GRU(units,
                             return_state=True,
                             return_sequences=True,
                             recurrent_activation='sigmoid',
                             activation='tanh')
    outputs = layer(inputs)
    output, state = outputs[0], outputs[1:]
    output = keras.layers.GRU(units, recurrent_activation='sigmoid',
                              activation='tanh')(output, initial_state=state)
    keras_model = keras.models.Model(inputs, output)
    verify_keras_frontend(keras_model, need_transpose=False)

def test_forward_GRU():
    _test_GRU(2, 4)
    _test_GRU(4, 3)
    _test_GRU_MultiLayer(4, 4)

if __name__ == '__main__':
    test_forward_elemwise_add()
    test_forward_activations()
    test_forward_dense()
    test_forward_pool()
    test_forward_conv()
    test_forward_upsample()
    test_forward_reshape()
    test_forward_crop()
    test_forward_vgg16()
    test_forward_xception()
    test_forward_resnet50()
    test_forward_mobilenet()

    test_forward_multi_inputs()
    test_forward_multi_outputs()
    test_forward_reuse_layers()
    test_forward_LSTM()
    test_forward_RNN()
    test_forward_GRU()
