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
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay.testing.config import ctx_list
import keras

# prevent Keras from using up all gpu memory
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
        shape_dict = {name: x.shape for (name, x) in zip(keras_model.input_names, xs)}
        mod, params = relay.frontend.from_keras(keras_model, shape_dict)
        with relay.transform.build_config(opt_level=2):
            graph, lib, params = relay.build(mod,
                                             target,
                                             params=params)
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
        inputs = [to_channels_first(x) for x in xs] if need_transpose else xs
        tvm_out = get_tvm_output(inputs, target, ctx)
        for kout, tout in zip(keras_out, tvm_out):
            if need_transpose:
                tout = to_channels_last(tout)
            tvm.testing.assert_allclose(kout, tout, rtol=1e-5, atol=1e-5)


def test_forward_merge():
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.Conv2D(8, (3, 3), padding="same")(data)
    y = keras.layers.Conv2D(8, (3, 3), padding="same")(x)
    z = keras.layers.Conv2D(8, (3, 3), padding="same")(y)
    merge_funcs = [keras.layers.Add(),
                   keras.layers.Subtract(),
                   keras.layers.Multiply(),
                   keras.layers.Maximum(),
                   keras.layers.Average(),
                   keras.layers.Concatenate()]
    for merge_func in merge_funcs:
        if isinstance(merge_func, keras.layers.merge.Subtract):
            out = merge_func([x, y])
        else:
            out = merge_func([x, y, z])
        keras_model = keras.models.Model(data, out)
        verify_keras_frontend(keras_model)


def test_forward_activations():
    data = keras.layers.Input(shape=(32,32,3))
    act_funcs = [keras.layers.Activation('softmax'),
                 keras.layers.Activation('softplus'),
                 keras.layers.Activation('relu'),
                 keras.layers.Activation('softsign'),
                 keras.layers.Activation('hard_sigmoid'),
                 keras.layers.Activation('sigmoid'),
                 keras.layers.Activation('tanh'),
                 keras.layers.Activation('linear'),
                 keras.layers.Activation('selu'),
                 keras.layers.ReLU(),
                 keras.layers.ReLU(max_value=6.),
                 keras.layers.LeakyReLU(alpha=0.3),
                 keras.layers.PReLU(weights=np.random.rand(1, 32, 32, 3)),
                 keras.layers.ELU(alpha=0.5),
                 keras.layers.ThresholdedReLU(theta=0.5)]
    for act_func in act_funcs:
        x = act_func(data)
        keras_model = keras.models.Model(data, x)
        verify_keras_frontend(keras_model)


def test_forward_dense():
    data = keras.layers.Input(shape=(32,32,1))
    x = keras.layers.Flatten()(data)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(10, activation='relu', kernel_initializer='uniform')(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_sequential():
    keras_model = keras.models.Sequential([
        keras.layers.Dense(16, input_dim=32, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    verify_keras_frontend(keras_model)


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


def test_forward_upsample(interpolation='nearest'):
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.UpSampling2D(size=(3,3), interpolation=interpolation)(data)
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


def test_forward_multi_inputs():
    data1 = keras.layers.Input(shape=(32,32,3))
    data2 = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.Conv2D(8, (3, 3), padding="same")(data1)
    y = keras.layers.Conv2D(8, (3, 3), padding="same")(data2)
    z = keras.layers.Average()([x, y])
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
    z = keras.layers.Add()([x, y])
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


def test_forward_rnn():
    data = keras.layers.Input(shape=(1,32))
    rnn_funcs = [keras.layers.LSTM(units=16, return_state=False,
                    recurrent_activation='sigmoid', activation='tanh'),
                 keras.layers.SimpleRNN(units=16, return_state=False,
                    activation='tanh'),
                 keras.layers.GRU(units=16, return_state=False,
                    recurrent_activation='sigmoid', activation='tanh')]
    for rnn_func in rnn_funcs:
        x = rnn_func(data)
        keras_model = keras.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)


def test_forward_vgg16():
    keras_model = keras.applications.VGG16(include_top=True, weights='imagenet',
        input_shape=(224,224,3), classes=1000)
    verify_keras_frontend(keras_model)


def test_forward_xception():
    keras_model = keras.applications.Xception(include_top=True, weights='imagenet',
        input_shape=(299,299,3), classes=1000)
    verify_keras_frontend(keras_model)


def test_forward_resnet50():
    keras_model = keras.applications.ResNet50(include_top=True, weights='imagenet',
        input_shape=(224,224,3), classes=1000)
    verify_keras_frontend(keras_model)


def test_forward_mobilenet():
    keras_model = keras.applications.MobileNet(include_top=True, weights='imagenet',
        input_shape=(224,224,3), classes=1000)
    verify_keras_frontend(keras_model)


if __name__ == '__main__':
    test_forward_merge()
    test_forward_activations()
    test_forward_dense()
    test_forward_sequential()
    test_forward_pool()
    test_forward_conv()
    test_forward_upsample(interpolation='nearest')
    test_forward_upsample(interpolation='bilinear')
    test_forward_reshape()
    test_forward_crop()
    test_forward_multi_inputs()
    test_forward_multi_outputs()
    test_forward_reuse_layers()
    test_forward_rnn()
    test_forward_vgg16()
    test_forward_xception()
    test_forward_resnet50()
    test_forward_mobilenet()
