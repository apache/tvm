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
# pylint: disable=invalid-name, missing-docstring
"""Unit tests for various models and operators"""
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import tvm.testing

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

from tensorflow import keras as tf_keras

# prevent Keras from using up all gpu memory
import keras
if tf.executing_eagerly():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session(config=config))


def pytest_generate_tests(metafunc):
    # This function generates the list of tests for pytest, based
    # on scenarios that will change the parameters in which the
    # tests use to run.
    # https://docs.pytest.org/en/latest/example/parametrize.html
    idlist = []
    argvalues = []
    for scenario in metafunc.cls.scenarios:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append([x[1] for x in items])
    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


# Scenarios:
# - classic keras, using keras from "import keras"
# - tensorflow keras, using keras from "from tensorflow import keras as tf_keras"
using_classic_keras = ("keras", {"keras": keras})
using_tensorflow_keras = ("tf_keras", {"keras": tf_keras})


def verify_keras_frontend(keras_model, need_transpose=True, layout="NCHW"):
    # Keras frontend currently supports tensorflow backend only.
    assert keras.backend.backend() == "tensorflow"

    if layout != "NCHW":
        need_transpose = False

    in_shapes = []
    for layer in keras_model._input_layers:
        if tf.executing_eagerly():
            in_shapes.append(tuple(dim if dim is not None else 1 for dim in layer.input.shape))
        else:
            in_shapes.append(
                tuple(dim.value if dim.value is not None else 1 for dim in layer.input.shape)
            )

    def get_keras_output(xs):
        return keras_model.predict(xs)

    def get_tvm_output(xs, target, dev, dtype="float32"):
        shape_dict = {name: x.shape for (name, x) in zip(keras_model.input_names, xs)}
        mod, params = relay.frontend.from_keras(keras_model, shape_dict, layout=layout)
        with tvm.transform.PassContext(opt_level=2):
            lib = relay.build(mod, target, params=params)
        m = graph_executor.GraphModule(lib["default"](dev))
        for name, x in zip(keras_model.input_names, xs):
            m.set_input(name, tvm.nd.array(x.astype(dtype)))
        m.run()
        return [m.get_output(i).numpy() for i in range(m.get_num_outputs())]

    def to_channels_first(arr):
        return arr.transpose([0, -1] + list(range(1, arr.ndim - 1)))

    def to_channels_last(arr):
        return arr.transpose([0] + list(range(2, arr.ndim)) + [1])

    xs = [np.random.uniform(size=shape, low=-1.0, high=1.0) for shape in in_shapes]
    keras_out = get_keras_output(xs)
    keras_out = keras_out if isinstance(keras_out, list) else [keras_out]
    for target, dev in tvm.testing.enabled_targets():
        inputs = [to_channels_first(x) for x in xs] if need_transpose else xs
        tvm_out = get_tvm_output(inputs, target, dev)
        for kout, tout in zip(keras_out, tvm_out):
            if need_transpose:
                tout = to_channels_last(tout)
            tvm.testing.assert_allclose(kout, tout, rtol=1e-5, atol=1e-5)


def get_mobilenet(keras_):
    if hasattr(keras.applications, "MobileNet"):
        # Keras 2.4.x and older
        MobileNet = keras_.applications.MobileNet
    else:
        # Keras 2.6.x and newer
        MobileNet = keras_.applications.mobilenet.MobileNet

    return MobileNet


@tvm.testing.uses_gpu
class TestKeras:
    '''Keras test'''
    scenarios = [using_classic_keras, using_tensorflow_keras]

    def test_forward_merge(self, keras_):
        data = keras_.layers.Input(shape=(32, 32, 3))
        x = keras_.layers.Conv2D(8, (3, 3), padding="same")(data)
        y = keras_.layers.Conv2D(8, (3, 3), padding="same")(x)
        z = keras_.layers.Conv2D(8, (3, 3), padding="same")(y)
        merge_funcs = [
            keras_.layers.Add(),
            keras_.layers.Subtract(),
            keras_.layers.Multiply(),
            keras_.layers.Maximum(),
            keras_.layers.Minimum(),
            keras_.layers.Average(),
            keras_.layers.Concatenate(),
        ]
        for merge_func in merge_funcs:
            class_name = type(merge_func).__name__
            if class_name in ("Subtract", "Dot"):
                out = merge_func([x, y])
            else:
                out = merge_func([x, y, z])
            keras_model = keras_.models.Model(data, out)
            verify_keras_frontend(keras_model)

    def test_forward_merge_dot(self, keras_):
        data1 = keras_.layers.Input(shape=(2, 2))
        data2 = keras_.layers.Input(shape=(2, 2))
        merge_funcs = [
            keras_.layers.Dot(axes=[1, 2]),
            keras_.layers.Dot(axes=[2, 1]),
            keras_.layers.Dot(axes=[1, 1]),
            keras_.layers.Dot(axes=[2, 2]),
            keras_.layers.Dot(axes=1),
            keras_.layers.Dot(axes=2),
        ]
        for merge_func in merge_funcs:
            out = merge_func([data1, data2])
            keras_model = keras.models.Model([data1, data2], out)
            verify_keras_frontend(keras_model)

    def test_forward_activations(self, keras_):
        data = keras_.layers.Input(shape=(32, 32, 3))
        act_funcs = [
            keras_.layers.Activation("softmax"),
            keras_.layers.Softmax(),
            keras_.layers.Softmax(axis=-1),
            keras_.layers.Softmax(axis=1),
            keras_.layers.Softmax(axis=2),
            keras_.layers.Softmax(axis=3),
            keras_.layers.Activation("softplus"),
            keras_.layers.Activation("relu"),
            keras_.layers.Activation("softsign"),
            keras_.layers.Activation("hard_sigmoid"),
            keras_.layers.Activation("sigmoid"),
            keras_.layers.Activation("tanh"),
            keras_.layers.Activation("linear"),
            keras_.layers.Activation("selu"),
            keras_.layers.ReLU(),
            keras_.layers.ReLU(max_value=6.0),
            keras_.layers.ReLU(max_value=6.0, threshold=0.0),
            keras_.layers.ReLU(max_value=6.0, threshold=1.0),
            keras_.layers.ReLU(max_value=6.0, threshold=1.0, negative_slope=0.0),
            keras_.layers.ReLU(max_value=6.0, threshold=1.0, negative_slope=0.5),
            keras_.layers.ReLU(max_value=6.0, threshold=1.0, negative_slope=1.0),
            keras_.layers.LeakyReLU(alpha=0.3),
            keras_.layers.PReLU(weights=np.random.rand(1, 32, 32, 3)),
            keras_.layers.ELU(alpha=0.5),
            keras_.layers.ThresholdedReLU(theta=0.5),
        ]
        for act_func in act_funcs:
            x = act_func(data)
            keras_model = keras_.models.Model(data, x)
            verify_keras_frontend(keras_model)
            verify_keras_frontend(keras_model, need_transpose=False, layout="NHWC")

    def test_forward_dense(self, keras_):
        data = keras_.layers.Input(shape=(32, 32, 1))
        x = keras_.layers.Flatten()(data)
        x = keras_.layers.Dropout(0.5)(x)
        x = keras_.layers.Dense(10, activation="relu", kernel_initializer="uniform")(x)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model)
        # RNN dense
        data = keras_.layers.Input(shape=(1, 32))
        x = keras_.layers.Dense(32, activation="relu", kernel_initializer="uniform")(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

    def test_forward_permute(self, keras_):
        data = keras_.layers.Input(shape=(2, 3, 4))
        x = keras_.layers.Permute([2, 3, 1])(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

    def test_forward_sequential(self, keras_):
        keras_model = keras_.models.Sequential(
            [
                keras_.layers.Dense(16, input_dim=32, activation="relu"),
                keras_.layers.Dropout(0.5),
                keras_.layers.Dense(8, activation="relu"),
                keras_.layers.Dropout(0.5),
                keras_.layers.Dense(1, activation="sigmoid"),
            ]
        )
        verify_keras_frontend(keras_model)

    def test_forward_pool(self, keras_):
        data = keras_.layers.Input(shape=(32, 32, 1))
        # maxpool
        x = keras_.layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model)
        # avgpool
        y = keras_.layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(data)
        keras_model = keras_.models.Model(data, y)
        verify_keras_frontend(keras_model)

    def test_forward_conv1d(self, keras_):
        data = keras_.layers.Input(shape=(32, 3))
        conv_funcs = [
            keras_.layers.Conv1D(filters=10, kernel_size=(3,), strides=(2,), padding="same"),
            keras_.layers.Conv1D(filters=10, kernel_size=(3,), dilation_rate=(2,), padding="same"),
            keras_.layers.Conv1D(filters=1, kernel_size=(3,), padding="valid", use_bias=False),
            keras_.layers.Conv1D(filters=10, kernel_size=(2,), padding="valid"),
            # Enable when relay conv1dtranspose handles NWC
            # keras_.layers.Conv1DTranspose(filters=10, kernel_size=(3), padding="valid"),
        ]
        for conv_func in conv_funcs:
            x = conv_func(data)
            keras_model = keras_.models.Model(data, x)
            verify_keras_frontend(keras_model, layout="NWC")

    def test_forward_conv(self, keras_):
        data = keras_.layers.Input(shape=(32, 32, 3))
        conv_funcs = [
            keras_.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(2, 2), padding="same"),
            keras_.layers.Conv2D(
                filters=10, kernel_size=(3, 3), dilation_rate=(2, 2), padding="same"
            ),
            keras_.layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same"),
            keras_.layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same"),
            keras_.layers.Conv2DTranspose(filters=10, kernel_size=(3, 3), padding="valid"),
            keras_.layers.SeparableConv2D(filters=10, kernel_size=(3, 3), padding="same"),
        ]
        for conv_func in conv_funcs:
            x = conv_func(data)
            keras_model = keras_.models.Model(data, x)
            verify_keras_frontend(keras_model)

    def test_forward_batch_norm(self, keras_):
        data = keras_.layers.Input(shape=(32, 32, 3))
        batch_norm_funcs = [
            keras_.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=False,
                beta_initializer="zeros",
                gamma_initializer="ones",
                moving_mean_initializer="zeros",
                moving_variance_initializer="ones",
            ),
            keras_.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
                beta_initializer="zeros",
                gamma_initializer="ones",
                moving_mean_initializer="zeros",
                moving_variance_initializer="ones",
            ),
            keras_.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=False,
                scale=True,
                beta_initializer="zeros",
                gamma_initializer="ones",
                moving_mean_initializer="zeros",
                moving_variance_initializer="ones",
            ),
            keras_.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=False,
                scale=False,
                beta_initializer="zeros",
                gamma_initializer="ones",
                moving_mean_initializer="zeros",
                moving_variance_initializer="ones",
            ),
        ]
        for batch_norm_func in batch_norm_funcs:
            x = batch_norm_func(data)
            keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model)

    def test_forward_upsample(self, keras_, interpolation="nearest"):
        data = keras_.layers.Input(shape=(32, 32, 3))
        x = keras_.layers.UpSampling2D(size=(3, 3), interpolation=interpolation)(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model)

    def test_forward_reshape(self, keras_):
        # input_shape len is 3, target_shape len is 3
        data = keras_.layers.Input(shape=(32, 32, 3))
        x = keras_.layers.Reshape(target_shape=(16, 64, 3))(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model)
        # input_shape len is 3, target_shape len is 2
        data = keras_.layers.Input(shape=(32, 8, 3))
        x = keras_.layers.Reshape(target_shape=(256, 3))(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model)
        # input_shape len is 2, target_shape len is 3
        data = keras_.layers.Input(shape=(256, 3))
        x = keras_.layers.Reshape(target_shape=(8, 32, 3))(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model)
        # input_shape len is 2, target_shape len is 1
        data = keras_.layers.Input(shape=(2, 8))
        x = keras_.layers.Reshape(target_shape=(16,))(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)
        # input_shape len is 1, target_shape len is 2
        data = keras_.layers.Input(shape=(16,))
        x = keras_.layers.Reshape(target_shape=(4, 4))(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)
        # input_shape len is 2, target_shape len is 2
        data = keras_.layers.Input(shape=(2, 8))
        x = keras_.layers.Reshape(target_shape=(4, 4))(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)
        # "non-square" target shape
        data = keras_.layers.Input(shape=(15,))
        x = keras_.layers.Reshape(target_shape=(5, 3))(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)
        # modify channel dim
        data = keras_.layers.Input(shape=(3, 2, 4))
        x = keras_.layers.Reshape(target_shape=(3, 8))(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model)

    def test_forward_crop(self, keras_):
        data = keras_.layers.Input(shape=(32, 32, 3))
        x = keras_.layers.Cropping2D(cropping=((1, 1), (1, 1)))(data)
        x = keras_.layers.Cropping2D(cropping=(1, 1))(x)
        x = keras_.layers.Cropping2D(cropping=1)(x)
        x = keras_.layers.Cropping2D(cropping=((0, 1), (1, 0)))(x)
        x = keras_.layers.Cropping2D(cropping=(1, 0))(x)
        x = keras_.layers.Cropping2D(cropping=0)(x)
        x = keras_.layers.Add()([x, x])
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model)

    def test_forward_multi_inputs(self, keras_):
        data1 = keras_.layers.Input(shape=(32, 32, 3))
        data2 = keras_.layers.Input(shape=(32, 32, 3))
        x = keras_.layers.Conv2D(8, (3, 3), padding="same")(data1)
        y = keras_.layers.Conv2D(8, (3, 3), padding="same")(data2)
        z = keras_.layers.Average()([x, y])
        z = keras_.layers.GlobalAveragePooling2D()(z)
        keras_model = keras_.models.Model([data1, data2], z)
        verify_keras_frontend(keras_model)

    def test_forward_multi_outputs(self, keras_):
        data = keras_.layers.Input(shape=(32, 32, 3))
        x = keras_.layers.Conv2D(8, (3, 3), padding="same")(data)
        x = keras_.layers.GlobalAveragePooling2D()(x)
        y = keras_.layers.Conv2D(8, (3, 3), padding="same")(data)
        y = keras_.layers.GlobalAveragePooling2D()(y)
        keras_model = keras_.models.Model(data, [x, y])
        verify_keras_frontend(keras_model)

    def test_forward_reuse_layers(self, keras_):
        # reuse conv2d
        data = keras_.layers.Input(shape=(32, 32, 3))
        conv2d = keras_.layers.Conv2D(8, (3, 3), padding="same")
        x = conv2d(data)
        y = conv2d(data)
        z = keras_.layers.Add()([x, y])
        z = keras_.layers.GlobalAveragePooling2D()(z)
        keras_model = keras_.models.Model(data, z)
        verify_keras_frontend(keras_model)
        # reuse add
        data = keras_.layers.Input(shape=(32, 32, 3))
        x = keras_.layers.Conv2D(8, (3, 3), padding="same")(data)
        add = keras_.layers.Add()
        x = add([x, x])
        x = add([x, x])
        z = keras_.layers.GlobalAveragePooling2D()(x)
        keras_model = keras_.models.Model(data, z)
        verify_keras_frontend(keras_model)

    def test_forward_lstm(self, keras_):
        data = keras_.layers.Input(shape=(10, 32))
        rnn_funcs = [
            keras_.layers.LSTM(16),
            keras_.layers.LSTM(16, return_sequences=True),
            keras_.layers.LSTM(16, return_sequences=True, use_bias=False),
        ]
        for rnn_func in rnn_funcs:
            x = rnn_func(data)
            keras_model = keras_.models.Model(data, x)
            verify_keras_frontend(keras_model, need_transpose=False)

    def test_forward_rnn(self, keras_):
        data = keras_.layers.Input(shape=(1, 32))
        rnn_funcs = [
            keras_.layers.LSTM(
                units=16, return_state=False, recurrent_activation="sigmoid", activation="tanh"
            ),
            keras_.layers.LSTM(
                units=16,
                return_state=False,
                recurrent_activation="sigmoid",
                activation="tanh",
                use_bias=False,
            ),
            keras_.layers.SimpleRNN(units=16, return_state=False, activation="tanh"),
            keras_.layers.SimpleRNN(
                units=16, return_state=False, activation="tanh", use_bias=False
            ),
            keras_.layers.GRU(
                units=16,
                return_state=False,
                recurrent_activation="sigmoid",
                activation="tanh",
                reset_after=False,
            ),
            keras_.layers.GRU(
                units=16,
                return_state=False,
                recurrent_activation="sigmoid",
                activation="tanh",
                reset_after=False,
                use_bias=False,
            ),
        ]
        for rnn_func in rnn_funcs:
            x = rnn_func(data)
            keras_model = keras_.models.Model(data, x)
            verify_keras_frontend(keras_model, need_transpose=False)

    def test_forward_vgg16(self, keras_, layout="NCHW"):
        if hasattr(keras_.applications, "VGG16"):
            # Keras 2.4.x and older
            VGG16 = keras_.applications.VGG16
        else:
            # Keras 2.6.x and newer
            VGG16 = keras_.applications.vgg16.VGG16

        keras_model = VGG16(
            include_top=True, weights="imagenet", input_shape=(224, 224, 3), classes=1000
        )
        verify_keras_frontend(keras_model, layout=layout)

    def test_forward_xception(self, keras_, layout="NCHW"):
        if hasattr(keras_.applications, "Xception"):
            # Keras 2.4.x and older
            Xception = keras_.applications.Xception
        else:
            # Keras 2.6.x and newer
            Xception = keras_.applications.xception.Xception

        keras_model = Xception(
            include_top=True, weights="imagenet", input_shape=(299, 299, 3), classes=1000
        )
        verify_keras_frontend(keras_model, layout=layout)

    def test_forward_resnet50(self, keras_, layout="NCHW"):
        if hasattr(keras_.applications, "ResNet50"):
            # Keras 2.4.x and older
            ResNet50 = keras_.applications.ResNet50
        else:
            # Keras 2.6.x and newer
            ResNet50 = keras_.applications.resnet.ResNet50

        keras_model = ResNet50(
            include_top=True, weights="imagenet", input_shape=(224, 224, 3), classes=1000
        )
        verify_keras_frontend(keras_model, layout=layout)

    def test_forward_mobilenet(self, keras_, layout="NCHW"):
        MobileNet = get_mobilenet(keras_)

        keras_model = MobileNet(
            include_top=True, weights="imagenet", input_shape=(224, 224, 3), classes=1000
        )
        verify_keras_frontend(keras_model, layout=layout)

    def test_forward_conv3d(self, keras_):
        data = keras_.layers.Input(shape=(32, 32, 32, 3))
        conv_funcs = [
            keras_.layers.Conv3D(
                filters=10, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same"
            ),
            keras_.layers.Conv3D(
                filters=10, kernel_size=(3, 3, 3), dilation_rate=(2, 2, 2), padding="same"
            ),
            keras_.layers.Conv3D(filters=1, kernel_size=(3, 3, 3), padding="valid", use_bias=False),
            keras_.layers.Conv3D(filters=10, kernel_size=(2, 2, 2), padding="valid"),
        ]
        for conv_func in conv_funcs:
            x = conv_func(data)
            keras_model = keras_.models.Model(data, x)
            verify_keras_frontend(keras_model, layout="NDHWC")

    def test_forward_conv3d_transpose(self, keras_):
        data = keras_.layers.Input(shape=(32, 32, 32, 3))
        conv_funcs = [
            keras_.layers.Conv3DTranspose(
                filters=10, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same"
            ),
            keras_.layers.Conv3DTranspose(
                filters=10, kernel_size=(1, 1, 1), dilation_rate=(1, 1, 1), padding="same"
            ),
            keras_.layers.Conv3DTranspose(
                filters=1, kernel_size=(3, 3, 3), padding="valid", use_bias=False
            ),
            keras_.layers.Conv3DTranspose(filters=10, kernel_size=(2, 2, 2), padding="valid"),
        ]
        for conv_func in conv_funcs:
            x = conv_func(data)
            keras_model = keras_.models.Model(data, x)
            verify_keras_frontend(keras_model, layout="NDHWC")

    def test_forward_pool3d(self, keras_):
        data = keras_.layers.Input(shape=(32, 32, 32, 1))
        pool_funcs = [  # maxpool
            keras_.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding="same"),
            keras_.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="valid"),
            # avgpool
            keras_.layers.AveragePooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same"),
            keras_.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding="valid"),
        ]
        for pool_func in pool_funcs:
            x = pool_func(data)
            keras_model = keras_.models.Model(data, x)
            verify_keras_frontend(keras_model, layout="NDHWC")

    def test_forward_upsample3d(self, keras_):
        data = keras_.layers.Input(shape=(32, 32, 32, 3))
        x = keras_.layers.UpSampling3D(size=(2, 3, 4))(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model, layout="NDHWC")

    def test_forward_zero_padding3d(self, keras_):
        data = keras_.layers.Input(shape=(32, 32, 32, 3))
        pad_funcs = [  # Integer
            keras_.layers.ZeroPadding3D(padding=2),
            # tuple of 3 ints
            keras_.layers.ZeroPadding3D(padding=(1, 2, 3)),
            # tuple of 3 tuples of 2 ints
            keras_.layers.ZeroPadding3D(padding=((1, 1), (2, 2), (2, 2))),
            # tuple of 3 tuples of 2 ints different values
            keras_.layers.ZeroPadding3D(padding=((1, 2), (2, 3), (3, 2))),
        ]
        for pad_func in pad_funcs:
            x = pad_func(data)
            keras_model = keras_.models.Model(data, x)
            verify_keras_frontend(keras_model, layout="NDHWC")

    def test_forward_embedding(self, keras_):
        data = keras_.layers.Input(shape=(2, 4), dtype="int32")
        x = keras_.layers.Embedding(10, 3)(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

        data = keras_.layers.Input(shape=(2, 3, 4), dtype="int32")
        x = keras_.layers.Embedding(4, 5)(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

        data = keras_.layers.Input(shape=(6, 2, 3, 4), dtype="int32")
        x = keras_.layers.Embedding(4, 5)(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

    def test_forward_repeat_vector(self, keras_):
        data = keras_.layers.Input(shape=(5,), dtype="float32")
        x = keras_.layers.Dense(6)(data)
        x = keras_.layers.RepeatVector(2)(x)

        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

        data = keras_.layers.Input(shape=(10,), dtype="float32")
        x = keras_.layers.RepeatVector(3)(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

        data = keras_.layers.Input(shape=(4,), dtype="float32")
        x = keras_.layers.RepeatVector(1)(data)
        keras_model = keras_.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

    def test_forward_global_pool3d(self, keras_):
        data = keras_.layers.Input(shape=(32, 32, 32, 1))
        pool_funcs = [  # global maxpool
            keras_.layers.GlobalMaxPooling3D(),
            # global avgpool
            keras_.layers.GlobalAveragePooling3D(),
        ]
        for pool_func in pool_funcs:
            x = pool_func(data)
            keras_model = keras_.models.Model(data, x)
            verify_keras_frontend(keras_model, layout="NDHWC")

    def test_forward_nested_layers(self, keras_):
        MobileNet = get_mobilenet(keras_)

        sub_model = MobileNet(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        keras_model = keras_.Sequential(
            [
                sub_model,
                keras_.layers.GlobalAveragePooling2D(),
                keras_.layers.Dense(1024, activation="relu"),
                keras_.layers.Dense(2, activation="sigmoid"),
            ]
        )
        verify_keras_frontend(keras_model)

    def test_forward_l2_normalize(self, keras_):
        data = keras_.layers.Input(shape=(16, 12, 8))
        K = keras_.backend
        l2_funcs = [
            keras_.layers.Lambda(lambda v: K.l2_normalize(v, axis=-2)),
            keras_.layers.Lambda(lambda v: K.l2_normalize(x=v, axis=-1)),
            keras_.layers.Lambda(lambda v: K.l2_normalize(axis=1, x=v)),
            keras_.layers.Lambda(lambda v: K.l2_normalize(v, 2)),
            keras_.layers.Lambda(lambda v: K.l2_normalize(v, axis=3)),
            keras_.layers.Lambda(lambda v: K.l2_normalize(v, axis=(2, 3))),
            keras_.layers.Lambda(lambda v: K.l2_normalize(v, (1, 2))),
            keras_.layers.Lambda(lambda v: K.l2_normalize(v, axis=[-2, -1])),
            keras_.layers.Lambda(lambda v: K.l2_normalize(v, [-3, -2])),
        ]
        for l2_func in l2_funcs:
            x = l2_func(data)
            keras_model = keras_.models.Model(data, x)
            verify_keras_frontend(keras_model, layout="NCHW")
            verify_keras_frontend(keras_model, layout="NHWC")

    def test_forward_time_distributed(self, keras_):
        conv2d_inputs = keras_.Input(shape=(10, 128, 128, 3))
        conv_2d_layer = keras_.layers.Conv2D(64, (3, 3))
        conv2d_model = keras_.models.Model(
            conv2d_inputs, keras_.layers.TimeDistributed(conv_2d_layer)(conv2d_inputs)
        )
        verify_keras_frontend(conv2d_model, layout="NDHWC")

        dense_inputs = keras_.Input(shape=(5, 1))
        dense_layer = keras_.layers.Dense(1)
        dense_model = keras_.models.Model(
            dense_inputs, keras_.layers.TimeDistributed(dense_layer)(dense_inputs)
        )
        verify_keras_frontend(dense_model, need_transpose=False)


if __name__ == "__main__":
    for k in [keras, tf_keras]:
        sut = TestKeras()
        sut.test_forward_merge_dot(keras_=k)
        sut.test_forward_merge(keras_=k)
        sut.test_forward_activations(keras_=k)
        sut.test_forward_dense(keras_=k)
        sut.test_forward_permute(keras_=k)
        sut.test_forward_sequential(keras_=k)
        sut.test_forward_pool(keras_=k)
        sut.test_forward_conv(keras_=k)
        sut.test_forward_conv1d(keras_=k)
        sut.test_forward_batch_norm(keras_=k)
        sut.test_forward_upsample(keras_=k, interpolation="nearest")
        sut.test_forward_upsample(keras_=k, interpolation="bilinear")
        sut.test_forward_reshape(keras_=k)
        sut.test_forward_crop(keras_=k)
        sut.test_forward_multi_inputs(keras_=k)
        sut.test_forward_multi_outputs(keras_=k)
        sut.test_forward_reuse_layers(keras_=k)
        sut.test_forward_lstm(keras_=k)
        sut.test_forward_rnn(keras_=k)
        sut.test_forward_vgg16(keras_=k)
        sut.test_forward_vgg16(keras_=k, layout="NHWC")
        sut.test_forward_xception(keras_=k)
        sut.test_forward_resnet50(keras_=k)
        sut.test_forward_resnet50(keras_=k, layout="NHWC")
        sut.test_forward_mobilenet(keras_=k)
        sut.test_forward_mobilenet(keras_=k, layout="NHWC")
        sut.test_forward_conv3d(keras_=k)
        sut.test_forward_conv3d_transpose(keras_=k)
        sut.test_forward_pool3d(keras_=k)
        sut.test_forward_global_pool3d(keras_=k)
        sut.test_forward_upsample3d(keras_=k)
        sut.test_forward_zero_padding3d(keras_=k)
        sut.test_forward_embedding(keras_=k)
        sut.test_forward_repeat_vector(keras_=k)
        sut.test_forward_l2_normalize(keras_=k)
        sut.test_forward_time_distributed(keras_=k)
