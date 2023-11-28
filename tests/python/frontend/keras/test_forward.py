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
"""Unit tests for various models and operators"""
from packaging import version as package_version
import numpy as np

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

from tensorflow import keras as tf_keras

# prevent Keras from using up all gpu memory
import keras

import pytest
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import tvm.testing

if tf.executing_eagerly():
    GPUS = tf.config.experimental.list_physical_devices("GPU")
    for gpu in GPUS:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    from keras.backend.tensorflow_backend import set_session

    CONFIG = tf.ConfigProto()
    CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session(config=CONFIG))


def pytest_generate_tests(metafunc):
    """
    This function generates the list of tests for pytest, based
    on scenarios that will change the parameters in which the
    tests use to run.
    https://docs.pytest.org/en/latest/example/parametrize.html
    """
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
USING_CLASSIC_KERAS = ("keras", {"keras_mod": keras})
USING_TENSORFLOW_KERAS = ("tf_keras", {"keras_mod": tf_keras})


def verify_keras_frontend(keras_model, need_transpose=True, layout="NCHW"):
    """Generic function to generate and compare Keras and TVM output"""
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

    def get_keras_output(in_data):
        return keras_model.predict(in_data)

    def get_tvm_output(in_data, target, dev, dtype="float32"):
        shape_dict = {name: x.shape for (name, x) in zip(keras_model.input_names, in_data)}
        mod, params = relay.frontend.from_keras(keras_model, shape_dict, layout=layout)
        with tvm.transform.PassContext(opt_level=2):
            lib = relay.build(mod, target, params=params)
        m = graph_executor.GraphModule(lib["default"](dev))
        for name, x in zip(keras_model.input_names, in_data):
            m.set_input(name, tvm.nd.array(x.astype(dtype)))
        m.run()
        return [m.get_output(i).numpy() for i in range(m.get_num_outputs())]

    def to_channels_first(arr):
        return arr.transpose([0, -1] + list(range(1, arr.ndim - 1)))

    def to_channels_last(arr):
        return arr.transpose([0] + list(range(2, arr.ndim)) + [1])

    in_data = [np.random.uniform(size=shape, low=-1.0, high=1.0) for shape in in_shapes]
    keras_out = get_keras_output(in_data)
    keras_out = keras_out if isinstance(keras_out, list) else [keras_out]
    for target, dev in tvm.testing.enabled_targets():
        inputs = [to_channels_first(x) for x in in_data] if need_transpose else in_data
        tvm_out = get_tvm_output(inputs, target, dev)
        for kout, tout in zip(keras_out, tvm_out):
            if need_transpose:
                tout = to_channels_last(tout)
            tvm.testing.assert_allclose(kout, tout, rtol=1e-5, atol=1e-5)


def get_mobilenet(keras_mod):
    if hasattr(keras_mod.applications, "MobileNet"):
        # Keras 2.4.x and older
        mobilenet_mod = keras_mod.applications.MobileNet
    else:
        # Keras 2.6.x and newer
        mobilenet_mod = keras_mod.applications.mobilenet.MobileNet

    return mobilenet_mod


@tvm.testing.uses_gpu
class TestKeras:
    """Keras test"""

    scenarios = [USING_CLASSIC_KERAS, USING_TENSORFLOW_KERAS]

    def test_forward_merge(self, keras_mod):
        """test_forward_merge"""
        data = keras_mod.layers.Input(shape=(32, 32, 3))
        conv2d_x = keras_mod.layers.Conv2D(8, (3, 3), padding="same")(data)
        conv2d_y = keras_mod.layers.Conv2D(8, (3, 3), padding="same")(conv2d_x)
        conv2d_z = keras_mod.layers.Conv2D(8, (3, 3), padding="same")(conv2d_y)
        merge_funcs = [
            keras_mod.layers.Add(),
            keras_mod.layers.Subtract(),
            keras_mod.layers.Multiply(),
            keras_mod.layers.Maximum(),
            keras_mod.layers.Minimum(),
            keras_mod.layers.Average(),
            keras_mod.layers.Concatenate(),
        ]
        for merge_func in merge_funcs:
            class_name = type(merge_func).__name__
            if class_name in ("Subtract", "Dot"):
                out = merge_func([conv2d_x, conv2d_y])
            else:
                out = merge_func([conv2d_x, conv2d_y, conv2d_z])
            keras_model = keras_mod.models.Model(data, out)
            verify_keras_frontend(keras_model)

    def test_forward_concatenate(self, keras_mod):
        """test_forward_concatenate"""
        data1 = keras_mod.layers.Input(shape=(1, 2, 2))
        data2 = keras_mod.layers.Input(shape=(1, 1, 2))
        merge_func = keras_mod.layers.Concatenate(axis=2)
        out = merge_func([data1, data2])
        keras_model = keras_mod.models.Model([data1, data2], out)
        verify_keras_frontend(keras_model, layout="NHWC")
        verify_keras_frontend(keras_model, layout="NCHW")
        # test default axis (e.g., -1)
        data1 = keras_mod.layers.Input(shape=(1, 2, 2))
        data2 = keras_mod.layers.Input(shape=(1, 2, 3))
        merge_func = keras_mod.layers.Concatenate()
        out = merge_func([data1, data2])
        keras_model = keras_mod.models.Model([data1, data2], out)
        verify_keras_frontend(keras_model, layout="NHWC")
        verify_keras_frontend(keras_model, layout="NCHW")
        # test axis at last dimension
        data1 = keras_mod.layers.Input(shape=(1, 2, 2))
        data2 = keras_mod.layers.Input(shape=(1, 2, 3))
        merge_func = keras_mod.layers.Concatenate(axis=3)
        out = merge_func([data1, data2])
        keras_model = keras_mod.models.Model([data1, data2], out)
        verify_keras_frontend(keras_model, layout="NHWC")
        verify_keras_frontend(keras_model, layout="NCHW")

    def test_forward_merge_dot(self, keras_mod):
        """test_forward_merge_dot"""
        data1 = keras_mod.layers.Input(shape=(2, 2))
        data2 = keras_mod.layers.Input(shape=(2, 2))
        merge_funcs = [
            keras_mod.layers.Dot(axes=[1, 2]),
            keras_mod.layers.Dot(axes=[2, 1]),
            keras_mod.layers.Dot(axes=[1, 1]),
            keras_mod.layers.Dot(axes=[2, 2]),
            keras_mod.layers.Dot(axes=1),
            keras_mod.layers.Dot(axes=2),
        ]
        for merge_func in merge_funcs:
            out = merge_func([data1, data2])
            keras_model = keras_mod.models.Model([data1, data2], out)
            verify_keras_frontend(keras_model)

    def test_forward_activations(self, keras_mod):
        """test_forward_activations"""
        data = keras_mod.layers.Input(shape=(32, 32, 3))
        act_funcs = [
            keras_mod.layers.Activation("softmax"),
            keras_mod.layers.Softmax(),
            keras_mod.layers.Softmax(axis=-1),
            keras_mod.layers.Softmax(axis=1),
            keras_mod.layers.Softmax(axis=2),
            keras_mod.layers.Softmax(axis=3),
            keras_mod.layers.Activation("softplus"),
            keras_mod.layers.Activation("relu"),
            keras_mod.layers.Activation("softsign"),
            keras_mod.layers.Activation("hard_sigmoid"),
            keras_mod.layers.Activation("sigmoid"),
            keras_mod.layers.Activation("tanh"),
            keras_mod.layers.Activation("linear"),
            keras_mod.layers.Activation("selu"),
            keras_mod.layers.Activation("swish"),
            keras_mod.layers.ReLU(),
            keras_mod.layers.ReLU(max_value=6.0),
            keras_mod.layers.ReLU(max_value=6.0, threshold=0.0),
            keras_mod.layers.ReLU(max_value=6.0, threshold=1.0),
            keras_mod.layers.ReLU(max_value=6.0, threshold=1.0, negative_slope=0.0),
            keras_mod.layers.ReLU(max_value=6.0, threshold=1.0, negative_slope=0.5),
            keras_mod.layers.ReLU(max_value=6.0, threshold=1.0, negative_slope=1.0),
            keras_mod.layers.LeakyReLU(alpha=0.3),
            keras_mod.layers.PReLU(weights=np.random.rand(1, 32, 32, 3)),
            keras_mod.layers.ELU(alpha=0.5),
            keras_mod.layers.ThresholdedReLU(theta=0.5),
        ]
        for act_func in act_funcs:
            x = act_func(data)
            keras_model = keras_mod.models.Model(data, x)
            verify_keras_frontend(keras_model)
            verify_keras_frontend(keras_model, need_transpose=False, layout="NHWC")
        # Test the input dimension = 1
        data = keras_mod.layers.Input(shape=(11,))
        act_func = keras_mod.layers.Softmax()
        x = act_func(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model)
        verify_keras_frontend(keras_model, need_transpose=False, layout="NHWC")

    def test_forward_activations_except(self, keras_mod):
        """
        test invalid attribute alpha=None for LeakyReLU and ELU.
        after version 2.3.1 in keras, checking was added to reject the invalid api call:
        LeakyReLU(alpha=None) and ELU(alpha=None),
        (see issue: https://github.com/tensorflow/tensorflow/pull/47017)
        Thus, it's necessary to check the keras version to avoid crash at LeakyReLU(alpha=None)
        and ELU(alpha=None)
        """
        if package_version.parse(keras_mod.__version__.split("-tf")[0]) <= package_version.parse(
            "2.3.1"
        ):
            act_funcs = [
                keras_mod.layers.LeakyReLU(alpha=None),
                keras_mod.layers.ELU(2, 3, 4),
                keras_mod.layers.ReLU(threshold=None),
            ]
            data = keras_mod.layers.Input(shape=(2, 3, 4))
            for act_func in act_funcs:
                layer = act_func(data)
                keras_model = keras_mod.models.Model(data, layer)
                with pytest.raises(tvm.error.OpAttributeInvalid):
                    verify_keras_frontend(keras_model)

    def test_forward_dense(self, keras_mod):
        """test_forward_dense"""
        data = keras_mod.layers.Input(shape=(32, 32, 1))
        x = keras_mod.layers.Flatten()(data)
        x = keras_mod.layers.Dropout(0.5)(x)
        x = keras_mod.layers.Dense(10, activation="relu", kernel_initializer="uniform")(x)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model)
        # RNN dense
        data = keras_mod.layers.Input(shape=(1, 32))
        x = keras_mod.layers.Dense(32, activation="relu", kernel_initializer="uniform")(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

    def test_forward_permute(self, keras_mod):
        data = keras_mod.layers.Input(shape=(2, 3, 4))
        x = keras_mod.layers.Permute([2, 3, 1])(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

    def test_forward_sequential(self, keras_mod):
        """test_forward_sequential"""
        keras_model = keras_mod.models.Sequential(
            [
                keras_mod.layers.Dense(16, input_dim=32, activation="relu"),
                keras_mod.layers.Dropout(0.5),
                keras_mod.layers.Dense(8, activation="relu"),
                keras_mod.layers.Dropout(0.5),
                keras_mod.layers.Dense(1, activation="sigmoid"),
            ]
        )
        verify_keras_frontend(keras_model)

    def test_forward_pool(self, keras_mod):
        """test_forward_pool"""
        data = keras_mod.layers.Input(shape=(32, 32, 1))
        # maxpool
        x = keras_mod.layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model)
        # avgpool
        y = keras_mod.layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(data)
        keras_model = keras_mod.models.Model(data, y)
        verify_keras_frontend(keras_model)
        # reject the invalid input shape
        data = keras_mod.layers.Input(shape=(0, 3, 6, 4))
        x = keras_mod.layers.GlobalAveragePooling3D()(data)
        keras_model = keras_mod.models.Model(data, x)
        with pytest.raises(ValueError):
            verify_keras_frontend(keras_model)

    def test_forward_conv1d(self, keras_mod):
        """test_forward_conv1d"""
        data = keras_mod.layers.Input(shape=(32, 3))
        conv_funcs = [
            keras_mod.layers.Conv1D(filters=10, kernel_size=(3,), strides=(2,), padding="same"),
            keras_mod.layers.Conv1D(
                filters=10, kernel_size=(3,), dilation_rate=(2,), padding="same"
            ),
            keras_mod.layers.Conv1D(filters=1, kernel_size=(3,), padding="valid", use_bias=False),
            keras_mod.layers.Conv1D(filters=10, kernel_size=(2,), padding="valid"),
            # Enable when relay conv1dtranspose handles NWC
            # keras.layers.Conv1DTranspose(filters=10, kernel_size=(3), padding="valid"),
        ]
        for conv_func in conv_funcs:
            x = conv_func(data)
            keras_model = keras_mod.models.Model(data, x)
            verify_keras_frontend(keras_model, layout="NWC")

    def test_forward_conv(self, keras_mod):
        """test_forward_conv"""
        data = keras_mod.layers.Input(shape=(32, 32, 3))
        conv_funcs = [
            keras_mod.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(2, 2), padding="same"),
            keras_mod.layers.Conv2D(
                filters=10, kernel_size=(3, 3), dilation_rate=(2, 2), padding="same"
            ),
            keras_mod.layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same"),
            keras_mod.layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same"),
            keras_mod.layers.Conv2DTranspose(filters=10, kernel_size=(3, 3), padding="valid"),
            keras_mod.layers.SeparableConv2D(filters=10, kernel_size=(3, 3), padding="same"),
            keras_mod.layers.SeparableConv2D(filters=10, kernel_size=(3, 3), dilation_rate=(2, 2)),
            keras_mod.layers.SeparableConv2D(filters=2, kernel_size=(3, 3), dilation_rate=2),
        ]
        for conv_func in conv_funcs:
            x = conv_func(data)
            keras_model = keras_mod.models.Model(data, x)
            verify_keras_frontend(keras_model)

    def test_forward_conv_transpose(self, keras_mod):
        """test_forward_conv_transpose"""
        data = keras_mod.layers.Input(shape=(32, 32, 128))
        conv_funcs = [
            keras_mod.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), padding="valid"),
            keras_mod.layers.Conv2DTranspose(
                filters=2, kernel_size=(3, 3), strides=(2, 2), output_padding=(1, 1)
            ),
        ]
        for conv_func in conv_funcs:
            x = conv_func(data)
            keras_model = keras_mod.models.Model(data, x)
            verify_keras_frontend(keras_model, layout="NHWC")

    def test_forward_batch_norm(self, keras_mod):
        """test_forward_batch_norm"""
        data = keras_mod.layers.Input(shape=(32, 32, 3))
        batch_norm_funcs = [
            keras_mod.layers.BatchNormalization(
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
            keras_mod.layers.BatchNormalization(
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
            keras_mod.layers.BatchNormalization(
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
            keras_mod.layers.BatchNormalization(
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
            keras_model = keras_mod.models.Model(data, x)
            verify_keras_frontend(keras_model)

    def test_forward_upsample(self, keras_mod, interpolation="nearest"):
        data = keras_mod.layers.Input(shape=(32, 32, 3))
        x = keras_mod.layers.UpSampling2D(size=(3, 3), interpolation=interpolation)(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model)
        # Height and width are not equal for the attribute size
        data = keras_mod.layers.Input(shape=(2, 1, 3))
        x = keras_mod.layers.UpSampling2D(size=(1, 2), interpolation=interpolation)(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model)

    def test_forward_reshape(self, keras_mod):
        """test_forward_reshape"""
        # input_shape len is 3, target_shape len is 3
        data = keras_mod.layers.Input(shape=(32, 32, 3))
        x = keras_mod.layers.Reshape(target_shape=(16, 64, 3))(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model)
        # input_shape len is 3, target_shape len is 2
        data = keras_mod.layers.Input(shape=(32, 8, 3))
        x = keras_mod.layers.Reshape(target_shape=(256, 3))(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model)
        # input_shape len is 2, target_shape len is 3
        data = keras_mod.layers.Input(shape=(256, 3))
        x = keras_mod.layers.Reshape(target_shape=(8, 32, 3))(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model)
        # input_shape len is 2, target_shape len is 1
        data = keras_mod.layers.Input(shape=(2, 8))
        x = keras_mod.layers.Reshape(target_shape=(16,))(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)
        # input_shape len is 1, target_shape len is 2
        data = keras_mod.layers.Input(shape=(16,))
        x = keras_mod.layers.Reshape(target_shape=(4, 4))(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)
        # input_shape len is 2, target_shape len is 2
        data = keras_mod.layers.Input(shape=(2, 8))
        x = keras_mod.layers.Reshape(target_shape=(4, 4))(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)
        # "non-square" target shape
        data = keras_mod.layers.Input(shape=(15,))
        x = keras_mod.layers.Reshape(target_shape=(5, 3))(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)
        # modify channel dim
        data = keras_mod.layers.Input(shape=(3, 2, 4))
        x = keras_mod.layers.Reshape(target_shape=(3, 8))(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model)

    def test_forward_crop(self, keras_mod):
        """test_forward_crop"""
        data = keras_mod.layers.Input(shape=(32, 32, 3))
        x = keras_mod.layers.Cropping2D(cropping=((1, 1), (1, 1)))(data)
        x = keras_mod.layers.Cropping2D(cropping=(1, 1))(x)
        x = keras_mod.layers.Cropping2D(cropping=1)(x)
        x = keras_mod.layers.Cropping2D(cropping=((0, 1), (1, 0)))(x)
        x = keras_mod.layers.Cropping2D(cropping=(1, 0))(x)
        x = keras_mod.layers.Cropping2D(cropping=0)(x)
        x = keras_mod.layers.Add()([x, x])
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model, layout="NHWC")
        verify_keras_frontend(keras_model, layout="NHWC")

        data = keras_mod.layers.Input(shape=(32, 32, 3))
        x = keras_mod.layers.Cropping2D(cropping=(2, 1))(data)
        x = keras_mod.layers.Cropping2D(cropping=(1, 2))(x)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model, layout="NHWC")
        verify_keras_frontend(keras_model, layout="NCHW")

    def test_forward_multi_inputs(self, keras_mod):
        data1 = keras_mod.layers.Input(shape=(32, 32, 3))
        data2 = keras_mod.layers.Input(shape=(32, 32, 3))
        x = keras_mod.layers.Conv2D(8, (3, 3), padding="same")(data1)
        y = keras_mod.layers.Conv2D(8, (3, 3), padding="same")(data2)
        average_z = keras_mod.layers.Average()([x, y])
        out = keras_mod.layers.GlobalAveragePooling2D()(average_z)
        keras_model = keras_mod.models.Model([data1, data2], out)
        verify_keras_frontend(keras_model)

    def test_forward_multi_outputs(self, keras_mod):
        data = keras_mod.layers.Input(shape=(32, 32, 3))
        x = keras_mod.layers.Conv2D(8, (3, 3), padding="same")(data)
        x = keras_mod.layers.GlobalAveragePooling2D()(x)
        y = keras_mod.layers.Conv2D(8, (3, 3), padding="same")(data)
        y = keras_mod.layers.GlobalAveragePooling2D()(y)
        keras_model = keras_mod.models.Model(data, [x, y])
        verify_keras_frontend(keras_model)

    def test_forward_reuse_layers(self, keras_mod):
        """test_forward_reuse_layers"""
        # reuse conv2d
        data = keras_mod.layers.Input(shape=(32, 32, 3))
        conv2d = keras_mod.layers.Conv2D(8, (3, 3), padding="same")
        x = conv2d(data)
        y = conv2d(data)
        add_z = keras_mod.layers.Add()([x, y])
        out = keras_mod.layers.GlobalAveragePooling2D()(add_z)
        keras_model = keras_mod.models.Model(data, out)
        verify_keras_frontend(keras_model)
        # reuse add
        data = keras_mod.layers.Input(shape=(32, 32, 3))
        x = keras_mod.layers.Conv2D(8, (3, 3), padding="same")(data)
        add = keras_mod.layers.Add()
        x = add([x, x])
        x = add([x, x])
        out = keras_mod.layers.GlobalAveragePooling2D()(x)
        keras_model = keras_mod.models.Model(data, out)
        verify_keras_frontend(keras_model)

    def test_forward_lstm(self, keras_mod):
        """test_forward_lstm"""
        data = keras_mod.layers.Input(shape=(10, 32))
        rnn_funcs = [
            keras_mod.layers.LSTM(16),
            keras_mod.layers.LSTM(16, return_sequences=True),
            keras_mod.layers.LSTM(16, go_backwards=True),
            keras_mod.layers.LSTM(16, return_sequences=True, go_backwards=True),
            keras_mod.layers.LSTM(16, return_sequences=True, use_bias=False),
        ]
        for rnn_func in rnn_funcs:
            x = rnn_func(data)
            keras_model = keras_mod.models.Model(data, x)
            verify_keras_frontend(keras_model, need_transpose=False)

    def test_forward_rnn(self, keras_mod):
        """test_forward_rnn"""
        data = keras_mod.layers.Input(shape=(1, 32))
        rnn_funcs = [
            keras_mod.layers.LSTM(
                units=16, return_state=False, recurrent_activation="sigmoid", activation="tanh"
            ),
            keras_mod.layers.LSTM(
                units=16,
                return_state=False,
                recurrent_activation="sigmoid",
                activation="tanh",
                use_bias=False,
            ),
            keras_mod.layers.SimpleRNN(units=16, return_state=False, activation="tanh"),
            keras_mod.layers.SimpleRNN(
                units=16, return_state=False, activation="tanh", use_bias=False
            ),
            keras_mod.layers.SimpleRNN(
                units=16, return_state=False, activation="tanh", go_backwards=True
            ),
            keras_mod.layers.GRU(
                units=16,
                return_state=False,
                recurrent_activation="sigmoid",
                activation="tanh",
                reset_after=False,
            ),
            keras_mod.layers.GRU(
                units=16,
                return_state=False,
                recurrent_activation="sigmoid",
                activation="tanh",
                reset_after=False,
                use_bias=False,
            ),
            keras_mod.layers.GRU(
                units=16,
                return_state=False,
                recurrent_activation="sigmoid",
                activation="tanh",
                reset_after=False,
                use_bias=False,
                go_backwards=True,
            ),
        ]
        for rnn_func in rnn_funcs:
            x = rnn_func(data)
            keras_model = keras_mod.models.Model(data, x)
            verify_keras_frontend(keras_model, need_transpose=False)

    def test_forward_vgg16(self, keras_mod, layout="NCHW"):
        """test_forward_vgg16"""
        if hasattr(keras_mod.applications, "VGG16"):
            # Keras 2.4.x and older
            vgg16_mod = keras_mod.applications.VGG16
        else:
            # Keras 2.6.x and newer
            vgg16_mod = keras_mod.applications.vgg16.VGG16

        keras_model = vgg16_mod(
            include_top=True, weights="imagenet", input_shape=(224, 224, 3), classes=1000
        )
        verify_keras_frontend(keras_model, layout=layout)

    def test_forward_xception(self, keras_mod, layout="NCHW"):
        """test_forward_vgg16"""
        if hasattr(keras_mod.applications, "Xception"):
            # Keras 2.4.x and older
            xception_mod = keras_mod.applications.Xception
        else:
            # Keras 2.6.x and newer
            xception_mod = keras_mod.applications.xception.Xception

        keras_model = xception_mod(
            include_top=True, weights="imagenet", input_shape=(299, 299, 3), classes=1000
        )
        verify_keras_frontend(keras_model, layout=layout)

    def test_forward_resnet50(self, keras_mod, layout="NCHW"):
        """test_forward_resnet50"""
        if hasattr(keras_mod.applications, "ResNet50"):
            # Keras 2.4.x and older
            resnet50_mod = keras_mod.applications.ResNet50
        else:
            # Keras 2.6.x and newer
            resnet50_mod = keras_mod.applications.resnet.ResNet50

        keras_model = resnet50_mod(
            include_top=True, weights="imagenet", input_shape=(224, 224, 3), classes=1000
        )
        verify_keras_frontend(keras_model, layout=layout)

    def test_forward_inception_v3(self, keras_mod, layout="NCHW"):
        """test_forward_inception_v3"""
        if hasattr(keras_mod.applications, "InceptionV3"):
            # Keras 2.4.x and older
            inception_v3_mod = keras_mod.applications.InceptionV3
        else:
            # Keras 2.6.x and newer
            inception_v3_mod = keras_mod.applications.inception_v3.InceptionV3

        keras_model = inception_v3_mod(
            include_top=True, weights=None, input_shape=(299, 299, 3), classes=1000
        )
        verify_keras_frontend(keras_model, layout=layout)

    def test_forward_mobilenet(self, keras_mod, layout="NCHW"):
        mobilenet_mod = get_mobilenet(keras_mod)

        keras_model = mobilenet_mod(
            include_top=True, weights="imagenet", input_shape=(224, 224, 3), classes=1000
        )
        verify_keras_frontend(keras_model, layout=layout)

    def test_forward_conv3d(self, keras_mod):
        """test_forward_conv3d"""
        data = keras_mod.layers.Input(shape=(32, 32, 32, 3))
        conv_funcs = [
            keras_mod.layers.Conv3D(
                filters=10, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same"
            ),
            keras_mod.layers.Conv3D(
                filters=10, kernel_size=(3, 3, 3), dilation_rate=(2, 2, 2), padding="same"
            ),
            keras_mod.layers.Conv3D(
                filters=1, kernel_size=(3, 3, 3), padding="valid", use_bias=False
            ),
            keras_mod.layers.Conv3D(filters=10, kernel_size=(2, 2, 2), padding="valid"),
        ]
        for conv_func in conv_funcs:
            x = conv_func(data)
            keras_model = keras_mod.models.Model(data, x)
            verify_keras_frontend(keras_model, layout="NDHWC")

    def test_forward_conv3d_transpose(self, keras_mod):
        """test_forward_conv3d_transpose"""
        data = keras_mod.layers.Input(shape=(32, 32, 32, 3))
        conv_funcs = [
            keras_mod.layers.Conv3DTranspose(
                filters=10, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same"
            ),
            keras_mod.layers.Conv3DTranspose(
                filters=10, kernel_size=(1, 1, 1), dilation_rate=(1, 1, 1), padding="same"
            ),
            keras_mod.layers.Conv3DTranspose(
                filters=1, kernel_size=(3, 3, 3), padding="valid", use_bias=False
            ),
            keras_mod.layers.Conv3DTranspose(filters=10, kernel_size=(2, 2, 2), padding="valid"),
            keras_mod.layers.Conv3DTranspose(
                filters=2, kernel_size=(3, 3, 3), strides=(2, 2, 2), output_padding=(1, 1, 1)
            ),
        ]
        for conv_func in conv_funcs:
            x = conv_func(data)
            keras_model = keras_mod.models.Model(data, x)
            verify_keras_frontend(keras_model, layout="NDHWC")

    def test_forward_pool3d(self, keras_mod):
        """test_forward_pool3d"""
        data = keras_mod.layers.Input(shape=(32, 32, 32, 1))
        pool_funcs = [  # maxpool
            keras_mod.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding="same"),
            keras_mod.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="valid"),
            # avgpool
            keras_mod.layers.AveragePooling3D(
                pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same"
            ),
            keras_mod.layers.AveragePooling3D(
                pool_size=(2, 2, 2), strides=(1, 1, 1), padding="valid"
            ),
        ]
        for pool_func in pool_funcs:
            x = pool_func(data)
            keras_model = keras_mod.models.Model(data, x)
            verify_keras_frontend(keras_model, layout="NDHWC")

    def test_forward_upsample3d(self, keras_mod):
        data = keras_mod.layers.Input(shape=(32, 32, 32, 3))
        x = keras_mod.layers.UpSampling3D(size=(2, 3, 4))(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model, layout="NDHWC")

    def test_forward_zero_padding3d(self, keras_mod):
        """test_forward_zero_padding3d"""
        data = keras_mod.layers.Input(shape=(32, 32, 32, 3))
        pad_funcs = [  # Integer
            keras_mod.layers.ZeroPadding3D(padding=2),
            # tuple of 3 ints
            keras_mod.layers.ZeroPadding3D(padding=(1, 2, 3)),
            # tuple of 3 tuples of 2 ints
            keras_mod.layers.ZeroPadding3D(padding=((1, 1), (2, 2), (2, 2))),
            # tuple of 3 tuples of 2 ints different values
            keras_mod.layers.ZeroPadding3D(padding=((1, 2), (2, 3), (3, 2))),
        ]
        for pad_func in pad_funcs:
            x = pad_func(data)
            keras_model = keras_mod.models.Model(data, x)
            verify_keras_frontend(keras_model, layout="NDHWC")

    def test_forward_embedding(self, keras_mod):
        """test_forward_embedding"""
        data = keras_mod.layers.Input(shape=(2, 4), dtype="int32")
        x = keras_mod.layers.Embedding(10, 3)(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

        data = keras_mod.layers.Input(shape=(2, 3, 4), dtype="int32")
        x = keras_mod.layers.Embedding(4, 5)(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

        data = keras_mod.layers.Input(shape=(6, 2, 3, 4), dtype="int32")
        x = keras_mod.layers.Embedding(4, 5)(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

    def test_forward_repeat_vector(self, keras_mod):
        """test_forward_repeat_vector"""
        data = keras_mod.layers.Input(shape=(5,), dtype="float32")
        x = keras_mod.layers.Dense(6)(data)
        x = keras_mod.layers.RepeatVector(2)(x)

        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

        data = keras_mod.layers.Input(shape=(10,), dtype="float32")
        x = keras_mod.layers.RepeatVector(3)(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

        data = keras_mod.layers.Input(shape=(4,), dtype="float32")
        x = keras_mod.layers.RepeatVector(1)(data)
        keras_model = keras_mod.models.Model(data, x)
        verify_keras_frontend(keras_model, need_transpose=False)

    def test_forward_global_pool3d(self, keras_mod):
        """test_forward_zero_padding3d"""
        data = keras_mod.layers.Input(shape=(32, 32, 32, 1))
        pool_funcs = [  # global maxpool
            keras_mod.layers.GlobalMaxPooling3D(),
            # global avgpool
            keras_mod.layers.GlobalAveragePooling3D(),
        ]
        for pool_func in pool_funcs:
            x = pool_func(data)
            keras_model = keras_mod.models.Model(data, x)
            verify_keras_frontend(keras_model, layout="NDHWC")

    def test_forward_nested_layers(self, keras_mod):
        """test_forward_nested_layers"""
        mobilenet_mod = get_mobilenet(keras_mod)

        sub_model = mobilenet_mod(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        keras_model = keras_mod.Sequential(
            [
                sub_model,
                keras_mod.layers.GlobalAveragePooling2D(),
                keras_mod.layers.Dense(1024, activation="relu"),
                keras_mod.layers.Dense(2, activation="sigmoid"),
            ]
        )
        verify_keras_frontend(keras_model)

    def test_forward_l2_normalize(self, keras_mod):
        """test_forward_l2_normalize"""
        data = keras_mod.layers.Input(shape=(16, 12, 8))
        k_backend = keras_mod.backend
        l2_funcs = [
            keras_mod.layers.Lambda(lambda v: k_backend.l2_normalize(v, axis=-2)),
            keras_mod.layers.Lambda(lambda v: k_backend.l2_normalize(x=v, axis=-1)),
            keras_mod.layers.Lambda(lambda v: k_backend.l2_normalize(axis=1, x=v)),
            keras_mod.layers.Lambda(lambda v: k_backend.l2_normalize(v, 2)),
            keras_mod.layers.Lambda(lambda v: k_backend.l2_normalize(v, axis=3)),
            keras_mod.layers.Lambda(lambda v: k_backend.l2_normalize(v, axis=(2, 3))),
            keras_mod.layers.Lambda(lambda v: k_backend.l2_normalize(v, (1, 2))),
            keras_mod.layers.Lambda(lambda v: k_backend.l2_normalize(v, axis=[-2, -1])),
            keras_mod.layers.Lambda(lambda v: k_backend.l2_normalize(v, [-3, -2])),
        ]
        for l2_func in l2_funcs:
            x = l2_func(data)
            keras_model = keras_mod.models.Model(data, x)
            verify_keras_frontend(keras_model, layout="NCHW")
            verify_keras_frontend(keras_model, layout="NHWC")

    def test_forward_time_distributed(self, keras_mod):
        """test_forward_time_distributed"""
        conv2d_inputs = keras_mod.Input(shape=(10, 128, 128, 3))
        conv_2d_layer = keras_mod.layers.Conv2D(64, (3, 3))
        conv2d_model = keras_mod.models.Model(
            conv2d_inputs, keras_mod.layers.TimeDistributed(conv_2d_layer)(conv2d_inputs)
        )
        verify_keras_frontend(conv2d_model, layout="NDHWC")

        dense_inputs = keras_mod.Input(shape=(5, 1))
        dense_layer = keras_mod.layers.Dense(1)
        dense_model = keras_mod.models.Model(
            dense_inputs, keras_mod.layers.TimeDistributed(dense_layer)(dense_inputs)
        )
        verify_keras_frontend(dense_model, need_transpose=False)

    def test_simplernn_with_infertype(self, keras_mod):
        """This test case is from https://github.com/apache/tvm/issues/14868"""
        input_shape = (2, 2, 2)
        x = keras_mod.layers.Input(shape=input_shape[1:], dtype="float32")
        layer = keras_mod.layers.SimpleRNN(units=4)
        y = layer(x)
        model = keras_mod.models.Model(x, y)
        mod, _ = relay.frontend.from_keras(model, {model.input_names[0]: input_shape})
        relay.transform.InferType()(mod)


if __name__ == "__main__":
    for k in [keras, tf_keras]:
        sut = TestKeras()
        sut.test_forward_concatenate(keras_mod=k)
        sut.test_forward_merge_dot(keras_mod=k)
        sut.test_forward_merge(keras_mod=k)
        sut.test_forward_activations(keras_mod=k)
        sut.test_forward_activations_except(keras_mod=k)
        sut.test_forward_dense(keras_mod=k)
        sut.test_forward_permute(keras_mod=k)
        sut.test_forward_sequential(keras_mod=k)
        sut.test_forward_pool(keras_mod=k)
        sut.test_forward_conv(keras_mod=k)
        sut.test_forward_conv1d(keras_mod=k)
        sut.test_forward_batch_norm(keras_mod=k)
        sut.test_forward_upsample(keras_mod=k, interpolation="nearest")
        sut.test_forward_upsample(keras_mod=k, interpolation="bilinear")
        sut.test_forward_reshape(keras_mod=k)
        sut.test_forward_crop(keras_mod=k)
        sut.test_forward_multi_inputs(keras_mod=k)
        sut.test_forward_multi_outputs(keras_mod=k)
        sut.test_forward_reuse_layers(keras_mod=k)
        sut.test_forward_lstm(keras_mod=k)
        sut.test_forward_rnn(keras_mod=k)
        sut.test_forward_vgg16(keras_mod=k)
        sut.test_forward_vgg16(keras_mod=k, layout="NHWC")
        sut.test_forward_xception(keras_mod=k)
        sut.test_forward_resnet50(keras_mod=k)
        sut.test_forward_resnet50(keras_mod=k, layout="NHWC")
        sut.test_forward_inception_v3(keras_mod=k)
        sut.test_forward_inception_v3(keras_mod=k, layout="NHWC")
        sut.test_forward_mobilenet(keras_mod=k)
        sut.test_forward_mobilenet(keras_mod=k, layout="NHWC")
        sut.test_forward_conv3d(keras_mod=k)
        sut.test_forward_conv3d_transpose(keras_mod=k)
        sut.test_forward_pool3d(keras_mod=k)
        sut.test_forward_global_pool3d(keras_mod=k)
        sut.test_forward_upsample3d(keras_mod=k)
        sut.test_forward_zero_padding3d(keras_mod=k)
        sut.test_forward_embedding(keras_mod=k)
        sut.test_forward_repeat_vector(keras_mod=k)
        sut.test_forward_l2_normalize(keras_mod=k)
        sut.test_forward_time_distributed(keras_mod=k)
        sut.test_simplernn_with_infertype(keras_mod=k)
