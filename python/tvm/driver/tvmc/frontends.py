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
Provides support to parse models from different frameworks into Relay networks.

Frontend classes do lazy-loading of modules on purpose, to reduce time spent on
loading the tool.
"""
import logging
import os
import sys
from abc import ABC
from abc import abstractmethod
from pathlib import Path

import numpy as np

from tvm import relay
from tvm.driver.tvmc.common import TVMCException


class Frontend(ABC):
    """Abstract class for command line driver frontend.

    Provide a unified way to import models (as files), and deal
    with any required preprocessing to create a TVM module from it."""

    @staticmethod
    @abstractmethod
    def name():
        """Frontend name"""

    @staticmethod
    @abstractmethod
    def suffixes():
        """File suffixes (extensions) used by this frontend"""

    @abstractmethod
    def load(self, path):
        """Load a model from a given path.

        Parameters
        ----------
        path: str
            Path to a file

        Returns
        -------
        mod : tvm.relay.Module
            The produced relay module.
        params : dict
            The parameters (weights) for the relay module.

        """


def import_keras():
    """ Lazy import function for Keras"""
    # Keras writes the message "Using TensorFlow backend." to stderr
    # Redirect stderr during the import to disable this
    stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        # pylint: disable=C0415
        import tensorflow as tf
        from tensorflow import keras

        return tf, keras
    finally:
        sys.stderr = stderr


class KerasFrontend(Frontend):
    """ Keras frontend for TVMC """

    @staticmethod
    def name():
        return "keras"

    @staticmethod
    def suffixes():
        return ["h5"]

    def load(self, path):
        # pylint: disable=C0103
        tf, keras = import_keras()

        # tvm build currently imports keras directly instead of tensorflow.keras
        try:
            model = keras.models.load_model(path)
        except ValueError as err:
            raise TVMCException(str(err))

        # There are two flavours of keras model, sequential and
        # functional, TVM expects a functional model, so convert
        # if required:
        if self.is_sequential_p(model):
            model = self.sequential_to_functional(model)

        in_shapes = []
        for layer in model._input_layers:
            if tf.executing_eagerly():
                in_shapes.append(tuple(dim if dim is not None else 1 for dim in layer.input.shape))
            else:
                in_shapes.append(
                    tuple(dim.value if dim.value is not None else 1 for dim in layer.input.shape)
                )

        inputs = [np.random.uniform(size=shape, low=-1.0, high=1.0) for shape in in_shapes]
        shape_dict = {name: x.shape for (name, x) in zip(model.input_names, inputs)}
        return relay.frontend.from_keras(model, shape_dict, layout="NHWC")

    def is_sequential_p(self, model):
        _, keras = import_keras()
        return isinstance(model, keras.models.Sequential)

    def sequential_to_functional(self, model):
        _, keras = import_keras()
        assert self.is_sequential_p(model)
        input_layer = keras.layers.Input(batch_shape=model.layers[0].input_shape)
        prev_layer = input_layer
        for layer in model.layers:
            prev_layer = layer(prev_layer)
        model = keras.models.Model([input_layer], [prev_layer])
        return model


class OnnxFrontend(Frontend):
    """ ONNX frontend for TVMC """

    @staticmethod
    def name():
        return "onnx"

    @staticmethod
    def suffixes():
        return ["onnx"]

    def load(self, path):
        # pylint: disable=C0415
        import onnx

        # pylint: disable=E1101
        model = onnx.load(path)

        # pylint: disable=E1101
        name = model.graph.input[0].name

        # pylint: disable=E1101
        proto_shape = model.graph.input[0].type.tensor_type.shape.dim
        shape = [d.dim_value for d in proto_shape]

        shape_dict = {name: shape}

        return relay.frontend.from_onnx(model, shape_dict)


class TensorflowFrontend(Frontend):
    """ TensorFlow frontend for TVMC """

    @staticmethod
    def name():
        return "pb"

    @staticmethod
    def suffixes():
        return ["pb"]

    def load(self, path):
        # pylint: disable=C0415
        import tensorflow as tf
        import tvm.relay.testing.tf as tf_testing

        with tf.io.gfile.GFile(path, "rb") as tf_graph:
            content = tf_graph.read()

        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(content)
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        logging.debug("relay.frontend.from_tensorflow")
        return relay.frontend.from_tensorflow(graph_def)


class TFLiteFrontend(Frontend):
    """ TFLite frontend for TVMC """

    _tflite_m = {
        0: "float32",
        1: "float16",
        2: "int32",
        3: "uint8",
        4: "int64",
        5: "string",
        6: "bool",
        7: "int16",
        8: "complex64",
        9: "int8",
    }

    @staticmethod
    def name():
        return "tflite"

    @staticmethod
    def suffixes():
        return ["tflite"]

    def load(self, path):
        # pylint: disable=C0415
        import tflite.Model as model

        with open(path, "rb") as tf_graph:
            content = tf_graph.read()

        # tflite.Model.Model is tflite.Model in 1.14 and 2.1.0
        try:
            tflite_model = model.Model.GetRootAsModel(content, 0)
        except AttributeError:
            tflite_model = model.GetRootAsModel(content, 0)

        try:
            version = tflite_model.Version()
            logging.debug("tflite version %s", version)
        except Exception:
            raise TVMCException("input file not tflite")

        if version != 3:
            raise TVMCException("input file not tflite version 3")

        logging.debug("tflite_input_type")
        shape_dict, dtype_dict = TFLiteFrontend._input_type(tflite_model)

        # parse TFLite model and convert into Relay computation graph
        logging.debug("relay.frontend.from_tflite")
        mod, params = relay.frontend.from_tflite(
            tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict
        )
        return mod, params

    @staticmethod
    def _decode_type(n):
        return TFLiteFrontend._tflite_m[n]

    @staticmethod
    def _input_type(model):
        subgraph_count = model.SubgraphsLength()
        assert subgraph_count > 0
        shape_dict = {}
        dtype_dict = {}
        for subgraph_index in range(subgraph_count):
            subgraph = model.Subgraphs(subgraph_index)
            inputs_count = subgraph.InputsLength()
            assert inputs_count >= 1
            for input_index in range(inputs_count):
                input_ = subgraph.Inputs(input_index)
                assert subgraph.TensorsLength() > input_
                tensor = subgraph.Tensors(input_)
                input_shape = tuple(tensor.ShapeAsNumpy())
                tensor_type = tensor.Type()
                input_name = tensor.Name().decode("utf8")
                shape_dict[input_name] = input_shape
                dtype_dict[input_name] = TFLiteFrontend._decode_type(tensor_type)

        return shape_dict, dtype_dict


class PyTorchFrontend(Frontend):
    """ PyTorch frontend for TVMC """

    @staticmethod
    def name():
        return "pytorch"

    @staticmethod
    def suffixes():
        # Torch Script is a zip file, but can be named pth
        return ["pth", "zip"]

    def load(self, path):
        # pylint: disable=C0415
        import torch

        traced_model = torch.jit.load(path)

        inputs = list(traced_model.graph.inputs())[1:]
        input_shapes = [inp.type().sizes() for inp in inputs]

        traced_model.eval()  # Switch to inference mode
        input_shapes = [("input{}".format(idx), shape) for idx, shape in enumerate(shapes)]
        logging.debug("relay.frontend.from_pytorch")
        return relay.frontend.from_pytorch(traced_model, input_shapes)


ALL_FRONTENDS = [
    KerasFrontend,
    OnnxFrontend,
    TensorflowFrontend,
    TFLiteFrontend,
    PyTorchFrontend,
]


def get_frontend_names():
    """Return the names of all supported frontends

    Returns
    -------
    list : list of str
        A list of frontend names as strings

    """
    return [frontend.name() for frontend in ALL_FRONTENDS]


def get_frontend_by_name(name):
    """
    This function will try to get a frontend instance, based
    on the name provided.

    Parameters
    ----------
    name : str
        the name of a given frontend

    Returns
    -------
    frontend : tvm.driver.tvmc.Frontend
        An instance of the frontend that matches with
        the file extension provided in `path`.

    """

    for frontend in ALL_FRONTENDS:
        if name == frontend.name():
            return frontend()

    raise TVMCException(
        "unrecognized frontend '{0}'. Choose from: {1}".format(name, get_frontend_names())
    )


def guess_frontend(path):
    """
    This function will try to imply which framework is being used,
    based on the extension of the file provided in the path parameter.

    Parameters
    ----------
    path : str
        The path to the model file.

    Returns
    -------
    frontend : tvm.driver.tvmc.Frontend
        An instance of the frontend that matches with
        the file extension provided in `path`.

    """

    suffix = Path(path).suffix.lower()
    if suffix.startswith("."):
        suffix = suffix[1:]

    for frontend in ALL_FRONTENDS:
        if suffix in frontend.suffixes():
            return frontend()

    raise TVMCException("failed to infer the model format. Please specify --model-format")


def load_model(path, model_format=None):
    """Load a model from a supported framework and convert it
    into an equivalent relay representation.

    Parameters
    ----------
    path : str
        The path to the model file.
    model_format : str, optional
        The underlying framework used to create the model.
        If not specified, this will be inferred from the file type.

    Returns
    -------
    mod : tvm.relay.Module
        The produced relay module.
    params : dict
        The parameters (weights) for the relay module.

    """

    if model_format is not None:
        frontend = get_frontend_by_name(model_format)
    else:
        frontend = guess_frontend(path)

    mod, params = frontend.load(path)

    return mod, params
