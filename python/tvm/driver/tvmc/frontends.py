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
import re
import importlib
from abc import ABC
from abc import abstractmethod
from typing import Optional, List, Dict
from pathlib import Path

import numpy as np

from tvm import relay
from tvm import parser
from tvm.driver.tvmc import TVMCException, TVMCImportError
from tvm.driver.tvmc.model import TVMCModel


# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


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
    def load(self, path, shape_dict=None, **kwargs):
        """Load a model from a given path.

        Parameters
        ----------
        path: str
            Path to a file
        shape_dict: dict, optional
            Mapping from input names to their shapes.

        Returns
        -------
        mod : tvm.IRModule
            The produced relay module.
        params : dict
            The parameters (weights) for the relay module.

        """


def lazy_import(pkg_name, from_pkg_name=None, hide_stderr=False):
    """Lazy import a frontend package or subpackage"""
    try:
        return importlib.import_module(pkg_name, package=from_pkg_name)
    except ImportError as error:
        raise TVMCImportError(pkg_name) from error
    finally:
        if hide_stderr:
            sys.stderr = stderr


class KerasFrontend(Frontend):
    """Keras frontend for TVMC"""

    @staticmethod
    def name():
        return "keras"

    @staticmethod
    def suffixes():
        return ["h5"]

    def load(self, path, shape_dict=None, **kwargs):
        # pylint: disable=C0103
        tf = lazy_import("tensorflow")
        keras = lazy_import("keras", from_pkg_name="tensorflow")

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
        input_shapes = {name: x.shape for (name, x) in zip(model.input_names, inputs)}
        if shape_dict is not None:
            input_shapes.update(shape_dict)
        kwargs.setdefault("layout", "NHWC")
        return relay.frontend.from_keras(model, input_shapes, **kwargs)

    def is_sequential_p(self, model):
        keras = lazy_import("keras", from_pkg_name="tensorflow")
        return isinstance(model, keras.models.Sequential)

    def sequential_to_functional(self, model):
        keras = lazy_import("keras", from_pkg_name="tensorflow")
        assert self.is_sequential_p(model)
        input_layer = keras.layers.Input(batch_shape=model.layers[0].input_shape)
        prev_layer = input_layer
        for layer in model.layers:
            prev_layer = layer(prev_layer)
        model = keras.models.Model([input_layer], [prev_layer])
        return model


class OnnxFrontend(Frontend):
    """ONNX frontend for TVMC"""

    @staticmethod
    def name():
        return "onnx"

    @staticmethod
    def suffixes():
        return ["onnx"]

    def load(self, path, shape_dict=None, **kwargs):
        onnx = lazy_import("onnx")

        # pylint: disable=E1101
        model = onnx.load(path)

        return relay.frontend.from_onnx(model, shape=shape_dict, **kwargs)


class TensorflowFrontend(Frontend):
    """TensorFlow frontend for TVMC"""

    @staticmethod
    def name():
        return "pb"

    @staticmethod
    def suffixes():
        return ["pb"]

    def load(self, path, shape_dict=None, **kwargs):
        tf = lazy_import("tensorflow")
        tf_testing = lazy_import("tvm.relay.testing.tf")

        with tf.io.gfile.GFile(path, "rb") as tf_graph:
            content = tf_graph.read()

        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(content)
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        logger.debug("parse TensorFlow model and convert into Relay computation graph")
        return relay.frontend.from_tensorflow(graph_def, shape=shape_dict, **kwargs)


class TFLiteFrontend(Frontend):
    """TFLite frontend for TVMC"""

    @staticmethod
    def name():
        return "tflite"

    @staticmethod
    def suffixes():
        return ["tflite"]

    def load(self, path, shape_dict=None, **kwargs):
        model = lazy_import("tflite.Model")

        with open(path, "rb") as tf_graph:
            content = tf_graph.read()

        # tflite.Model.Model is tflite.Model in 1.14 and 2.1.0
        try:
            tflite_model = model.Model.GetRootAsModel(content, 0)
        except AttributeError:
            tflite_model = model.GetRootAsModel(content, 0)

        try:
            version = tflite_model.Version()
            logger.debug("tflite version %s", version)
        except Exception:
            raise TVMCException("input file not tflite")

        if version != 3:
            raise TVMCException("input file not tflite version 3")

        logger.debug("parse TFLite model and convert into Relay computation graph")
        mod, params = relay.frontend.from_tflite(tflite_model, shape_dict=shape_dict, **kwargs)
        return mod, params


class PyTorchFrontend(Frontend):
    """PyTorch frontend for TVMC"""

    @staticmethod
    def name():
        return "pytorch"

    @staticmethod
    def suffixes():
        # Torch Script is a zip file, but can be named pth
        return ["pth", "zip"]

    def load(self, path, shape_dict=None, **kwargs):
        torch = lazy_import("torch")

        if shape_dict is None:
            raise TVMCException("--input-shapes must be specified for %s" % self.name())

        traced_model = torch.jit.load(path)
        traced_model.eval()  # Switch to inference mode

        # Convert shape dictionary to list for Pytorch frontend compatibility
        input_shapes = list(shape_dict.items())

        logger.debug("parse Torch model and convert into Relay computation graph")
        return relay.frontend.from_pytorch(
            traced_model, input_shapes, keep_quantized_weight=True, **kwargs
        )


class PaddleFrontend(Frontend):
    """PaddlePaddle frontend for TVMC"""

    @staticmethod
    def name():
        return "paddle"

    @staticmethod
    def suffixes():
        return ["pdmodel"]

    def load(self, path, shape_dict=None, **kwargs):
        # pylint: disable=C0415
        import paddle

        paddle.enable_static()
        paddle.disable_signal_handler()

        if not os.path.exists(path):
            raise TVMCException("File {} is not exist.".format(path))
        if not path.endswith(".pdmodel"):
            raise TVMCException("Path of model file should be endwith suffixes '.pdmodel'.")
        prefix = "".join(path.strip().split(".")[:-1])
        params_file_path = prefix + ".pdiparams"
        if not os.path.exists(params_file_path):
            raise TVMCException("File {} is not exist.".format(params_file_path))

        # pylint: disable=E1101
        exe = paddle.static.Executor(paddle.CPUPlace())
        prog, _, _ = paddle.static.load_inference_model(prefix, exe)

        return relay.frontend.from_paddle(prog, shape_dict=shape_dict, **kwargs)


class RelayFrontend(Frontend):
    """Relay frontend for TVMC"""

    @staticmethod
    def name():
        return "relay"

    @staticmethod
    def suffixes():
        return ["relay"]

    def load(self, path, shape_dict=None, **kwargs):
        with open(path, "r", encoding="utf-8") as relay_text:
            text = relay_text.read()
        if shape_dict is None:
            logger.warning(
                "Specify --input-shapes to ensure that model inputs "
                "will not be considered as constants."
            )

        def _validate_text(text):
            """Check the provided file contents.
            The relay.txt artifact contained in the MLF is missing the version header and
            the metadata which is required to use meta[relay.Constant]."""

            if re.compile(r".*\#\[version\.*").match(text) is None:
                raise TVMCException(
                    "The relay model does not include the required version information."
                )
            if re.compile(r".*meta\[.+\].*", re.DOTALL).match(text):
                if "#[metadata]" not in text:
                    raise TVMCException(
                        "The relay model does not include the required #[metadata] section. "
                        "Use ir_mod.astext(show_meta_data=True) to export compatible code."
                    )

        _validate_text(text)

        ir_mod = parser.fromtext(text)

        if shape_dict:
            input_names = shape_dict.keys()
        else:
            input_names = []

        def _gen_params(ir_mod, skip_names=None):
            """Populate the all the params in the mode with ones."""
            main_func = ir_mod["main"]
            shape_dict = {p.name_hint: p.checked_type.concrete_shape for p in main_func.params}
            type_dict = {p.name_hint: p.checked_type.dtype for p in main_func.params}
            params = {}
            for name, shape in shape_dict.items():
                if skip_names and name in skip_names:
                    continue

                if "int" in type_dict[name]:
                    data = np.random.randint(128, size=shape, dtype=type_dict[name])
                else:
                    data = np.random.uniform(-1, 1, size=shape).astype(type_dict[name])
                params[name] = data
            return params

        params = _gen_params(ir_mod, skip_names=input_names)

        return ir_mod, params


ALL_FRONTENDS = [
    KerasFrontend,
    OnnxFrontend,
    TensorflowFrontend,
    TFLiteFrontend,
    PyTorchFrontend,
    PaddleFrontend,
    RelayFrontend,
]


def get_frontend_names():
    """Return the names of all supported frontends

    Returns
    -------
    list : list of str
        A list of frontend names as strings

    """
    return [frontend.name() for frontend in ALL_FRONTENDS]


def get_frontend_by_name(name: str):
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


def guess_frontend(path: str):
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


def load_model(
    path: str,
    model_format: Optional[str] = None,
    shape_dict: Optional[Dict[str, List[int]]] = None,
    **kwargs,
):
    """Load a model from a supported framework and convert it
    into an equivalent relay representation.

    Parameters
    ----------
    path : str
        The path to the model file.
    model_format : str, optional
        The underlying framework used to create the model.
        If not specified, this will be inferred from the file type.
    shape_dict : dict, optional
        Mapping from input names to their shapes.

    Returns
    -------
    tvmc_model : TVMCModel
        The produced model package.

    """

    if model_format is not None:
        frontend = get_frontend_by_name(model_format)
    else:
        frontend = guess_frontend(path)

    mod, params = frontend.load(path, shape_dict, **kwargs)

    return TVMCModel(mod, params)
