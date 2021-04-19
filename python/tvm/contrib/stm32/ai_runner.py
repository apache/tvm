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
AI runner
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from abc import ABC, abstractmethod
import time as t
import logging
from enum import Enum
import numpy as np


class AiRunnerError(Exception):
    """Base exceptions for errors raised by AIRunner"""

    error = 800
    idx = 0

    def __init__(self, mess=None):
        self.mess = mess
        super(AiRunnerError, self).__init__(mess)

    def code(self):
        return self.error + self.idx

    def __str__(self):
        _mess = ""
        if self.mess is not None:
            _mess = "{}".format(self.mess)
        else:
            _mess = type(self).__doc__.split("\n")[0]
        _msg = "E{}({}): {}".format(self.code(), type(self).__name__, _mess)
        return _msg


class HwIOError(AiRunnerError):
    """Low-level IO error"""

    idx = 1


class NotInitializedMsgError(AiRunnerError):
    """Message is not fully initialized"""

    idx = 2


class InvalidMsgError(AiRunnerError, ValueError):
    """Message is not correctly formatted"""

    idx = 3


class InvalidParamError(AiRunnerError):
    """Invali parameter"""

    idx = 4


class NotConnectedError(AiRunnerError):
    """STM AI run-time is not connected"""

    idx = 10


def get_logger(name, debug=False, verbosity=0):
    """get_logger"""
    if name is None:
        # without configuring dummy before.
        return logging.getLogger("_DummyLogger_")

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        # logger 'name' already created
        return logger

    if debug:
        _lvl = logging.DEBUG
    elif verbosity:
        _lvl = logging.INFO
    else:
        _lvl = logging.WARNING
    logger.setLevel(_lvl)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(_lvl)
    # formatter = logging.Formatter('%(name)-12s:%(levelname)-7s: %(message)s')
    formatter = logging.Formatter("%(name)s:%(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False

    return logger


def generate_rnd(types, shapes, batch_size=4, rng=np.random.RandomState(42)):
    """Generate list of random arrays"""
    no_list = False
    if not isinstance(types, list) and not isinstance(shapes, list):
        types = [types]
        shapes = [shapes]
        no_list = True
    batch_size = max(1, batch_size)
    inputs = []
    for type_, shape_ in zip(types, shapes):
        shape_ = (batch_size,) + shape_[1:]
        if type_ == np.bool:
            in_ = rng.randint(2, size=shape_).astype(np.bool)
        elif type_ != np.float32:
            high = np.iinfo(type_).max
            low = np.iinfo(type_).min
            #  “discrete uniform” distribution
            in_ = rng.randint(low=low, high=high + 1, size=shape_)
        else:
            # uniformly distributed over the half-open interval [low, high)
            in_ = rng.uniform(low=-1.0, high=1.0, size=shape_)

        in_ = np.ascontiguousarray(in_.astype(type_))
        inputs.append(in_)
    return inputs[0] if no_list else inputs


class AiHwDriver(ABC):
    """Base class to handle the LL IO COM functions"""

    def __init__(self, parent=None):
        self._parent = parent
        self._hdl = None
        # return

    @property
    def is_connected(self):
        # return True if self._hdl else False
        return self._hdl

    def set_parent(self, parent):
        self._parent = parent

    def get_config(self):
        return dict()

    @abstractmethod
    def _connect(self, desc=None, **kwargs):
        pass

    @abstractmethod
    def _disconnect(self):
        pass

    @abstractmethod
    def _read(self, size, timeout=0):
        return 0

    @abstractmethod
    def _write(self, data, timeout=0):
        return 0

    def connect(self, desc=None, **kwargs):
        self.disconnect()
        return self._connect(desc=desc, kwargs=kwargs)

    def disconnect(self):
        if self.is_connected:
            self._disconnect()

    def read(self, size, timeout=0):
        if self.is_connected:
            return self._read(size, timeout)
        raise NotConnectedError()

    def write(self, data, timeout=0):
        if self.is_connected:
            return self._write(data, timeout)
        raise NotConnectedError()

    def short_desc(self):
        return "UNDEFINED"


class AiRunnerDriver(ABC):
    """Base class interface for an AI Runner driver"""

    def __init__(self, parent):
        if not hasattr(parent, "get_logger"):
            raise InvalidParamError("Invalid parent type, get_logger() attr is expected")
        self._parent = parent
        self._logger = parent.get_logger()
        self._logger.debug("creating {} object".format(self.__class__.__name__))
        # return

    @abstractmethod
    def connect(self, desc=None, **kwargs):
        """Connect to the stm.ai run-time"""
        return False

    @property
    def is_connected(self):
        """Indicate if the diver is connected"""
        return False

    @abstractmethod
    def disconnect(self):
        """Disconnect to the stm.ai run-time"""
        # pass

    @abstractmethod
    def discover(self, flush=False):
        """Return list of available networks"""
        return []

    @abstractmethod
    def get_info(self, name=None):
        """Get c-network details (including runtime)"""
        return dict()

    @abstractmethod
    def invoke_sample(self, inputs, **kwargs):
        """Invoke the c-network with a given input (sample mode)"""
        return [], dict()

    def check_inputs(self, inputs, name):  # pylint: disable=unused-argument
        """Specific function to check the inputs"""
        return False


class AiRunnerSession:
    """
    Interface to use a model
    """

    def __init__(self, name):
        """Constructor"""
        self._parent = None
        self._name = name
        # return

    def __str__(self):
        return self.name

    @property
    def is_active(self):
        # return True if self._parent else False
        return self._parent

    @property
    def name(self):
        """Return the name of the model"""
        return self._name

    def _acquire(self, parent):
        self._parent = parent

    def _release(self):
        """Release the resources"""
        self._parent = None

    def get_input_infos(self):
        """Get model input details"""
        if self._parent:
            return self._parent.get_input_infos(self.name)
        return list()

    def get_output_infos(self):
        """Get model outputs details"""
        if self._parent:
            return self._parent.get_output_infos(self.name)
        return list()

    def get_info(self):
        """Get model details (including runtime)"""
        if self._parent:
            return self._parent.get_info(self.name)
        return dict()

    def invoke(self, inputs, **kwargs):
        """Invoke the c-network"""
        if self._parent:
            kwargs.pop("name", None)
            return self._parent.invoke(inputs, name=self.name, **kwargs)
        return list(), dict()

    def summary(self, print_fn=None):
        """Summary model & runtime infos"""
        if self._parent:
            return self._parent.summary(name=self.name, print_fn=print_fn)
        return None


class AiRunnerCallback:
    """
    Abstract base class used to build new callbacks
    """

    def __init__(self):
        pass

    def on_sample_begin(self, idx):
        """Called at the beginning of each sample

        Arguments:
            idx: Integer, index of the sample

        """

    def on_sample_end(self, idx, data, logs=None):
        """Called at the end of each sample

        Arguments:
            idx: Integer, index of the sample
            data: List, output tensors

        """

    def on_node_begin(self, idx, data, logs=None):
        """Called before each c-node

        Arguments:
            idx: Integer, index of the c-node
            data: List, input tensors

        """

    def on_node_end(self, idx, data, logs=None):
        """Called at the end of each c-node

        Arguments:
            idx: Integer, index of the c-node
            data: List, output tensors

        """


class AiRunner:
    """
    AI Runner interface for stm.ai runtime.
    """

    class Caps(Enum):
        IO_ONLY = 0
        PER_LAYER = 1
        PER_LAYER_WITH_DATA = PER_LAYER | 2
        SELF_TEST = 4

    class Mode(Enum):
        IO_ONLY = 0
        PER_LAYER = 1
        PER_LAYER_WITH_DATA = PER_LAYER | 2

    def __init__(self, logger=None, debug=False, verbosity=0):
        """Constructor"""
        self._sessions = []
        self._names = []
        self._drv = None
        if logger is None:
            logger = get_logger(self.__class__.__name__, debug, verbosity)
        self._logger = logger
        self._logger.debug(
            #'creating {} object'.format(self.__class__.__name__))
            "creating %s object",
            self.__class__.__name__,
        )

    def get_logger(self):
        return self._logger

    def __str__(self):
        return self.short_desc()

    @property
    def is_connected(self):
        """Indicate if the associated runtime is connected"""
        return False if not self._drv else self._drv.is_connected

    def _check_name(self, name):
        """Return a valid c-network name"""
        if not self._names:
            return None
        if isinstance(name, int):
            idx = max(0, min(name, len(self._names) - 1))
            return self._names[idx]
        if name is None or not isinstance(name, str):
            return self._names[0]
        if name in self._names:
            return name
        return None

    def get_info(self, name=None):
        """Get model details (including runtime infos)"""
        name_ = self._check_name(name)
        return self._drv.get_info(name_) if name_ else dict()

    @property
    def name(self):
        """Return default network name (first)"""
        return self._check_name(None)

    def get_input_infos(self, name=None):
        """Get model input details"""
        info_ = self.get_info(name)
        return info_["inputs"] if info_ else list()

    def get_output_infos(self, name=None):
        """Get model output details"""
        info_ = self.get_info(name)
        return info_["outputs"] if info_ else list()

    def _align_requested_mode(self, mode):
        """Align requested mode with drv capabilities"""
        if mode not in AiRunner.Mode:
            mode = AiRunner.Mode.IO_ONLY
        if mode == AiRunner.Mode.PER_LAYER_WITH_DATA:
            if AiRunner.Caps.PER_LAYER_WITH_DATA not in self._drv.capabilities:
                mode = AiRunner.Mode.PER_LAYER
        if mode == AiRunner.Mode.PER_LAYER:
            if AiRunner.Caps.PER_LAYER not in self._drv.capabilities:
                mode = AiRunner.Mode.IO_ONLY
        return mode

    def _check_inputs(self, inputs, name):
        """Check the coherence of the inputs (data type and shape)"""

        if self._drv.check_inputs(inputs, name):
            return

        in_desc = self.get_input_infos(name)

        if len(inputs) != len(in_desc):
            msg = "Input number is inconsistent {} instead {}".format(len(inputs), len(in_desc))
            raise HwIOError(msg)

        for idx, ref in enumerate(in_desc):
            in_shape = (ref["shape"][0],) + inputs[idx].shape[1:]
            if inputs[idx].dtype != ref["type"]:
                msg = "invalid dtype - {} instead {}".format(inputs[idx].dtype, ref["type"])
                msg += " for the input #{}".format(idx + 1)
                raise InvalidParamError(msg)
            if in_shape != ref["shape"]:
                msg = "invalid shape - {} instead {}".format(in_shape, ref["shape"])
                msg += " for the input #{}".format(idx + 1)
                raise InvalidParamError(msg)

    def invoke(self, inputs, **kwargs):
        """Invoke the c-network (batch mode)"""
        import tqdm  # pylint: disable=import-outside-toplevel

        name_ = self._check_name(kwargs.pop("name", None))

        if name_ is None:
            return list(), dict()

        if not isinstance(inputs, list):
            inputs = [inputs]

        self._check_inputs(inputs, name_)

        callback = kwargs.pop("callback", None)  # AiRunnerCallback())
        mode = self._align_requested_mode(kwargs.pop("mode", AiRunner.Mode.IO_ONLY))
        disable_pb = kwargs.pop("disable_pb", False) or callback

        batch_size = inputs[0].shape[0]
        profiler = {
            "info": dict(),
            "c_durations": [],  # Inference time by sample w/o cb by node if enabled
            "c_nodes": [],
            "debug": {
                "exec_times": [],  # real inference time by sample with cb by node overhead
                "host_duration": 0.0,  # host execution time (on whole batch)
            },
        }

        start_time = t.perf_counter()
        outputs = []
        pb_ = None
        for batch in range(batch_size):
            if not pb_ and not disable_pb and (t.perf_counter() - start_time) > 1:
                pb_ = tqdm.tqdm(
                    total=batch_size,
                    file=sys.stdout,
                    unit_scale=False,  # desc='Running..',
                    leave=False,
                )
                pb_.update(batch)
            elif pb_:
                pb_.update(1)
            s_inputs = [np.expand_dims(in_[batch], axis=0) for in_ in inputs]
            if callback:
                callback.on_sample_begin(batch)
            s_outputs, s_dur = self._drv.invoke_sample(
                s_inputs, name=name_, profiler=profiler, mode=mode, callback=callback
            )
            if batch == 0:
                outputs = s_outputs
            else:
                for idx, out_ in enumerate(s_outputs):
                    outputs[idx] = np.append(outputs[idx], out_, axis=0)
            if callback:
                callback.on_sample_end(batch, s_outputs, logs={"dur": s_dur})
        profiler["debug"]["host_duration"] = (t.perf_counter() - start_time) * 1000.0
        profiler["info"] = self.get_info(name_)
        if pb_:
            pb_.close()

        return outputs, profiler

    def generate_rnd_inputs(self, name=None, batch_size=4, rng=np.random.RandomState(42)):
        """Generate input data with random values"""
        if isinstance(name, AiRunnerSession):
            name = name.name
        name_ = self._check_name(name)
        if name_ is None:
            return []
        info_ = self.get_input_infos(name_)
        return generate_rnd(
            [t_["type"] for t_ in info_], [s_["shape"] for s_ in info_], batch_size, rng=rng
        )

    @property
    def names(self):
        """Return available nets as a list of name"""
        return self._names

    def _release(self):
        """Release all resources"""
        if self._names:
            self._logger.debug("_release(%s)", str(self))
        for net_ in self._sessions:
            net_._release()
            self._sessions.remove(net_)
        self._names = []
        if self.is_connected:
            self._drv.disconnect()
            self._drv = None
        return True

    def short_desc(self):
        if self.is_connected:
            desc_ = "{} {}".format(self._drv.short_desc(), self.names)
            return desc_
        return "not connected"

    def connect(self, desc=None, **kwargs):
        """Connect to a given runtime defined by desc"""
        from .ai_resolver import ai_runner_resolver  # pylint: disable=import-outside-toplevel

        self._logger.debug("connect(desc='%s')", str(desc))
        self._release()

        self._drv, desc = ai_runner_resolver(self, desc)
        self._logger.debug("desc for the driver: '%s'", str(desc))

        if self._drv is None:
            self._logger.error(desc)
            return False

        try:
            self._drv.connect(desc, **kwargs)
            if not self._drv.is_connected:
                self._release()
            else:
                self._names = self._drv.discover(flush=True)
                for name_ in self._names:
                    self._sessions.append(AiRunnerSession(name_))
        except Exception as exc_:  # pylint: disable=broad-except
            msg_ = 'binding with runtime "{}" has failed\n {}'.format(desc, str(exc_))
            self._logger.error(msg_)
            self._release()

        return self.is_connected

    def session(self, name=None):
        """Return session handler for the given model name/idx"""
        if not self.is_connected:
            return None
        name = self._check_name(name)
        if name:
            for ses_ in self._sessions:
                if name == ses_.name:
                    ses_._acquire(self)
                    return ses_
        return None

    def disconnect(self):
        return self._release()

    def summary(self, name=None, print_fn=None, level=0):
        """Prints a summary of the model & associated runtime"""

        if print_fn is None:
            print_fn = print

        dict_info = self.get_info(name)
        if dict_info:

            def _attr(attr, val):
                print_fn("{:20s} : {}".format(str(attr), str(val)))

            def _tens_to_str(val):
                ext_ = ""
                if val["scale"]:
                    ext_ = ", scale={}, zp={}".format(val["scale"], val["zero_point"])
                if val.get("from_act", False):
                    ext_ += ", (allocated in activations buffer)"
                _attr(val["name"], "{}, {}{}".format(val["shape"], val["type"], ext_))

            print_fn('\nSummary "{}" - {}'.format(dict_info["name"], self._names))
            print_fn("-" * 80)
            _attr(
                "inputs/outputs",
                "{}/{}".format(len(dict_info["inputs"]), len(dict_info["outputs"])),
            )
            for in_ in dict_info["inputs"]:
                _tens_to_str(in_)
            for out_ in dict_info["outputs"]:
                _tens_to_str(out_)
            _attr("n_nodes", dict_info["n_nodes"])
            _attr("compile_datetime", dict_info.get("compile_datetime", "undefined"))
            _attr("model_datetime", dict_info.get("model_datetime", "undefined"))
            if level:
                _attr("activations", dict_info["activations"])
                _attr("weights", dict_info["weights"])
                _attr("macc", dict_info["macc"] if dict_info["macc"] > 1 else "n.a.")
            print_fn("-" * 80)
            _attr(
                "runtime",
                "{} {}.{}.{}".format(
                    dict_info["runtime"]["name"], *dict_info["runtime"]["version"]
                ),
            )
            _attr("capabilities", [str(n) for n in dict_info["runtime"]["capabilities"]])
            for key, value in dict_info["device"].items():
                _attr(key, value)
            print_fn("-" * 80)
            print_fn("")


if __name__ == "__main__":
    pass
