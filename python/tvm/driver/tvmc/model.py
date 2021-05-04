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
This file contains the definition of a set of classes that wrap the outputs
of TVMC functions to create a simpler and more intuitive API.

There is one class for each required stage of a TVM workflow.
The TVMCModel represents the result of importing a model into TVM, it
contains the precompiled graph definition and parameters that define
what the model does.

Compiling a TVMCModel produces a TVMCPackage, which contains the generated
artifacts that allow the model to be run on the target hardware.

Running a TVMCPackage produces a TVMCResult, which contains the outputs of
the model and the measured runtime.

Examples
--------
The following code shows a full lifecycle for a model using tvmc, first the
model is imported from an exterior framework, in this case onnx, then it
is tuned to find the best schedules on CPU, then compiled into a TVMCPackage,
and finally run.

.. code-block:: python
    tvmc_model = tvmc.load("my_model.onnx")
    tuning_records = tvmc.tune(tvmc_model, target="llvm")
    tvmc_package = tvmc.compile(tvmc_model, target="llvm", tuning_records=tuning_records)
    result = tvmc.run(tvmc_package, device="cpu")
    print(result)
"""
import os
import tarfile
from typing import Optional, Union, List, Dict, Callable, TextIO
import numpy as np

import tvm
import tvm.contrib.cc
from tvm import relay
from tvm.contrib import utils
from tvm.relay.backend.executor_factory import GraphExecutorFactoryModule

from .common import TVMCException


class TVMCModel(object):
    """Initialize a TVMC model from a relay model definition or a saved file.

    Parameters
    ----------
    mod : tvm.IRModule, optional
        The relay module corresponding to this model.
    params : dict, optional
        A parameter dictionary for the model.
    model_path: str, optional
        An alternative way to load a TVMCModel, the path to a previously
        saved model.
    """

    def __init__(
        self,
        mod: Optional[tvm.IRModule] = None,
        params: Optional[Dict[str, tvm.nd.NDArray]] = None,
        model_path: Optional[str] = None,
    ):
        if (mod is None or params is None) and (model_path is None):
            raise TVMCException(
                "Either mod and params must be provided "
                "or a path to a previously saved TVMCModel"
            )
        self._tmp_dir = utils.tempdir()
        if model_path is not None:
            self.load(model_path)
        else:
            self.mod = mod
            self.params = params if params else {}

    def save(self, model_path: str):
        """Save the TVMCModel to disk.

        Note that this saves the graph representation,
        the parameters, and the tuning records if applicable. It will not save any
        compiled artifacts.

        Parameters
        ----------
        model_path : str
            A full path to save this TVMCModel to including the output file name.
            The file will be saved as a tar file so using a ".tar" extension is advised.
        """
        temp = self._tmp_dir

        # Save relay graph
        relay_name = "model.json"
        relay_path = temp.relpath(relay_name)
        with open(relay_path, "w") as relay_file:
            relay_file.write(tvm.ir.save_json(self.mod))

        # Save params
        params_name = "model.params"
        params_path = temp.relpath(params_name)
        with open(params_path, "wb") as params_file:
            params_file.write(relay.save_param_dict(self.params))

        # Create a tar file.
        with tarfile.open(model_path, "w") as tar:
            tar.add(relay_path, relay_name)
            tar.add(params_path, params_name)
            # If default tuning records exist, save them as well.
            if os.path.exists(self.default_tuning_records_path()):
                tar.add(self.default_tuning_records_path(), "tuning_records")
            # Also save the compiled package if it can be found.
            if os.path.exists(self.default_package_path()):
                tar.add(self.default_package_path(), "model_package.tar")

    def load(self, model_path: str):
        """Load a TVMCModel from disk.

        Parameters
        ----------
        model_path : str
            A path to load the TVMCModel from.
        """
        temp = self._tmp_dir
        t = tarfile.open(model_path)
        t.extractall(temp.relpath("."))

        # Load relay IR.
        relay_path = temp.relpath("model.json")
        with open(relay_path, "r") as relay_file:
            self.mod = tvm.ir.load_json(relay_file.read())

        # Load parameter dictionary.
        params_path = temp.relpath("model.params")
        with open(params_path, "rb") as params_file:
            self.params = relay.load_param_dict(params_file.read())

    def default_tuning_records_path(self):
        """Get a full path for storing tuning records in this model's temporary direcotry

        Note that when this path is used, the tuning records will be saved and loaded
        when calling `save` and `load`.

        Returns
        -------
        records_path: str
            A path to the default location for tuning records.
        """
        return self._tmp_dir.relpath("tuning_records")

    def default_package_path(self):
        """Get a full path for storing a compiled package in this model's temporary direcotry

        Note that when this path is used, the package will be saved and loaded
        when calling `save` and `load`.

        Returns
        -------
        records_path: str
            A path to the default location for tuning records.
        """
        return self._tmp_dir.relpath("model_package.tar")

    def export_package(
        self,
        executor_factory: GraphExecutorFactoryModule,
        package_path: Optional[str] = None,
        cross: Optional[Union[str, Callable]] = None,
        lib_format: str = "so",
    ):
        """Save this TVMCModel to file.
        Parameters
        ----------
        executor_factory : GraphExecutorFactoryModule
            The factory containing compiled the compiled artifacts needed to run this model.
        package_path : str, None
            Where the model should be saved. Note that it will be packaged as a .tar file.
            If not provided, the package will be saved to a generically named file in tmp.
        cross : str or callable object, optional
            Function that performs the actual compilation.
        lib_format : str
            How to export the modules function library. Must be one of "so" or "tar".

        Returns
        -------
        package_path : str
            The path that the package was saved to.
        """
        if lib_format not in ["so", "tar"]:
            raise TVMCException("Only .so and .tar export formats are supported.")
        lib_name = "mod." + lib_format
        graph_name = "mod.json"
        param_name = "mod.params"

        temp = self._tmp_dir
        if package_path is None:
            package_path = self.default_package_path()
        path_lib = temp.relpath(lib_name)

        if not cross:
            executor_factory.get_lib().export_library(path_lib)
        else:
            executor_factory.get_lib().export_library(
                path_lib, tvm.contrib.cc.cross_compiler(cross)
            )
        self.lib_path = path_lib

        with open(temp.relpath(graph_name), "w") as graph_file:
            graph_file.write(executor_factory.get_graph_json())

        with open(temp.relpath(param_name), "wb") as params_file:
            params_file.write(relay.save_param_dict(executor_factory.get_params()))

        # Package up all the temp files into a tar file.
        with tarfile.open(package_path, "w") as tar:
            tar.add(path_lib, lib_name)
            tar.add(temp.relpath(graph_name), graph_name)
            tar.add(temp.relpath(param_name), param_name)

        return package_path

    def summary(self, file: TextIO = None):
        """Print the IR corressponding to this model.

        Arguments
        ---------
        file: Writable, optional
            If specified, the summary will be written to this file.
        """
        print(self.mod, file=file)


class TVMCPackage(object):
    """Load a saved TVMCPackage from disk.

    Parameters
    ----------
    package_path : str
        The path to the saved TVMCPackage that will be loaded.
    """

    def __init__(self, package_path: str):
        self._tmp_dir = utils.tempdir()
        self.package_path = package_path
        self.import_package(self.package_path)

    def import_package(self, package_path: str):
        """Load a TVMCPackage from a previously exported TVMCModel.

        Parameters
        ----------
        package_path : str
            The path to the saved TVMCPackage.
        """
        lib_name_so = "mod.so"
        lib_name_tar = "mod.tar"
        graph_name = "mod.json"
        param_name = "mod.params"

        temp = self._tmp_dir
        t = tarfile.open(package_path)
        t.extractall(temp.relpath("."))

        with open(temp.relpath(param_name), "rb") as param_file:
            self.params = bytearray(param_file.read())
        self.graph = open(temp.relpath(graph_name)).read()
        if os.path.exists(temp.relpath(lib_name_so)):
            self.lib_name = lib_name_so
        elif os.path.exists(temp.relpath(lib_name_tar)):
            self.lib_name = lib_name_tar
        else:
            raise TVMCException("Couldn't find exported library in the package.")
        self.lib_path = temp.relpath(self.lib_name)


class TVMCResult(object):
    """A class that stores the results of tvmc.run and provides helper utilities."""

    def __init__(self, outputs: Dict[str, np.ndarray], times: List[str]):
        """Create a convenience wrapper around the output of tvmc.run

        Parameters
        ----------
        outputs : dict
            Outputs dictionary mapping the name of the output to its numpy value.
        times : list of float
            The execution times measured by the time evaluator in seconds to produce outputs.
        """
        self.outputs = outputs
        self.times = times

    def format_times(self):
        """Format the mean, max, min and std of the execution times.

        This has the effect of producing a small table that looks like:
        .. code-block::
            Execution time summary:
            mean (ms)   max (ms)    min (ms)    std (ms)
            0.14310    0.16161    0.12933    0.01004

        Returns
        -------
        str
            A formatted string containing the statistics.
        """

        # timestamps
        mean_ts = np.mean(self.times) * 1000
        std_ts = np.std(self.times) * 1000
        max_ts = np.max(self.times) * 1000
        min_ts = np.min(self.times) * 1000

        header = "Execution time summary:\n{0:^10} {1:^10} {2:^10} {3:^10}".format(
            "mean (ms)", "max (ms)", "min (ms)", "std (ms)"
        )
        stats = "{0:^10.2f} {1:^10.2f} {2:^10.2f} {3:^10.2f}".format(
            mean_ts, max_ts, min_ts, std_ts
        )

        return "%s\n%s\n" % (header, stats)

    def get_output(self, name: str):
        """A helper function to grab one of the outputs by name.

        Parameters
        ----------
        name : str
            The name of the output to return

        Returns
        -------
        output : np.ndarray
            The output corresponding to name.
        """
        return self.outputs[name]

    def save(self, output_path: str):
        """Save the numpy outputs to disk as a .npz file.

        Parameters
        ----------
        output_path : str
            The path to save the numpy results to.
        """
        np.savez(output_path, **self.outputs)

    def __str__(self):
        stat_table = self.format_times()
        output_keys = f"Output Names:\n {list(self.outputs.keys())}"
        return stat_table + "\n" + output_keys
