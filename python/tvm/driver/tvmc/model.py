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
import json
from typing import Optional, Union, Dict, Callable, TextIO
import numpy as np

import tvm
import tvm.contrib.cc
from tvm import relay
from tvm.contrib import utils
from tvm.relay.backend.executor_factory import GraphExecutorFactoryModule
from tvm.runtime.module import BenchmarkResult

try:
    from tvm.micro import export_model_library_format
except ImportError:
    export_model_library_format = None

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

    def export_classic_format(
        self,
        executor_factory: GraphExecutorFactoryModule,
        package_path: Optional[str] = None,
        cross: Optional[Union[str, Callable]] = None,
        cross_options: Optional[str] = None,
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
        cross_options : str, optional
            Command line options to be passed to the cross compiler.
        lib_format : str
            How to export the modules function library. Must be one of "so" or "tar".

        Returns
        -------
        package_path : str
            The path that the package was saved to.
        """
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
            if not cross_options:
                executor_factory.get_lib().export_library(
                    path_lib, tvm.contrib.cc.cross_compiler(cross)
                )
            else:
                executor_factory.get_lib().export_library(
                    path_lib, tvm.contrib.cc.cross_compiler(cross, options=cross_options.split(" "))
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

    def export_package(
        self,
        executor_factory: GraphExecutorFactoryModule,
        package_path: Optional[str] = None,
        cross: Optional[Union[str, Callable]] = None,
        cross_options: Optional[str] = None,
        output_format: str = "so",
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
        cross_options : str, optional
            Command line options to be passed to the cross compiler.
        output_format : str
            How to save the modules function library. Must be one of "so" and "tar" to save
            using the classic format or "mlf" to save using the Model Library Format.

        Returns
        -------
        package_path : str
            The path that the package was saved to.
        """
        if output_format not in ["so", "tar", "mlf"]:
            raise TVMCException("Only 'so', 'tar', and 'mlf' output formats are supported.")

        if output_format == "mlf" and cross:
            raise TVMCException("Specifying the MLF output and a cross compiler is not supported.")

        if output_format in ["so", "tar"]:
            package_path = self.export_classic_format(
                executor_factory, package_path, cross, cross_options, output_format
            )
        elif output_format == "mlf":
            if export_model_library_format:
                package_path = export_model_library_format(executor_factory, package_path)
            else:
                raise Exception("micro tvm is not enabled. Set USE_MICRO to ON in config.cmake")

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
        temp = self._tmp_dir
        t = tarfile.open(package_path)
        t.extractall(temp.relpath("."))

        if os.path.exists(temp.relpath("metadata.json")):
            # Model Library Format (MLF)
            self.lib_name = None
            self.lib_path = None
            with open(temp.relpath("metadata.json")) as metadata_json:
                metadata = json.load(metadata_json)

            has_graph_executor = "graph" in metadata["executors"]
            graph = temp.relpath("executor-config/graph/graph.json") if has_graph_executor else None
            params = temp.relpath("parameters/default.params")

            self.type = "mlf"
        else:
            # Classic format
            lib_name_so = "mod.so"
            lib_name_tar = "mod.tar"
            if os.path.exists(temp.relpath(lib_name_so)):
                self.lib_name = lib_name_so
            elif os.path.exists(temp.relpath(lib_name_tar)):
                self.lib_name = lib_name_tar
            else:
                raise TVMCException("Couldn't find exported library in the package.")
            self.lib_path = temp.relpath(self.lib_name)

            graph = temp.relpath("mod.json")
            params = temp.relpath("mod.params")

            self.type = "classic"

        with open(params, "rb") as param_file:
            self.params = bytearray(param_file.read())

        if graph is not None:
            with open(graph) as graph_file:
                self.graph = graph_file.read()
        else:
            self.graph = None


class TVMCResult(object):
    """A class that stores the results of tvmc.run and provides helper utilities."""

    def __init__(self, outputs: Dict[str, np.ndarray], times: BenchmarkResult):
        """Create a convenience wrapper around the output of tvmc.run

        Parameters
        ----------
        outputs : dict
            Outputs dictionary mapping the name of the output to its numpy value.
        times : BenchmarkResult
            The execution times measured by the time evaluator in seconds to produce outputs.
        """
        self.outputs = outputs
        self.times = times

    def format_times(self):
        """Format the mean, max, min and std of the execution times.

        This has the effect of producing a small table that looks like:
        .. code-block::
            Execution time summary:
            mean (ms)  median (ms) max (ms)    min (ms)    std (ms)
            0.14310      0.14310   0.16161     0.12933    0.01004

        Returns
        -------
        str
            A formatted string containing the statistics.
        """
        return str(self.times)

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
