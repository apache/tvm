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

# pylint: disable=invalid-name, unused-import, import-outside-toplevel, inconsistent-return-statements
"""Runtime Module namespace."""
import os
import ctypes
import struct
from typing import Sequence
import numpy as np

import tvm._ffi
from tvm._ffi.base import _LIB, check_call, c_str, string_types, _RUNTIME_ONLY
from tvm._ffi.libinfo import find_include_path
from .packed_func import PackedFunc, PackedFuncHandle, _set_class_module

from . import _ffi_api


class BenchmarkResult:
    """Runtimes from benchmarking"""

    def __init__(self, results: Sequence[float]):
        """Construct a new BenchmarkResult from a sequence of runtimes.

        Parameters
        ----------
        results : Sequence[float]
            Raw times from benchmarking

        Attributes
        ----------
        min : float
            Minimum runtime in seconds of all results.
        mean : float
            Mean runtime in seconds of all results. If py:meth:`Module.time_evaluator` or
            `benchmark` is called with `number` > 0, then each result is already the mean of a
            `number` of runtimes, so this becomes the mean of means.
        median : float
            Median runtime in seconds of all results. If py:meth:`Module.time_evaluator` is called
            with `number` > 0, then each result is already the mean of a `number` of runtimes, so
            this becomes the median of means.
        max : float
            Maximum runtime in seconds of all results. If py:meth:`Module.time_evaluator` is called
            with `number` > 0, then each result is already the mean of a `number` of runtimes, so
            this becomes the maximum of those means.
        std : float
            Standard deviation in seconds of runtimes. If py:meth:`Module.time_evaluator` is called
            with `number` > 0, then each result is already the mean of a `number` of runtimes, so
            this becomes the standard deviation of means.
        results : Sequence[float]
            The collected runtimes (in seconds). This may be a series of mean runtimes if
            py:meth:`Module.time_evaluator` or `benchmark` was run with `number` > 1.
        """
        self.results = results
        self.mean = np.mean(self.results)
        self.std = np.std(self.results)
        self.median = np.median(self.results)
        self.min = np.min(self.results)
        self.max = np.max(self.results)

    def __repr__(self):
        return "BenchmarkResult(min={}, mean={}, median={}, max={}, std={}, results={})".format(
            self.min, self.mean, self.median, self.max, self.std, self.results
        )

    def __str__(self):
        return """Execution time summary:
{:^12} {:^12} {:^12} {:^12} {:^12}
{:^12.4f} {:^12.4f} {:^12.4f} {:^12.4f} {:^12.4f}
               """.format(
            "mean (ms)",
            "median (ms)",
            "max (ms)",
            "min (ms)",
            "std (ms)",
            self.mean * 1000,
            self.median * 1000,
            self.max * 1000,
            self.min * 1000,
            self.std * 1000,
        )


class Module(object):
    """Runtime Module."""

    __slots__ = ["handle", "_entry", "entry_name"]

    def __init__(self, handle):
        self.handle = handle
        self._entry = None
        self.entry_name = "__tvm_main__"

    def __del__(self):
        if _LIB:
            check_call(_LIB.TVMModFree(self.handle))

    def __hash__(self):
        return ctypes.cast(self.handle, ctypes.c_void_p).value

    @property
    def entry_func(self):
        """Get the entry function

        Returns
        -------
        f : tvm.runtime.PackedFunc
            The entry function if exist
        """
        if self._entry:
            return self._entry
        self._entry = self.get_function(self.entry_name)
        return self._entry

    def implements_function(self, name, query_imports=False):
        """Returns True if the module has a definition for the global function with name. Note
        that has_function(name) does not imply get_function(name) is non-null since the module
        may be, eg, a CSourceModule which cannot supply a packed-func implementation of the function
        without further compilation. However, get_function(name) non null should always imply
        has_function(name).

        Parameters
        ----------
        name : str
            The name of the function

        query_imports : bool
            Whether to also query modules imported by this module.

        Returns
        -------
        b : Bool
            True if module (or one of its imports) has a definition for name.
        """
        return _ffi_api.ModuleImplementsFunction(self, name, query_imports)

    def get_function(self, name, query_imports=False):
        """Get function from the module.

        Parameters
        ----------
        name : str
            The name of the function

        query_imports : bool
            Whether also query modules imported by this module.

        Returns
        -------
        f : tvm.runtime.PackedFunc
            The result function.
        """
        ret_handle = PackedFuncHandle()
        check_call(
            _LIB.TVMModGetFunction(
                self.handle, c_str(name), ctypes.c_int(query_imports), ctypes.byref(ret_handle)
            )
        )
        if not ret_handle.value:
            raise AttributeError("Module has no function '%s'" % name)
        return PackedFunc(ret_handle, False)

    def import_module(self, module):
        """Add module to the import list of current one.

        Parameters
        ----------
        module : tvm.runtime.Module
            The other module.
        """
        check_call(_LIB.TVMModImport(self.handle, module.handle))

    def __getitem__(self, name):
        if not isinstance(name, string_types):
            raise ValueError("Can only take string as function name")
        return self.get_function(name)

    def __eq__(self, other):
        return self.handle.value == other.handle.value

    def __call__(self, *args):
        if self._entry:
            return self._entry(*args)
        # pylint: disable=not-callable
        return self.entry_func(*args)

    def __repr__(self):
        return "Module(%s, %x)" % (self.type_key, self.handle.value)

    @property
    def type_key(self):
        """Get type key of the module."""
        return _ffi_api.ModuleGetTypeKey(self)

    @property
    def format(self):
        """Get the format of the module."""
        return _ffi_api.ModuleGetFormat(self)

    def get_source(self, fmt=""):
        """Get source code from module, if available.

        Parameters
        ----------
        fmt : str, optional
            The specified format.

        Returns
        -------
        source : str
            The result source code.
        """
        return _ffi_api.ModuleGetSource(self, fmt)

    @property
    def imported_modules(self):
        """Get imported modules

        Returns
        ----------
        modules : list of Module
            The module
        """
        nmod = _ffi_api.ModuleImportsSize(self)
        return [_ffi_api.ModuleGetImport(self, i) for i in range(nmod)]

    @property
    def is_dso_exportable(self):
        """Returns true if module is 'DSO exportable', ie can be included in result of
        export_library by the external compiler directly.

        Returns
        -------
        b : Bool
            True if the module is DSO exportable.
        """
        return _ffi_api.ModuleIsDSOExportable(self)

    def save(self, file_name, fmt=""):
        """Save the module to file.

        This do not save the dependent device modules.
        See also export_shared

        Parameters
        ----------
        file_name : str
            The name of the file.
        fmt : str
            The format of the file.

        See Also
        --------
        runtime.Module.export_library : export the module to shared library.
        """
        _ffi_api.ModuleSaveToFile(self, file_name, fmt)

    def time_evaluator(
        self,
        func_name,
        dev,
        number=10,
        repeat=1,
        min_repeat_ms=0,
        limit_zero_time_iterations=100,
        cooldown_interval_ms=0,
        repeats_to_cooldown=1,
        f_preproc="",
    ):
        """Get an evaluator that measures time cost of running function.

        Parameters
        ----------
        func_name: str
            The name of the function in the module.

        dev: Device
            The device we should run this function on.

        number: int
            The number of times to run this function for taking average.
            We call these runs as one `repeat` of measurement.

        repeat: int, optional
            The number of times to repeat the measurement.
            In total, the function will be invoked (1 + number x repeat) times,
            where the first one is warm up and will be discarded.
            The returned result contains `repeat` costs,
            each of which is an average of `number` costs.

        min_repeat_ms: int, optional
            The minimum duration of one `repeat` in milliseconds.
            By default, one `repeat` contains `number` runs. If this parameter is set,
            the parameters `number` will be dynamically adjusted to meet the
            minimum duration requirement of one `repeat`.
            i.e., When the run time of one `repeat` falls below this time, the `number` parameter
            will be automatically increased.

        limit_zero_time_iterations: int, optional
            The maximum number of repeats when measured time is equal to 0.
            It helps to avoid hanging during measurements.

        cooldown_interval_ms: int, optional
            The cooldown interval in milliseconds between the number of repeats defined by
            `repeats_to_cooldown`.

        repeats_to_cooldown: int, optional
            The number of repeats before the cooldown is activated.

        f_preproc: str, optional
            The preprocess function name we want to execute before executing the time evaluator.

        Note
        ----
        The function will be invoked  (1 + number x repeat) times,
        with the first call discarded in case there is lazy initialization.

        Returns
        -------
        ftimer : function
            The function that takes same argument as func and returns a BenchmarkResult.
            The ProfileResult reports `repeat` time costs in seconds.
        """
        try:
            feval = _ffi_api.RPCTimeEvaluator(
                self,
                func_name,
                dev.device_type,
                dev.device_id,
                number,
                repeat,
                min_repeat_ms,
                limit_zero_time_iterations,
                cooldown_interval_ms,
                repeats_to_cooldown,
                f_preproc,
            )

            def evaluator(*args):
                """Internal wrapped evaluator."""
                # Wrap feval so we can add more stats in future.
                blob = feval(*args)
                fmt = "@" + ("d" * repeat)
                results = struct.unpack(fmt, blob)
                return BenchmarkResult(results)

            return evaluator
        except NameError:
            raise NameError("time_evaluator is only supported when RPC is enabled")

    def _collect_from_import_tree(self, filter_func):
        """Helper function to collect modules from the tree matching a filter_func, then return it.

        Parameters
        ----------
        filter_func : Callable[[Module], bool]
            A function which is invoked for each Module discovered in the import tree (including
            self).

        Returns
        -------
        list[Module] :
            A list of matching Module.
        """
        visited, stack, dso_modules = set(), [], []
        # append root module
        visited.add(self)
        stack.append(self)
        while stack:
            module = stack.pop()
            if filter_func(module):
                dso_modules.append(module)
            for m in module.imported_modules:
                if m not in visited:
                    visited.add(m)
                    stack.append(m)
        return dso_modules

    def _collect_dso_modules(self):
        return self._collect_from_import_tree(lambda m: m.is_dso_exportable)

    def export_library(self, file_name, fcompile=None, addons=None, workspace_dir=None, **kwargs):
        """
        Export the module and all imported modules into a single device library.

        This function only works on host LLVM modules, other runtime::Module
        subclasses will work with this API but they must support implement
        the save and load mechanisms of modules completely including saving
        from streams and files. This will pack your non-shared library module
        into a single shared library which can later be loaded by TVM.

        Parameters
        ----------
        file_name : str
            The name of the shared library.

        fcompile : function(target, file_list, kwargs), optional
            The compilation function to use create the final library object during
            export.

            For example, when fcompile=_cc.create_shared, or when it is not supplied but
            module is "llvm," this is used to link all produced artifacts
            into a final dynamic library.

            This behavior is controlled by the type of object exported.
            If fcompile has attribute object_format, will compile host library
            to that format. Otherwise, will use default format "o".

        workspace_dir : str, optional
            The path of the directory used to create the intermediate
            artifacts when exporting the module.
            If this is not provided a temporary dir will be created.

        kwargs : dict, optional
            Additional arguments passed to fcompile

        Returns
        -------
        result of fcompile()  : unknown, optional
            If the compilation function returns an artifact it would be returned via
            export_library, if any.
        """
        # NOTE: this function depends on contrib library features
        # which are only available in when TVM function is available.
        if _RUNTIME_ONLY:
            raise RuntimeError("Cannot call export_library in runtime only mode")
        # Extra dependencies during runtime.
        from pathlib import Path
        from tvm.contrib import cc as _cc, tar as _tar, utils as _utils

        if isinstance(file_name, Path):
            file_name = str(file_name)

        if self.type_key == "stackvm":
            if not file_name.endswith(".stackvm"):
                raise ValueError(
                    "Module[%s]: can only be saved as stackvm format."
                    "did you build with LLVM enabled?" % self.type_key
                )
            self.save(file_name)
            return

        modules = self._collect_dso_modules()
        if workspace_dir is None:
            temp = _utils.tempdir()
            workspace_dir = temp.temp_dir
        files = addons if addons else []
        is_system_lib = False
        has_c_module = False
        llvm_target_string = None
        for index, module in enumerate(modules):
            if fcompile is not None and hasattr(fcompile, "object_format"):
                if module.type_key == "c":
                    assert module.format in [
                        "c",
                        "cc",
                        "cpp",
                        "cu",
                    ], "The module.format needs to be either c, cc, cpp or cu."
                    object_format = module.format
                    has_c_module = True
                else:
                    object_format = fcompile.object_format
            else:
                if module.type_key == "c":
                    if len(module.format) > 0:
                        assert module.format in [
                            "c",
                            "cc",
                            "cpp",
                            "cu",
                        ], "The module.format needs to be either c, cc, cpp, or cu."
                        object_format = module.format
                    else:
                        object_format = "c"
                    if "cc" in kwargs:
                        if kwargs["cc"] == "nvcc":
                            object_format = "cu"
                    has_c_module = True
                else:
                    assert module.type_key == "llvm" or module.type_key == "static_library"
                    object_format = "o"
            path_obj = os.path.join(workspace_dir, f"lib{index}.{object_format}")
            module.save(path_obj)
            files.append(path_obj)
            is_system_lib = (
                module.type_key == "llvm" and module.get_function("__tvm_is_system_module")()
            )
            llvm_target_string = (
                module.type_key == "llvm" and module.get_function("_get_target_string")()
            )
        if not fcompile:
            if file_name.endswith(".tar"):
                fcompile = _tar.tar
            else:
                fcompile = _cc.create_shared

        if llvm_target_string is None and hasattr(fcompile, "get_target_triple"):
            triple = fcompile.get_target_triple()
            assert triple, "Target triple should not be empty"
            llvm_target_string = "llvm -mtriple " + triple

        if getattr(fcompile, "need_system_lib", False) and not is_system_lib:
            raise ValueError("%s need --system-lib option" % str(fcompile))

        if self.imported_modules:
            if enabled("llvm") and llvm_target_string:
                path_obj = os.path.join(workspace_dir, f"devc.{object_format}")
                m = _ffi_api.ModulePackImportsToLLVM(self, is_system_lib, llvm_target_string)
                m.save(path_obj)
                files.append(path_obj)
            else:
                path_cc = os.path.join(workspace_dir, "devc.c")
                with open(path_cc, "w") as f:
                    f.write(_ffi_api.ModulePackImportsToC(self, is_system_lib))
                files.append(path_cc)

        # The imports could contain a c module but the object format could be tar
        # Thus, it would not recognize the following include paths as options
        # which are there assuming a c compiler is the fcompile.
        if has_c_module and not file_name.endswith(".tar"):
            options = []
            if "options" in kwargs:
                opts = kwargs["options"]
                options = opts if isinstance(opts, (list, tuple)) else [opts]
            opts = options + ["-I" + path for path in find_include_path()]
            kwargs.update({"options": opts})

        return fcompile(file_name, files, **kwargs)


def system_lib():
    """Get system-wide library module singleton.

    System lib is a global module that contains self register functions in startup.
    Unlike normal dso modules which need to be loaded explicitly.
    It is useful in environments where dynamic loading api like dlopen is banned.

    To build system lib function, simply specify target option ```llvm --system-lib```
    The system lib will be available as long as the result code is linked by the program.

    The system lib is intended to be linked and loaded during the entire life-cyle of the program.
    If you want dynamic loading features, use dso modules instead.

    Returns
    -------
    module : runtime.Module
        The system-wide library module.
    """
    return _ffi_api.SystemLib()


def load_module(path, fmt=""):
    """Load module from file.

    Parameters
    ----------
    path : str
        The path to the module file.

    fmt : str, optional
        The format of the file, if not specified
        it will be inferred from suffix of the file.

    Returns
    -------
    module : runtime.Module
        The loaded module

    Note
    ----
    This function will automatically call
    cc.create_shared if the path is in format .o or .tar
    """
    if os.path.isfile(path):
        path = os.path.realpath(path)
    else:
        raise ValueError("cannot find file %s" % path)

    # High level handling for .o and .tar file.
    # We support this to be consistent with RPC module load.
    if path.endswith(".o"):
        # Extra dependencies during runtime.
        from tvm.contrib import cc as _cc

        _cc.create_shared(path + ".so", path)
        path += ".so"
    elif path.endswith(".tar"):
        # Extra dependencies during runtime.
        from tvm.contrib import cc as _cc, utils as _utils, tar as _tar

        tar_temp = _utils.tempdir(custom_path=path.replace(".tar", ""))
        _tar.untar(path, tar_temp.temp_dir)
        files = [tar_temp.relpath(x) for x in tar_temp.listdir()]
        _cc.create_shared(path + ".so", files)
        path += ".so"
    # Redirect to the load API
    return _ffi_api.ModuleLoadFromFile(path, fmt)


def load_static_library(path, func_names):
    """Load the .o library at path which implements functions with func_names.
    Unlike the generic load_module the result will remain as a static_library
    and will not be relinked on-the-fly into a .so library."""
    return _ffi_api.ModuleLoadStaticLibrary(path, func_names)


def enabled(target):
    """Whether module runtime is enabled for target

    Parameters
    ----------
    target : str
        The target device type.

    Returns
    -------
    enabled : bool
        Whether runtime is enabled.

    Examples
    --------
    The following code checks if gpu is enabled.

    >>> tvm.runtime.enabled("gpu")
    """
    return _ffi_api.RuntimeEnabled(target)


def num_threads() -> int:
    """Get the number of threads in use by the TVM runtime.

    Returns
    -------
    int
        Number of threads in use.
    """
    return _ffi_api.NumThreads()


_set_class_module(Module)
