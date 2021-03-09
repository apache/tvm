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
from collections import namedtuple

import tvm._ffi
from tvm._ffi.base import _LIB, check_call, c_str, string_types, _RUNTIME_ONLY
from tvm._ffi.libinfo import find_include_path
from .packed_func import PackedFunc, PackedFuncHandle, _set_class_module

from . import _ffi_api


# profile result of time evaluator
ProfileResult = namedtuple("ProfileResult", ["mean", "results"])


class Module(object):
    """Runtime Module."""

    __slots__ = ["handle", "_entry", "entry_name"]

    def __init__(self, handle):
        self.handle = handle
        self._entry = None
        self.entry_name = "__tvm_main__"

    def __del__(self):
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

    def time_evaluator(self, func_name, ctx, number=10, repeat=1, min_repeat_ms=0, f_preproc=""):
        """Get an evaluator that measures time cost of running function.

        Parameters
        ----------
        func_name: str
            The name of the function in the module.

        ctx: TVMContext
            The context we should run this function on.

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
        f_preproc: str, optional
            The preprocess function name we want to execute before executing the time evaluator.

        Note
        ----
        The function will be invoked  (1 + number x repeat) times,
        with the first call discarded in case there is lazy initialization.

        Returns
        -------
        ftimer : function
            The function that takes same argument as func and returns a ProfileResult.
            The ProfileResult reports `repeat` time costs in seconds.
        """
        try:
            feval = _ffi_api.RPCTimeEvaluator(
                self,
                func_name,
                ctx.device_type,
                ctx.device_id,
                number,
                repeat,
                min_repeat_ms,
                f_preproc,
            )

            def evaluator(*args):
                """Internal wrapped evaluator."""
                # Wrap feval so we can add more stats in future.
                blob = feval(*args)
                fmt = "@" + ("d" * repeat)
                results = struct.unpack(fmt, blob)
                mean = sum(results) / float(repeat)
                return ProfileResult(mean=mean, results=results)

            return evaluator
        except NameError:
            raise NameError("time_evaluate is only supported when RPC is enabled")

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
        is_dso_exportable = lambda m: (m.type_key == "llvm" or m.type_key == "c")
        return self._collect_from_import_tree(is_dso_exportable)

    def export_library(self, file_name, fcompile=None, addons=None, workspace_dir=None, **kwargs):
        """Export the module and its imported device code one library.

        This function only works on host llvm modules.
        It will pack all the imported modules

        Parameters
        ----------
        file_name : str
            The name of the shared library.

        fcompile : function(target, file_list, kwargs), optional
            Compilation function to use create dynamic library.
            If fcompile has attribute object_format, will compile host library
            to that format. Otherwise, will use default format "o".

        workspace_dir : str, optional
            the path to a directory used to create intermediary
            artifacts for the process exporting of the library.
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
        llvm_target_triple = None
        for index, module in enumerate(modules):
            if fcompile is not None and hasattr(fcompile, "object_format"):
                if module.type_key == "c":
                    object_format = "c"
                    has_c_module = True
                else:
                    object_format = fcompile.object_format
            else:
                if module.type_key == "llvm":
                    object_format = "o"
                else:
                    assert module.type_key == "c"
                    object_format = "c"
                    has_c_module = True
            path_obj = os.path.join(workspace_dir, f"lib{index}.{object_format}")
            module.save(path_obj)
            files.append(path_obj)
            is_system_lib = (
                module.type_key == "llvm" and module.get_function("__tvm_is_system_module")()
            )
            llvm_target_triple = (
                module.type_key == "llvm" and module.get_function("_get_target_triple")()
            )
        if not fcompile:
            if file_name.endswith(".tar"):
                fcompile = _tar.tar
            else:
                fcompile = _cc.create_shared

        if llvm_target_triple is None and hasattr(fcompile, "get_target_triple"):
            llvm_target_triple = fcompile.get_target_triple()

        if getattr(fcompile, "need_system_lib", False) and not is_system_lib:
            raise ValueError("%s need --system-lib option" % str(fcompile))

        if self.imported_modules:
            if enabled("llvm") and llvm_target_triple:
                path_obj = os.path.join(workspace_dir, f"devc.{object_format}")
                m = _ffi_api.ModulePackImportsToLLVM(self, is_system_lib, llvm_target_triple)
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

    # c++ compiler/linker
    cc = os.environ.get("CXX", "g++")

    # High level handling for .o and .tar file.
    # We support this to be consistent with RPC module load.
    if path.endswith(".o"):
        # Extra dependencies during runtime.
        from tvm.contrib import cc as _cc

        _cc.create_shared(path + ".so", path, cc=cc)
        path += ".so"
    elif path.endswith(".tar"):
        # Extra dependencies during runtime.
        from tvm.contrib import cc as _cc, utils as _utils, tar as _tar

        tar_temp = _utils.tempdir(custom_path=path.replace(".tar", ""))
        _tar.untar(path, tar_temp.temp_dir)
        files = [tar_temp.relpath(x) for x in tar_temp.listdir()]
        _cc.create_shared(path + ".so", files, cc=cc)
        path += ".so"
    # TODO(weberlo): we should probably use a more distinctive suffix for uTVM object files
    elif path.endswith(".obj"):
        fmt = "micro_dev"
    # Redirect to the load API
    return _ffi_api.ModuleLoadFromFile(path, fmt)


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


_set_class_module(Module)
