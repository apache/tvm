"""Container of compiled functions of TVM."""
from __future__ import absolute_import as _abs
from ._ffi.function import ModuleBase, _set_class_module
from ._ffi.function import _init_api
from .contrib import cc_compiler as _cc, util as _util

class Module(ModuleBase):
    """Module container of all TVM generated functions"""
    def __repr__(self):
        return "Module(%s, %x)" % (self.type_key, self.handle.value)

    @property
    def type_key(self):
        """Get type key of the module."""
        return _GetTypeKey(self)

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
        return _GetSource(self, fmt)

    @property
    def imported_modules(self):
        """Get imported modules

        Returns
        ----------
        modules : list of Module
            The module
        """
        nmod = _ImportsSize(self)
        return [_GetImport(self, i) for i in range(nmod)]

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
        Module.export_library : export the module to shared library.
        """
        _SaveToFile(self, file_name, fmt)

    def export_library(self, file_name):
        """Export the module and its imported device code one library.

        This function only works on host llvm modules.
        It will pack all the imported modules

        Parameters
        ----------
        file_name : str
            The name of the shared library.
        """
        if self.type_key != "llvm":
            raise ValueError("Module[%s]: Only llvm support export shared" % self.type_key)
        temp = _util.tempdir()
        path_obj = temp.relpath("lib.o")
        self.save(path_obj)
        files = [path_obj]
        if self.imported_modules:
            path_cc = temp.relpath("devc.cc")
            with open(path_cc, "w") as f:
                f.write(_PackImportsToC(self))
            files.append(path_cc)
        _cc.create_shared(file_name, files)

    def time_evaluator(self, func_name, ctx, number):
        """Get an evaluator that measures time cost of running function.

        Parameters
        ----------
        func_name: str
            The name of the function in the module.

        ctx: TVMContext
            The context we should run this function on.

        number: int
            The number of repeative times to run evaluation.

        Note
        ----
        The function will be invoked number + 1 times,
        with the first call discarded in case there is lazy initialization.

        Returns
        -------
        ftimer : Function
            The function that takes same argument as func
            and return a float representing seconds per function call.
        """
        try:
            return _RPCTimeEvaluator(
                self, func_name, ctx.device_type,  ctx.device_id, number)
        except NameError:
            raise NameError("time_evaluate is only supported when RPC is enabled")


def load(path, fmt=""):
    """Load module from file

    Parameters
    ----------
    path : str
        The path to the module file.

    fmt : str, optional
        The format of the file, if not specified
        it will be inferred from suffix of the file.

    Returns
    -------
    module : Module
        The loaded module
    """
    return _LoadFromFile(path, fmt)


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

    >>> tvm.module.enabled("gpu")
    """
    return _Enabled(target)


_init_api("tvm.module")
_set_class_module(Module)
