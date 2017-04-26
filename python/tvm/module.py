"""Container of compiled functions of TVM."""
from __future__ import absolute_import as _abs
from ._ffi.function import ModuleBase, _init_module_module
from ._ffi.function import _init_api


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

        Parameters
        ----------
        file_name : str
            The name of the file.
        fmt : str
            The format of the file.
        """
        _SaveToFile(self, file_name, fmt)


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
_init_module_module(Module)
