"""Runtime module related stuffs"""
# pylint: disable=unused-import, invalid-name, undefined-variable
from __future__ import absolute_import as _abs
from ._ctypes._function import ModuleBase, _init_module_module

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
        """
        return _GetSource(self, fmt)

    @property
    def imported_modules(self):
        """Get imported modules

        Returns
        ----------
        modules : list of Modules
            The module
        """
        nmod = ImportsSize(self)
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
    """
    return _LoadFromFile(path, fmt)

_init_module_module(Module)
