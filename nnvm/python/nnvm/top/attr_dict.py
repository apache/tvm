# pylint: disable=invalid-name
"""Attr dictionary object used by schedule functions"""

import json
import tvm

_dict_get = tvm.get_global_func("nnvm.compiler._dict_get")
_dict_size = tvm.get_global_func("nnvm.compiler._dict_size")
_dict_keys = tvm.get_global_func("nnvm.compiler._dict_keys")

class AttrDict(object):
    """Attribute dictionary in nnvm.

    Used by python registration of compute and schedule function.
    """
    _tvm_tcode = 18

    def __init__(self, handle):
        self.handle = handle

    def __del__(self):
        tvm.nd.free_extension_handle(self.handle, 18)

    @property
    def _tvm_handle(self):
        return self.handle.value

    def __getitem__(self, key):
        return _dict_get(self, key)

    def keys(self):
        """Get list of keys in the dict.

        Returns
        -------
        keys : list of str
            List of keys
        """
        return [x.value for x in _dict_keys(self)]

    def get_int_tuple(self, key):
        """Get tuple of integer from attr dict

        Parameters
        ----------
        key : str
            The attr key

        Returns
        -------
        tuple : tuple of int
            The result tuple
        """
        return tuple(json.loads(self[key]))

    def get_int(self, key):
        """Get integer from attr dict

        Parameters
        ----------
        key : str
            The attr key

        Returns
        -------
        value : int
            The result value
        """
        return int(self[key])

    def get_bool(self, key):
        """Get bool from attr dict

        Parameters
        ----------
        key : str
            The attr key

        Returns
        -------
        value : bool
            The result value
        """
        return self[key] != "False"

    def __repr__(self):
        return str({k : self[k] for k in self.keys()})


tvm.register_extension(AttrDict, AttrDict)
