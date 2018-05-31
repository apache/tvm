# pylint: disable=invalid-name
"""Attr dictionary object used by schedule functions"""
import tvm

_dict_get = tvm.get_global_func("nnvm.compiler._dict_get")
_dict_size = tvm.get_global_func("nnvm.compiler._dict_size")
_dict_keys = tvm.get_global_func("nnvm.compiler._dict_keys")

class AttrDict(object):
    """Attribute dictionary in nnvm.

    Used by python registration of compute and schedule function.
    AttrDict is passed as the first argument to schedule and compute function.
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
        return tuple(int(x) for x in self[key][1:-1].split(",") if x)

    def get_int_pair_tuple(self, key):
        """Get tuple of integer pairs from attr dict

        Parameters
        ----------
        key : str
            The attr key

        Returns
        -------
        tuple : tuple of int pairs
            The result tuple
        """
        flat = [int(x.strip(' [] ')) for x in self[key][1:-1].split(",")]
        return tuple((flat[i], flat[i+1]) for i in range(0, len(flat), 2))

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

    def get_float_tuple(self, key):
        """Get tuple of float from attr dict

        Parameters
        ----------
        key : str
            The attr key

        Returns
        -------
        tuple : tuple of float
            The result tuple
        """
        return tuple(float(x) for x in self[key][1:-1].split(",") if x)

    def get_float(self, key):
        """Get float from attr dict

        Parameters
        ----------
        key : str
            The attr key

        Returns
        -------
        value : float
            The result value
        """
        return float(self[key])

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
        lowercase = self[key].lower()
        if lowercase == "1":
            return True
        elif lowercase == "0":
            return False
        elif lowercase == "true":
            return True
        elif lowercase == "false":
            return False
        else:
            raise ValueError("Wrong bool format for key %s" % key)

    def get_string(self, key):
        """Get string from attr dict

        Parameters
        ----------
        key : str
            The attr key

        Returns
        -------
        value : str
            The result value
        """
        return self[key]

    def __repr__(self):
        return str({k : self[k] for k in self.keys()})


tvm.register_extension(AttrDict, AttrDict)
