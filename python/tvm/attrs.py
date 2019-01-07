""" TVM Attribute module, which is mainly used for defining attributes of operators"""
from ._ffi.node import NodeBase, register_node as _register_tvm_node
from ._ffi.function import _init_api
from . import _api_internal


@_register_tvm_node
class Attrs(NodeBase):
    """Attribute node, which is mainly use for defining attributes of relay operators.

    Used by function registered in python side, such as compute, schedule and alter_layout.
    Attrs is passed as the first argument to these functions.
    """
    def list_field_info(self):
        """ Get fields information

        Returns
        -------
        infos: list of AttrFieldInfo
            List of field information
        """
        return _api_internal._AttrsListFieldInfo(self)

    def keys(self):
        """Get list of names in the attribute.

        Returns
        -------
        keys : list of str
            List of keys
        """
        fields = self.list_field_info()
        for field in fields:
            yield field.name

    def get_int_tuple(self, key):
        """Get a python int tuple of a key

        Parameters
        ----------
        key: str

        Returns
        -------
        value: Tuple of int
        """
        return tuple(x.value for x in self.__getattr__(key))

    def get_int(self, key):
        """Get a python int value of a key

        Parameters
        ----------
        key: str

        Returns
        -------
        value: int
        """
        return self.__getattr__(key)

    def get_str(self, key):
        """Get a python int value of a key

        Parameters
        ----------
        key: str

        Returns
        -------
        value: int
        """
        return self.__getattr__(key)

    def __getitem__(self, item):
        return self.__getattr__(item)


_init_api("tvm.attrs")
