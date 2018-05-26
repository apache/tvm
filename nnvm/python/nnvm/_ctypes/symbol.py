# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, too-many-lines,
# pylint: disable=len-as-condition, consider-iterating-dictionary
"""Symbolic configuration API."""
from __future__ import absolute_import as _abs

import copy
import ctypes
import sys
from .._base import _LIB
from .._base import c_array, c_str, nn_uint, py_str
from .._base import SymbolHandle, OpHandle
from .._base import check_call, ctypes2docstring
from ..name import NameManager
from ..attribute import AttrScope

class SymbolBase(object):
    """Symbol is symbolic graph."""
    __slots__ = ["handle"]
    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolHandle
            the handle to the underlying C++ Symbol
        """
        self.handle = handle

    def __del__(self):
        check_call(_LIB.NNSymbolFree(self.handle))

    def __call__(self, *args, **kwargs):
        """Invoke symbol as function on inputs.

        Parameters
        ----------
        args:
            provide positional arguments

        kwargs:
            provide keyword arguments
        Returns
        -------
        the resulting symbol
        """
        s = copy.deepcopy(self)
        s._compose(*args, **kwargs)
        return s

    def _compose(self, *args, **kwargs):
        """Compose symbol on inputs.

        This call mutates the current symbol.

        Parameters
        ----------
        args:
            provide positional arguments

        kwargs:
            provide keyword arguments

        Returns
        -------
        the resulting symbol
        """
        name = kwargs.pop('name', None)

        if name:
            name = c_str(name)
        if len(args) != 0 and len(kwargs) != 0:
            raise TypeError('compose only accept input Symbols \
                either as positional or keyword arguments, not both')

        for arg in args:
            if not isinstance(arg, SymbolBase):
                raise TypeError('Compose expect `Symbol` as arguments')
        for val in kwargs.values():
            if not isinstance(val, SymbolBase):
                raise TypeError('Compose expect `Symbol` as arguments')

        num_args = len(args) + len(kwargs)
        if len(kwargs) != 0:
            keys = c_array(ctypes.c_char_p, [c_str(key) for key in kwargs.keys()])
            args = c_array(SymbolHandle, [s.handle for s in kwargs.values()])
        else:
            keys = None
            args = c_array(SymbolHandle, [s.handle for s in args])
        check_call(_LIB.NNSymbolCompose(
            self.handle, name, num_args, keys, args))

    def _set_attr(self, **kwargs):
        """Set the attribute of the symbol.

        Parameters
        ----------
        **kwargs
            The attributes to set
        """
        keys = c_array(ctypes.c_char_p,
                       [c_str(key) for key in kwargs.keys()])
        vals = c_array(ctypes.c_char_p,
                       [c_str(str(val)) for val in kwargs.values()])
        num_args = nn_uint(len(kwargs))
        check_call(_LIB.NNSymbolSetAttrs(
            self.handle, num_args, keys, vals))


_symbol_cls = SymbolBase

def _set_symbol_class(cls):
    global _symbol_cls
    _symbol_cls = cls


def _make_atomic_symbol_function(handle, name):
    """Create an atomic symbol function by handle and funciton name."""
    real_name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = nn_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()
    ret_type = ctypes.c_char_p()

    check_call(_LIB.NNGetOpInfo(
        handle, ctypes.byref(real_name), ctypes.byref(desc),
        ctypes.byref(num_args),
        ctypes.byref(arg_names),
        ctypes.byref(arg_types),
        ctypes.byref(arg_descs),
        ctypes.byref(ret_type)))
    param_str = ctypes2docstring(num_args, arg_names, arg_types, arg_descs)
    func_name = name
    desc = py_str(desc.value)

    doc_str = ('%s\n\n' +
               '%s\n' +
               'name : string, optional.\n' +
               '    Name of the resulting symbol.\n\n' +
               'Returns\n' +
               '-------\n' +
               'symbol: Symbol\n' +
               '    The result symbol.')
    doc_str = doc_str % (desc, param_str)

    def creator(*args, **kwargs):
        """Activation Operator of Neural Net.
        The parameters listed below can be passed in as keyword arguments.

        Parameters
        ----------
        name : string, required.
            Name of the resulting symbol.

        Returns
        -------
        symbol: Symbol
            the resulting symbol
        """
        param_keys = []
        param_vals = []
        symbol_kwargs = {}
        name = kwargs.pop('name', None)
        attr = kwargs.pop('attr', None)

        for k, v in kwargs.items():
            if isinstance(v, SymbolBase):
                symbol_kwargs[k] = v
            else:
                param_keys.append(c_str(k))
                param_vals.append(c_str(str(v)))
        # create atomic symbol
        param_keys = c_array(ctypes.c_char_p, param_keys)
        param_vals = c_array(ctypes.c_char_p, param_vals)
        sym_handle = SymbolHandle()
        check_call(_LIB.NNSymbolCreateAtomicSymbol(
            handle,
            nn_uint(len(param_keys)),
            param_keys, param_vals,
            ctypes.byref(sym_handle)))

        if len(args) != 0 and len(symbol_kwargs) != 0:
            raise TypeError(
                '%s can only accept input'
                'Symbols either as positional or keyword arguments, not both' % func_name)
        s = _symbol_cls(sym_handle)
        attr = AttrScope.current.get(attr)
        if attr:
            s._set_attr(**attr)
        hint = func_name.lower()
        name = NameManager.current.get(name, hint)
        s._compose(*args, name=name, **symbol_kwargs)
        return s

    creator.__name__ = func_name
    creator.__doc__ = doc_str
    return creator


def _init_symbol_module(symbol_class, root_namespace):
    """List and add all the atomic symbol functions to current module."""
    _set_symbol_class(symbol_class)
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.NNListAllOpNames(ctypes.byref(size),
                                     ctypes.byref(plist)))
    op_names = []
    for i in range(size.value):
        op_names.append(py_str(plist[i]))

    module_obj = sys.modules["%s.symbol" % root_namespace]
    module_internal = sys.modules["%s._symbol_internal" % root_namespace]
    for name in op_names:
        hdl = OpHandle()
        check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
        function = _make_atomic_symbol_function(hdl, name)
        if function.__name__.startswith('_'):
            setattr(module_internal, function.__name__, function)
            setattr(module_obj, function.__name__, function)
        else:
            setattr(module_obj, function.__name__, function)
