"""Symbolic configuration API."""
from __future__ import absolute_import as _abs
import sys as _sys
import os as _os
import ctypes as _ctypes

from numbers import Number as _Number
from . import _base
from ._base import _LIB, check_call as _check_call
from . import _symbol_internal as _internal
from .attribute import AttrScope

# Use different verison of SymbolBase
# When possible, use cython to speedup part of computation.

try:
    if int(_os.environ.get("MXNET_ENABLE_CYTHON", True)) == 0:
        from ._ctypes.symbol import SymbolBase,  _init_symbol_module
    elif _sys.version_info >= (3, 0):
        from ._cy3.symbol import SymbolBase,  _init_symbol_module
    else:
        from ._cy2.symbol import SymbolBase,  _init_symbol_module
except ImportError:
    from ._ctypes.symbol import SymbolBase,  _init_symbol_module


class Symbol(SymbolBase):
    """Symbol is basic operation unit for symbolic graph compostion."""
    # disable dictionary storage, also do not have parent type.
    __slots__ = []

    def __add__(self, other):
        if isinstance(other, Symbol):
            return _internal.__add_symbol__(self, other)
        elif isinstance(other, _Number):
            return _internal.__add_scalar__(self, scalar=other)
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Symbol):
            return _internal.__sub_symbol__(self, other)
        if isinstance(other, _Number):
            return _internal.__sub_scalar__(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rsub__(self, other):
        if isinstance(other, _Number):
            return _internal.__rsub_scalar__(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __mul__(self, other):
        if isinstance(other, Symbol):
            return _internal.__mul_symbol__(self, other)
        if isinstance(other, _Number):
            return _internal.__mul_scalar__(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, Symbol):
            return _internal.__div_symbol__(self, other)
        if isinstance(other, _Number):
            return _internal.__div_scalar__(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rdiv__(self, other):
        if isinstance(other, _Number):
            return _internal.__rdiv_scalar__(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __pow__(self, other):
        if isinstance(other, Symbol):
            return _internal.__pow_symbol__(self, other)
        if isinstance(other, _Number):
            return _internal.__pow_scalar__(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rpow__(self, other):
        if isinstance(other, _Number):
            return _internal.__rpow_scalar__(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __neg__(self):
        return self.__mul__(-1.0)

    def __copy__(self):
        return self.__deepcopy__()

    def __deepcopy__(self, _=None):
        handle = _base.SymbolHandle()
        _base.check_call(_LIB.NNSymbolCopy(self.handle,
                                           _ctypes.byref(handle)))
        return Symbol(handle)

    def __getitem__(self, index):
        if isinstance(index, _base.string_types):
            idx = None
            for i, name in enumerate(self.list_output_names()):
                if name == index:
                    if idx is not None:
                        raise ValueError('There are multiple outputs with name \"%s\"' % index)
                    idx = i
            if idx is None:
                raise ValueError('Cannot find output that matches name \"%s\"' % index)
            index = idx
        if not isinstance(index, int):
            raise TypeError('Symbol only support integer index to fetch i-th output')
        handle = _base.SymbolHandle()
        _check_call(_LIB.NNSymbolGetOutput(
            self.handle, _base.nn_uint(index), _ctypes.byref(handle)))
        return Symbol(handle=handle)

    def attr(self, key):
        """Get attribute string from the symbol, this function only works for non-grouped symbol.

        Parameters
        ----------
        key : str
            The key to get attribute from.

        Returns
        -------
        value : str
            The attribute value of the key, returns None if attribute do not exist.
        """
        ret = _ctypes.c_char_p()
        success = _ctypes.c_int()
        _check_call(_LIB.NNSymbolGetAttr(
            self.handle, _base.c_str(key), _ctypes.byref(ret), _ctypes.byref(success)))
        if success.value != 0:
            return _base.py_str(ret.value)
        else:
            return None

    def list_attr(self, recursive=False):
        """Get all attributes from the symbol.

        Parameters
        ----------
        recursive : bool
            Default `False`. When `recursive` is `True`, list recursively all the
            attributes in the descendents. The attribute names are pre-pended with
            the symbol names to avoid conflicts. If `False`, then only attributes
            that belongs to this symbol is returned, and the attribute names will
            **not** be pre-pended with the symbol name.
        """
        size = _base.nn_uint()
        pairs = _ctypes.POINTER(_ctypes.c_char_p)()
        option = _ctypes.c_int(0) if recursive else _ctypes.c_int(1)
        _check_call(_LIB.NNSymbolListAttrs(
            self.handle, option, _ctypes.byref(size), _ctypes.byref(pairs)))
        return {_base.py_str(pairs[i*2]): _base.py_str(pairs[i*2+1]) for i in range(size.value)}

    def get_internals(self):
        """Get a new grouped symbol whose output contains all the internal outputs of this symbol.

        Returns
        -------
        sgroup : Symbol
            The internal of the symbol.
        """
        handle = _base.SymbolHandle()
        _check_call(_LIB.NNSymbolGetInternals(
            self.handle, _ctypes.byref(handle)))
        return Symbol(handle=handle)

    def _get_list_copt(self, option):
        """internal function to get list option"""
        if option == 'all':
            return _ctypes.c_int(0)
        elif option == 'read_only':
            return _ctypes.c_int(1)
        elif option == 'aux_state':
            return _ctypes.c_int(2)
        else:
            raise ValueError("option need to be in {'all', 'read_only, 'aux_state'}")

    def list_input_variables(self, option='all'):
        """List all the input variables in the symbol.

        Parameters
        ----------
        option : {'all', 'read_only', 'aux_state'}, optional
           The listing option
           - 'all' will list all the arguments.
           - 'read_only' lists arguments that are readed by the graph.
           - 'aux_state' lists arguments that are mutated by the graph as state.
        Returns
        -------
        vars : list of symbol
            List of all the variables
        """
        size = _ctypes.c_uint()
        sarr = _ctypes.POINTER(_base.SymbolHandle)()
        _check_call(_LIB.NNSymbolListInputVariables(
            self.handle, self._get_list_copt(option),
            _ctypes.byref(size), _ctypes.byref(sarr)))
        return [Symbol(_base.SymbolHandle(sarr[i])) for i in range(size.value)]

    def list_input_names(self, option='all'):
        """List all the inputs in the symbol.

        Parameters
        ----------
        option : {'all', 'read_only', 'aux_state'}, optional
           The listing option
           - 'all' will list all the arguments.
           - 'read_only' lists arguments that are readed by the graph.
           - 'aux_state' lists arguments that are mutated by the graph as state.
        Returns
        -------
        args : list of string
            List of all the arguments.
        """
        size = _ctypes.c_uint()
        sarr = _ctypes.POINTER(_ctypes.c_char_p)()
        _check_call(_LIB.NNSymbolListInputNames(
            self.handle, self._get_list_copt(option),
            _ctypes.byref(size), _ctypes.byref(sarr)))
        return [_base.py_str(sarr[i]) for i in range(size.value)]

    def list_output_names(self):
        """List all outputs in the symbol.

        Returns
        -------
        returns : list of string
            List of all the outputs.
        """
        size = _ctypes.c_uint()
        sarr = _ctypes.POINTER(_ctypes.c_char_p)()
        _check_call(_LIB.NNSymbolListOutputNames(
            self.handle, _ctypes.byref(size), _ctypes.byref(sarr)))
        return [_base.py_str(sarr[i]) for i in range(size.value)]

    def debug_str(self):
        """Get a debug string.

        Returns
        -------
        debug_str : string
            Debug string of the symbol.
        """
        debug_str = _ctypes.c_char_p()
        _check_call(_LIB.NNSymbolPrint(
            self.handle, _ctypes.byref(debug_str)))
        return _base.py_str(debug_str.value)

    def _add_control_deps(self, deps):
        """Add control flow dependencies.
        This makes current op depend on the deps.
        Only use when necessary,
        this function mutate the current symbol node.

        Returns
        -------
        deps : Symbol for list of symbol
            The dependencies
        """
        if isinstance(deps, list):
            deps = Group(deps)
        _check_call(_LIB.NNAddControlDeps(
            self.handle, deps.handle))


def Variable(name, **kwargs):
    """Create a symbolic variable with specified name.

    Parameters
    ----------
    name : str
        Name of the variable.
    kwargs : dict of string -> string
        Additional attributes to set on the variable.

    Returns
    -------
    variable : Symbol
        The created variable symbol.
    """
    if not isinstance(name, _base.string_types):
        raise TypeError('Expect a string for variable `name`')
    handle = _base.SymbolHandle()
    _base.check_call(_LIB.NNSymbolCreateVariable(
        _base.c_str(name), _ctypes.byref(handle)))
    ret = Symbol(handle)
    attr = AttrScope.current.get(kwargs)
    if attr:
        ret._set_attr(**attr)
    return ret


def Group(symbols):
    """Create a symbol that groups symbols together.

    Parameters
    ----------
    symbols : list
        List of symbols to be grouped.

    Returns
    -------
    sym : Symbol
        The created group symbol.
     """
    ihandles = []
    for sym in symbols:
        if not isinstance(sym, Symbol):
            raise TypeError('Expect Symbols in the list input')
        ihandles.append(sym.handle)
    handle = _base.SymbolHandle()
    _check_call(_LIB.NNSymbolCreateGroup(
        _base.nn_uint(len(ihandles)),
        _base.c_array(_base.SymbolHandle, ihandles),
        _ctypes.byref(handle)))
    return Symbol(handle)

# Set the real symbol class to Symbol
_init_symbol_module(Symbol, "nnvm")
