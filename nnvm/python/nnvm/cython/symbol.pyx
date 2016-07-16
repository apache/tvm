from __future__ import absolute_import as _abs

import sys as _sys
import ctypes as _ctypes
from .._base import NNVMError
from ..name import NameManager
from ..attribute import AttrScope
from libcpp.vector cimport vector
from libcpp.string cimport string
from cpython.version cimport PY_MAJOR_VERSION

include "./base.pyi"

cdef extern from "nnvm/c_api.h":
    const char* NNGetLastError();
    int NNSymbolCreateVariable(const char *name, SymbolHandle *out);
    int NNSymbolCreateGroup(nn_uint num_symbols,
                            SymbolHandle *symbols,
                            SymbolHandle *out);
    int NNSymbolListAtomicSymbolCreators(nn_uint *out_size,
                                         AtomicSymbolCreator **out_array);
    int NNSymbolCreateAtomicSymbol(AtomicSymbolCreator creator,
                                   nn_uint num_param,
                                   const char **keys,
                                   const char **vals,
                                   SymbolHandle *out);
    int NNSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
                                    const char **name,
                                    const char **description,
                                    nn_uint *num_doc_args,
                                    const char ***arg_names,
                                    const char ***arg_type_infos,
                                    const char ***arg_descriptions,
                                    const char **return_type);
    int NNSymbolFree(SymbolHandle symbol);
    int NNSymbolPrint(SymbolHandle symbol, const char **out_str);
    int NNSymbolCopy(SymbolHandle symbol, SymbolHandle *out);
    int NNSymbolGetAttr(SymbolHandle symbol,
                        const char* key,
                        const char** out,
                        int *success);
    int NNSymbolSetAttrs(SymbolHandle symbol,
                         nn_uint num_param,
                         const char** keys,
                         const char** values);
    int NNSymbolListAttrs(SymbolHandle symbol,
                          int recursive_option,
                          nn_uint *out_size,
                          const char*** out);
    int NNSymbolListArguments(SymbolHandle symbol,
                              nn_uint *out_size,
                              const char ***out_str_array);
    int NNSymbolListOutputs(SymbolHandle symbol,
                            nn_uint *out_size,
                            const char ***out_str_array);
    int NNSymbolGetInternals(SymbolHandle symbol,
                             SymbolHandle *out);
    int NNSymbolGetOutput(SymbolHandle symbol,
                          nn_uint index,
                          SymbolHandle *out);
    int NNSymbolCompose(SymbolHandle sym,
                        const char* name,
                        nn_uint num_args,
                        const char** keys,
                        SymbolHandle* args);


cdef class Symbol:
    """Symbol is symbolic graph."""
    # handle for symbolic operator.
    cdef SymbolHandle handle

    def __init__(self, handle):
        cdef unsigned long ptr
        if handle is None:
            self.handle = NULL
        else:
            ptr = handle.value
            self.handle = <SymbolHandle>(ptr)

    def __dealloc__(self):
        CALL(NNSymbolFree(self.handle))

    @property
    def handle(self):
        return _ctypes.cast(<unsigned long>self.handle, _ctypes.c_void_p)

    def __copy__(self):
        return self.__deepcopy__()

    def __deepcopy__(self, _ = None):
        cdef SymbolHandle handle
        CALL(NNSymbolCopy(self.handle, &handle))
        return NewSymbol(handle)

    def __getitem__(self, index):
        if isinstance(index, str):
            idx = None
            for i, name in enumerate(self.list_outputs()):
                if name == index:
                    if idx is not None:
                        raise ValueError('There are multiple outputs with name \"%s\"' % index)
                    idx = i
            if idx is None:
                raise ValueError('Cannot find output that matches name \"%s\"' % index)
            index = idx
        if not isinstance(index, int):
            raise TypeError('Symbol only support integer index to fetch i-th output')
        cdef SymbolHandle handle
        cdef nn_uint c_index = index
        CALL(NNSymbolGetOutput(self.handle, c_index, &handle))
        return NewSymbol(handle)

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
        cdef const char* ret
        cdef int success
        key = c_str(key)

        CALL(NNSymbolGetAttr(
            self.handle, key, &ret, &success))
        if success != 0:
            return py_str(ret.value)
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
        cdef nn_uint size
        cdef const char** pairs
        cdef int option
        option = 0 if recursive else 1
        CALL(NNSymbolListAttrs(
            self.handle, option, &size, &pairs))
        return {py_str(pairs[i*2]): py_str(pairs[i*2+1]) for i in range(size)}

    def _set_attr(self, **kwargs):
        """Set the attribute of the symbol.

        Parameters
        ----------
        **kwargs
            The attributes to set
        """
        SymbolSetAttr(self.handle, kwargs)

    def get_internals(self):
        """Get a new grouped symbol whose output contains all the internal outputs of this symbol.

        Returns
        -------
        sgroup : Symbol
            The internal of the symbol.
        """
        cdef SymbolHandle handle
        CALL(NNSymbolGetInternals(self.handle, &handle))
        return NewSymbol(handle)

    def list_arguments(self):
        """List all the arguments in the symbol.

        Returns
        -------
        args : list of string
            List of all the arguments.
        """
        cdef nn_uint size
        cdef const char ** sarr
        CALL(NNSymbolListArguments(self.handle, &size, &sarr))
        return [py_str(sarr[i]) for i in range(size)]

    def list_outputs(self):
        """List all outputs in the symbol.

        Returns
        -------
        returns : list of string
            List of all the outputs.
        """
        cdef nn_uint size
        cdef const char ** sarr
        CALL(NNSymbolListOutputs(self.handle, &size, &sarr))
        return [py_str(sarr[i]) for i in range(size)]

    def debug_str(self):
        cdef const char* out_str
        CALL(NNSymbolPrint(self.handle, &out_str))
        return py_str(out_str)


cdef SymbolSetAttr(SymbolHandle handle, dict kwargs):
    cdef vector[string] sparam_keys
    cdef vector[string] sparam_vals
    cdef nn_uint num_args
    for k, v in kwargs.items():
        sparam_keys.push_back(c_str(k))
        sparam_vals.push_back(c_str(str(v)))
    # keep strings in vector
    cdef vector[const char*] param_keys = SVec2Ptr(sparam_keys)
    cdef vector[const char*] param_vals = SVec2Ptr(sparam_vals)
    num_args = param_keys.size()
    CALL(NNSymbolSetAttrs(
        handle, num_args, CBeginPtr(param_keys), CBeginPtr(param_vals)))


cdef NewSymbol(SymbolHandle handle):
    """Create a new symbol given handle"""
    sym = Symbol(None)
    sym.handle = handle
    return sym


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
    cdef SymbolHandle handle
    name = c_str(name)
    CALL(NNSymbolCreateVariable(name, &handle))
    return NewSymbol(handle)


cdef _make_atomic_symbol_function(AtomicSymbolCreator handle):
    """Create an atomic symbol function by handle and funciton name."""
    cdef const char *name
    cdef const char *desc
    cdef nn_uint num_args
    cdef const char** arg_names
    cdef const char** arg_types
    cdef const char** arg_descs
    cdef const char* return_type

    CALL(NNSymbolGetAtomicSymbolInfo(
        handle, &name, &desc,
        &num_args, &arg_names,
        &arg_types, &arg_descs,
        &return_type))
    param_str = BuildDoc(num_args, arg_names, arg_types, arg_descs)
    func_name = py_str(name)
    doc_str = ('%s\n\n' +
               '%s\n' +
               'name : string, optional.\n' +
               '    Name of the resulting symbol.\n\n' +
               'Returns\n' +
               '-------\n' +
               'symbol: Symbol\n' +
               '    The result symbol.')
    doc_str = doc_str % (desc, param_str)
    func_hint = func_name.lower()

    def creator(*args, **kwargs):
        cdef vector[string] sparam_keys
        cdef vector[string] sparam_vals
        cdef vector[SymbolHandle] symbol_args
        cdef vector[string] ssymbol_keys
        cdef SymbolHandle ret_handle

        name = kwargs.pop("name", None)
        attr = kwargs.pop("attr", None)

        if len(kwargs) != 0:
            for k, v in kwargs.items():
                if isinstance(v, Symbol):
                    ssymbol_keys.push_back(c_str(k))
                    symbol_args.push_back((<Symbol>v).handle)
                else:
                    sparam_keys.push_back(c_str(k))
                    sparam_vals.push_back(c_str(str(v)))

        if len(args) != 0:
            if symbol_args.size() != 0:
                raise TypeError("compose only accept input Symbols\
                    either as positional or keyword arguments, not both")
            for v in args:
                if not isinstance(v, Symbol):
                    raise TypeError('Compose expect `Symbol` as arguments')
                symbol_args.push_back((<Symbol>v).handle)

        cdef vector[const char*] param_keys = SVec2Ptr(sparam_keys)
        cdef vector[const char*] param_vals = SVec2Ptr(sparam_vals)
        cdef vector[const char*] symbol_keys = SVec2Ptr(ssymbol_keys)

        CALL(NNSymbolCreateAtomicSymbol(
            handle,
            <nn_uint>param_keys.size(),
            CBeginPtr(param_keys),
            CBeginPtr(param_vals),
            &ret_handle))
        num_args = <nn_uint>(symbol_args.size())

        attr = AttrScope.current.get(attr)
        if attr:
            SymbolSetAttr(ret_handle, attr)
        name = NameManager.current.get(name, func_hint)

        cdef const char* c_name = NULL

        if name:
            name = c_str(name)
            c_name = name

        CALL(NNSymbolCompose(
            ret_handle,
            c_name,
            num_args,
            &symbol_keys[0] if symbol_keys.size() != 0 else NULL,
            &symbol_args[0] if symbol_args.size() != 0 else NULL))
        return NewSymbol(ret_handle)

    creator.__name__ = func_name
    creator.__doc__ = doc_str
    return creator


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
    cdef vector[SymbolHandle] ihandles
    cdef SymbolHandle handle

    for sym in symbols:
        if not isinstance(sym, Symbol):
            raise TypeError("Expect Symbols in the list input")
        ihandles.push_back((<Symbol>sym).handle)
    if ihandles.size() == 0:
        raise ValueError("expect at least one element in the input")
    CALL(NNSymbolCreateGroup(<nn_uint>ihandles.size(),
                             &ihandles[0], &handle))
    return NewSymbol(handle)


def _init_symbol_module():
    """List and add all the atomic symbol functions to current module."""
    cdef AtomicSymbolCreator* plist
    cdef nn_uint size
    CALL(NNSymbolListAtomicSymbolCreators(&size, &plist))
    module_obj = _sys.modules["nnvm.symbol"]
    for i in range(size):
        function = _make_atomic_symbol_function(plist[i])
        if function.__name__.startswith('_'):
            setattr(Symbol, function.__name__, staticmethod(function))
        else:
            setattr(module_obj, function.__name__, function)


# Initialize the atomic symbol in startups
_init_symbol_module()
