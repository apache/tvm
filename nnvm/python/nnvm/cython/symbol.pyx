from __future__ import absolute_import as _abs

import sys as _sys
import ctypes as _ctypes
from numbers import Number as _Number
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

cdef class SymbolBase:
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

    def _set_attr(self, **kwargs):
        """Set the attribute of the symbol.

        Parameters
        ----------
        **kwargs
            The attributes to set
        """
        SymbolSetAttr(self.handle, kwargs)


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


_symbol_cls = SymbolBase

def _set_symbol_class(cls):
    global _symbol_cls
    _symbol_cls = cls

cdef NewSymbol(SymbolHandle handle):
    """Create a new symbol given handle"""
    sym = _symbol_cls(None)
    (<SymbolBase>sym).handle = handle
    return sym

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
                if isinstance(v, SymbolBase):
                    ssymbol_keys.push_back(c_str(k))
                    symbol_args.push_back((<SymbolBase>v).handle)
                else:
                    sparam_keys.push_back(c_str(k))
                    sparam_vals.push_back(c_str(str(v)))

        if len(args) != 0:
            if symbol_args.size() != 0:
                raise TypeError("compose only accept input Symbols\
                    either as positional or keyword arguments, not both")
            for v in args:
                if not isinstance(v, SymbolBase):
                    raise TypeError('Compose expect `Symbol` as arguments')
                symbol_args.push_back((<SymbolBase>v).handle)

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


def _init_symbol_module():
    """List and add all the atomic symbol functions to current module."""
    cdef AtomicSymbolCreator* plist
    cdef nn_uint size
    CALL(NNSymbolListAtomicSymbolCreators(&size, &plist))
    module_obj = _sys.modules["nnvm.symbol"]
    module_internal = _sys.modules["nnvm._symbol_internal"]
    for i in range(size):
        function = _make_atomic_symbol_function(plist[i])

        if function.__name__.startswith('_'):
            setattr(module_internal, function.__name__, function)
        else:
            setattr(module_obj, function.__name__, function)

# Initialize the atomic symbol in startups
_init_symbol_module()
