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
    int NNListAllOpNames(nn_uint *out_size,
                      const char ***out_array);
    int NNGetOpHandle(const char *op_name,
                      OpHandle *handle);
    int NNGetOpInfo(OpHandle op,
                    const char **name,
                    const char **description,
                    nn_uint *num_doc_args,
                    const char ***arg_names,
                    const char ***arg_type_infos,
                    const char ***arg_descriptions,
                    const char **return_type);
    int NNListOpNames(nn_uint *out_size,
                      const char ***out_array);
    int NNSymbolCreateAtomicSymbol(OpHandle op,
                                   nn_uint num_param,
                                   const char **keys,
                                   const char **vals,
                                   SymbolHandle *out);
    int NNSymbolFree(SymbolHandle symbol);
    int NNSymbolSetAttrs(SymbolHandle symbol,
                         nn_uint num_param,
                         const char** keys,
                         const char** values);
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

cdef _set_symbol_class(cls):
    global _symbol_cls
    _symbol_cls = cls

cdef NewSymbol(SymbolHandle handle):
    """Create a new symbol given handle"""
    sym = _symbol_cls(None)
    (<SymbolBase>sym).handle = handle
    return sym

cdef _make_atomic_symbol_function(OpHandle handle, string name):
    """Create an atomic symbol function by handle and funciton name."""
    cdef const char *real_name
    cdef const char *desc
    cdef nn_uint num_args
    cdef const char** arg_names
    cdef const char** arg_types
    cdef const char** arg_descs
    cdef const char* return_type

    CALL(NNGetOpInfo(
        handle, &real_name, &desc,
        &num_args, &arg_names,
        &arg_types, &arg_descs,
        &return_type))

    param_str = BuildDoc(num_args, arg_names, arg_types, arg_descs)
    func_name = py_str(name.c_str())
    doc_str = ('%s\n\n' +
               '%s\n' +
               'Returns\n' +
               '-------\n' +
               'result: Tensor\n' +
               '    The result Tensor.')
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


def _init_symbol_module(symbol_class, root_namespace):
    """List and add all the atomic symbol functions to current module."""
    cdef const char** op_name_ptrs
    cdef nn_uint size
    cdef vector[string] op_names
    cdef OpHandle handle

    _set_symbol_class(symbol_class)
    CALL(NNListAllOpNames(&size, &op_name_ptrs))
    for i in range(size):
        op_names.push_back(string(op_name_ptrs[i]));
    module_obj = _sys.modules["%s.symbol" % root_namespace]
    module_internal = _sys.modules["%s._symbol_internal" % root_namespace]
    for i in range(op_names.size()):
        CALL(NNGetOpHandle(op_names[i].c_str(), &handle))
        function = _make_atomic_symbol_function(handle, op_names[i])
        if function.__name__.startswith('_'):
            setattr(module_internal, function.__name__, function)
            setattr(module_obj, function.__name__, function)
        else:
            setattr(module_obj, function.__name__, function)
