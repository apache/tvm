import sys
from libcpp.vector cimport vector

ctypedef void* SymbolHandle
ctypedef void* AtomicSymbolCreator
ctypedef unsigned nn_uint

cdef extern from "nnvm/c_api.h":
    int NNSymbolFree(SymbolHandle symbol)
    int NNSymbolCreateVariable(const char *name, SymbolHandle *out)
    const char* NNGetLastError()
    int NNSymbolPrint(SymbolHandle symbol, const char **out_str)
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
    int NNSymbolCompose(SymbolHandle sym,
                        const char* name,
                        nn_uint num_args,
                        const char** keys,
                        SymbolHandle* args);

cdef CALL(int ret):
    if ret != 0:
        raise RuntimeError(NNGetLastError())

cdef const char** CBeginPtr(vector[const char*]& vec):
    if (vec.size() != 0):
        return &vec[0]
    else:
        return NULL

cdef ctypes2docstring(nn_uint num_args,
                      const char** arg_names,
                      const char** arg_types,
                      const char** arg_descs,
                      remove_dup=True):
    """Convert ctypes returned doc string information into parameters docstring.

    num_args : nn_uint
        Number of arguments.

    arg_names : ctypes.POINTER(ctypes.c_char_p)
        Argument names.

    arg_types : ctypes.POINTER(ctypes.c_char_p)
        Argument type information.

    arg_descs : ctypes.POINTER(ctypes.c_char_p)
        Argument description information.

    remove_dup : boolean, optional
        Whether remove duplication or not.

    Returns
    -------
    docstr : str
        Python docstring of parameter sections.
    """
    param_keys = set()
    param_str = []
    for i in range(num_args):
        key = arg_names[i]
        if key in param_keys and remove_dup:
            continue
        param_keys.add(key)
        type_info = arg_types[i]
        ret = '%s : %s' % (key, type_info)
        if len(arg_descs[i]) != 0:
            ret += '\n    ' + arg_descs[i]
        param_str.append(ret)
    doc_str = ('Parameters\n' +
               '----------\n' +
               '%s\n')
    doc_str = doc_str % ('\n'.join(param_str))
    return doc_str


cdef class Symbol:
    # handle for symbolic operator.
    cdef SymbolHandle handle

    def __dealloc__(self):
        CALL(NNSymbolFree(self.handle))

    def debug_str(self):
        cdef const char* out_str
        CALL(NNSymbolPrint(self.handle, &out_str))
        return str(out_str)

cdef NewSymbol(SymbolHandle handle):
    """Create a new symbol given handle"""
    sym = Symbol()
    sym.handle = handle
    return sym


def Variable(const char* name, **kwargs):
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

    param_str = ctypes2docstring(num_args, arg_names, arg_types, arg_descs)
    func_name = name
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
        cdef vector[const char*] param_keys
        cdef vector[const char*] param_vals
        cdef vector[SymbolHandle] symbol_args
        cdef vector[const char*] symbol_keys
        cdef SymbolHandle ret_handle
        cdef const char* c_name = NULL

        name = kwargs.pop('name', None)
        attr = kwargs.pop('attr', None)
        if name:
            c_name = name

        if len(kwargs) != 0:
            for k, v in kwargs.items():
                if isinstance(v, Symbol):
                    symbol_keys.push_back(k)
                    symbol_args.push_back((<Symbol>v).handle)
                else:
                    param_keys.push_back(k)
                    param_vals.push_back(str(v))

        if len(args) != 0:
            if symbol_args.size() != 0:
                raise TypeError("compose only accept input Symbols\
                    either as positional or keyword arguments, not both")
            for v in args:
                if not isinstance(v, Symbol):
                    raise TypeError('Compose expect `Symbol` as arguments')
                symbol_args.push_back((<Symbol>v).handle)

        CALL(NNSymbolCreateAtomicSymbol(
            handle,
            <nn_uint>param_keys.size(),
            CBeginPtr(param_keys),
            CBeginPtr(param_vals),
            &ret_handle))
        num_args = <nn_uint>(symbol_args.size())
        CALL(NNSymbolCompose(
            ret_handle, c_name, num_args,
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
    module_obj = sys.modules[__name__]
    for i in range(size):
        function = _make_atomic_symbol_function(plist[i])
        if function.__name__.startswith('_'):
            setattr(Symbol, function.__name__, staticmethod(function))
        else:
            setattr(module_obj, function.__name__, function)

# Initialize the atomic symbol in startups
_init_symbol_module()
