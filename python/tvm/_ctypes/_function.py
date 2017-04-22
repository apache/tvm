# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-branches, global-statement
"""Function configuration API."""
from __future__ import absolute_import

import ctypes
import sys
import traceback
from numbers import Number, Integral

from .._base import _LIB, check_call
from .._base import c_str, py_str, string_types
from ._types import TVMValue, TypeCode, TVMType, TVMByteArray
from ._types import TVMPackedCFunc, TVMCFuncFinalizer
from ._types import RETURN_SWITCH, C_TO_PY_ARG_SWITCH, _wrap_arg_func
from ._node import NodeBase, NodeGeneric, convert_to_node
from ._ndarray import NDArrayBase

FunctionHandle = ctypes.c_void_p
ModuleHandle = ctypes.c_void_p
TVMRetValueHandle = ctypes.c_void_p

def _ctypes_free_resource(rhandle):
    """callback to free resources when it it not needed."""
    pyobj = ctypes.cast(rhandle, ctypes.py_object)
    ctypes.pythonapi.Py_DecRef(pyobj)

# Global callback that is always alive
TVM_FREE_PYOBJ = TVMCFuncFinalizer(_ctypes_free_resource)
ctypes.pythonapi.Py_IncRef(ctypes.py_object(TVM_FREE_PYOBJ))

def convert_to_tvm_func(pyfunc):
    """Convert a python function to TVM function

    Parameters
    ----------
    pyfunc : python function
        The python function to be converted.

    Returns
    -------
    tvmfunc: tvm.nd.Function
        The converted tvm function.
    """
    local_pyfunc = pyfunc
    def cfun(args, type_codes, num_args, ret, _):
        """ ctypes function """
        num_args = num_args.value if isinstance(num_args, ctypes.c_int) else num_args
        pyargs = [C_TO_PY_ARG_SWITCH[type_codes[i]](args[i]) for i in range(num_args)]
        # pylint: disable=broad-except
        try:
            rv = local_pyfunc(*pyargs)
        except Exception:
            msg = traceback.format_exc()
            _LIB.TVMAPISetLastError(c_str(msg))
            return -1

        if rv is not None:
            if isinstance(rv, tuple):
                raise ValueError("PackedFunction can only support one reurn value")
            temp_args = []
            values, tcodes, _ = _make_tvm_args((rv,), temp_args)
            if not isinstance(ret, TVMRetValueHandle):
                ret = TVMRetValueHandle(ret)
            check_call(_LIB.TVMCFuncSetReturn(ret, values[0], ctypes.c_int(tcodes[0])))
            _ = temp_args
            _ = rv
        return 0

    handle = FunctionHandle()
    f = TVMPackedCFunc(cfun)
    # NOTE: We will need to use python-api to increase ref count of the f
    # TVM_FREE_PYOBJ will be called after it is no longer needed.
    pyobj = ctypes.py_object(f)
    ctypes.pythonapi.Py_IncRef(pyobj)
    check_call(_LIB.TVMFuncCreateFromCFunc(
        f, pyobj, TVM_FREE_PYOBJ, ctypes.byref(handle)))
    return Function(handle, False)


def _make_tvm_args(args, temp_args):
    """Pack arguments into c args tvm call accept"""
    num_args = len(args)
    values = (TVMValue * num_args)()
    type_codes = (ctypes.c_int * num_args)()
    for i, arg in enumerate(args):
        if arg is None:
            values[i].v_handle = None
            type_codes[i] = TypeCode.NULL
        elif isinstance(arg, NDArrayBase):
            values[i].v_handle = ctypes.cast(arg.handle, ctypes.c_void_p)
            type_codes[i] = TypeCode.ARRAY_HANDLE
        elif isinstance(arg, NodeBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.NODE_HANDLE
        elif isinstance(arg, Integral):
            values[i].v_int64 = arg
            type_codes[i] = TypeCode.INT
        elif isinstance(arg, Number):
            values[i].v_float64 = arg
            type_codes[i] = TypeCode.FLOAT
        elif isinstance(arg, TVMType):
            values[i].v_str = c_str(str(arg))
            type_codes[i] = TypeCode.STR
        elif isinstance(arg, bytearray):
            arr = TVMByteArray()
            arr.data = ctypes.cast(
                (ctypes.c_byte * len(arg)).from_buffer(arg),
                ctypes.POINTER(ctypes.c_byte))
            arr.size = len(arg)
            values[i].v_handle = ctypes.c_void_p(ctypes.addressof(arr))
            temp_args.append(arr)
            type_codes[i] = TypeCode.BYTES
        elif isinstance(arg, string_types):
            values[i].v_str = c_str(arg)
            type_codes[i] = TypeCode.STR
        elif isinstance(arg, (list, tuple, dict, NodeGeneric)):
            arg = convert_to_node(arg)
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.NODE_HANDLE
            temp_args.append(arg)
        elif isinstance(arg, ModuleBase):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.MODULE_HANDLE
        elif isinstance(arg, Function):
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.FUNC_HANDLE
        elif callable(arg):
            arg = convert_to_tvm_func(arg)
            values[i].v_handle = arg.handle
            type_codes[i] = TypeCode.FUNC_HANDLE
            temp_args.append(arg)
        else:
            raise TypeError("Don't know how to handle type %s" % type(arg))
    return values, type_codes, num_args


class Function(object):
    """The PackedFunc object used in TVM.

    Function plays an key role to bridge front and backend in TVM.
    Function provide a type-erased interface, you can call function with positional arguments.

    The compiled module returns Function.
    TVM backend also registers and exposes its API as Functions.
    For example, the developer function exposed in tvm.ir_pass are actually
    C++ functions that are registered as PackedFunc

    The following are list of common usage scenario of tvm.Function.

    - Automatic exposure of C++ API into python
    - To call PackedFunc from python side
    - To call python callbacks to inspect results in generated code
    - Bring python hook into C++ backend

    See Also
    --------
    tvm.register_func: How to register global function.
    tvm.get_global_func: How to get global function.
    """
    __slots__ = ["handle", "is_global"]
    # pylint: disable=no-member
    def __init__(self, handle, is_global):
        """Initialize the function with handle

        Parameters
        ----------
        handle : FunctionHandle
            the handle to the underlying function.

        is_global : bool
            Whether this is a global function in python
        """
        self.handle = handle
        self.is_global = is_global

    def __del__(self):
        if not self.is_global:
            check_call(_LIB.TVMFuncFree(self.handle))

    def __call__(self, *args):
        """Call the function with positional arguments

        args : list
           The positional arguments to the function call.
        """
        temp_args = []
        values, tcodes, num_args = _make_tvm_args(args, temp_args)
        ret_val = TVMValue()
        ret_tcode = ctypes.c_int()
        check_call(_LIB.TVMFuncCall(
            self.handle, values, tcodes, ctypes.c_int(num_args),
            ctypes.byref(ret_val), ctypes.byref(ret_tcode)))
        _ = temp_args
        _ = args
        return RETURN_SWITCH[ret_tcode.value](ret_val)


class ModuleBase(object):
    """Base class for module"""
    __slots__ = ["handle", "_entry"]
    def __init__(self, handle):
        self.handle = handle
        self._entry = None

    @property
    def entry_func(self):
        """Get the entry function

        Returns
        -------
        f : Function
            The entry function if exist
        """
        if self._entry:
            return self._entry
        else:
            self._entry = self.get_function("__tvm_main__")
            return self._entry

    def get_function(self, name, query_imports=False):
        """Get function from the module.

        Parameters
        ----------
        name : str
            The name of the function

        query_imports : bool
            Whether also query modules imported by this module.

        Returns
        -------
        f : Function
            The result function.
        """
        ret_handle = FunctionHandle()
        check_call(_LIB.TVMModGetFunction(
            self.handle, c_str(name),
            ctypes.c_int(query_imports),
            ctypes.byref(ret_handle)))
        if not ret_handle.value:
            raise AttributeError(
                "Module has no function '%s'" %  name)
        return Function(ret_handle, False)

    def import_module(self, module):
        """Add module to the import list of current one.

        Parameters
        ----------
        module : Module
            The other module.
        """
        check_call(_LIB.TVMModImport(self.handle, module.handle))

    def precompile(self, func_name, ctx):
        """Add module to the import list of current one.

        Parameters
        ----------
        func_name : str
            The name of function to be precompiled.

        ctx : Context
            The context to be precompiled.
        """
        check_call(_LIB.TVMModPreCompile(
            self.handle, c_str(func_name), ctx))

    def __getitem__(self, name):
        if not isinstance(name, string_types):
            raise ValueError("Can only take string as function name")
        return self.get_function(name)

    def __del__(self):
        check_call(_LIB.TVMModFree(self.handle))

    def __call__(self, *args):
        if self._entry:
            return self._entry(*args)
        else:
            f = self.entry_func
            return f(*args)

_module_cls = None

def _return_module(x):
    """Return function"""
    handle = x.v_handle
    if not isinstance(handle, ModuleHandle):
        handle = ModuleHandle(handle)
    return _module_cls(handle)

def _handle_return_func(x):
    """Return function"""
    handle = x.v_handle
    if not isinstance(handle, FunctionHandle):
        handle = FunctionHandle(handle)
    return Function(handle, False)

# setup return handle for function type
RETURN_SWITCH[TypeCode.FUNC_HANDLE] = _handle_return_func
RETURN_SWITCH[TypeCode.MODULE_HANDLE] = _return_module
C_TO_PY_ARG_SWITCH[TypeCode.FUNC_HANDLE] = _wrap_arg_func(
    _handle_return_func, TypeCode.FUNC_HANDLE)
C_TO_PY_ARG_SWITCH[TypeCode.MODULE_HANDLE] = _wrap_arg_func(
    _return_module, TypeCode.MODULE_HANDLE)


def register_func(func_name, f=None, override=False):
    """Register global function

    Parameters
    ----------
    func_name : str or function
        The function name

    f : function, optional
        The function to be registered.

    override: boolean optional
        Whether override existing entry.

    Returns
    -------
    fregister : function
        Register function if f is not specified.

    Examples
    --------
    The following code registers my_packed_func as global function.
    Note that we simply get it back from global function table to invoke
    it from python side. However, we can also invoke the same function
    from C++ backend, or in the compiled TVM code.

    .. code-block:: python

      targs = (10, 10.0, "hello")
      @tvm.register_func
      def my_packed_func(*args):
          assert(tuple(args) == targs)
          return 10
      # Get it out from global function table
      f = tvm.get_global_func("my_packed_func")
      assert isinstance(f, tvm.nd.Function)
      y = f(*targs)
      assert y == 10
    """
    if callable(func_name):
        f = func_name
        func_name = f.__name__

    if not isinstance(func_name, str):
        raise ValueError("expect string function name")

    ioverride = ctypes.c_int(override)
    def register(myf):
        """internal register function"""
        if not isinstance(myf, Function):
            myf = convert_to_tvm_func(myf)
        check_call(_LIB.TVMFuncRegisterGlobal(
            c_str(func_name), myf.handle, ioverride))
    if f:
        register(f)
    else:
        return register


def get_global_func(name):
    """Get a global function by name

    Parameters
    ----------
    name : str
        The name of the global function

    Returns
    -------
    func : tvm.Function
        The function to be returned.
    """
    handle = FunctionHandle()
    check_call(_LIB.TVMFuncGetGlobal(c_str(name), ctypes.byref(handle)))
    return Function(handle, False)


def list_global_func_names():
    """Get list of global functions registered.

    Returns
    -------
    names : list
       List of global functions names.
    """
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.TVMFuncListGlobalNames(ctypes.byref(size),
                                           ctypes.byref(plist)))
    fnames = []
    for i in range(size.value):
        fnames.append(py_str(plist[i]))
    return fnames


def _get_api(f):
    flocal = f
    flocal.is_global = True
    def my_api_func(*args):
        """

        This is a type erased API that calls into Global PackedFunc.
        These APIs corresponds to functions registered from C++ backend
        and can be used as developer functions.

        args : list
          The positional arguments to the function call.

        Returns
        -------
        value : int, float, None, Node or Function
        The result of the API function call.
        """
        return flocal(*args)
    return my_api_func

def _init_api(namespace):
    """Initialize api for a given module name

    mod : str
       The name of the module.
    """
    module = sys.modules[namespace]
    assert namespace.startswith("tvm.")
    prefix = namespace[4:]

    for name in list_global_func_names():
        if prefix == "api":
            fname = name
            if name.startswith("_"):
                target_module = sys.modules["tvm._api_internal"]
            else:
                target_module = module
        else:
            if not name.startswith(prefix):
                continue
            fname = name[len(prefix)+1:]
            target_module = module

        if fname.find(".") != -1:
            continue
        f = get_global_func(name)
        ff = _get_api(f)
        ff.__name__ = fname
        ff.__doc__ = ("TVM PackedFunc %s. " % fname)
        setattr(target_module, ff.__name__, ff)


def _init_module_module(module_class):
    """Initialize the module."""
    global _module_cls
    _module_cls = module_class
