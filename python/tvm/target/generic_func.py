# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Generic function."""

import tvm._ffi

try:
    from decorator import decorate
except ImportError:
    # Allow decorator to be missing in runtime
    if not tvm._ffi.base._RUNTIME_ONLY:
        raise

from tvm.runtime import Object
from .target import Target
from . import _ffi_api


@tvm._ffi.register_object
class GenericFunc(Object):
    """GenericFunc node reference. This represents a generic function
    that may be specialized for different targets. When this object is
    called, a specialization is chosen based on the current target.

    Note
    ----
    Do not construct an instance of this object, it should only ever be
    used as a return value from calling into C++.
    """

    def __call__(self, *args):
        return _ffi_api.GenericFuncCallFunc(self, *args)

    def set_default(self, func, allow_override=False):
        """Set the default function to be used if no specializations match
        the current target.

        Parameters
        ----------
        func : function
            The default function

        allow_override : bool
            Whether to allow the current default to be overridden
        """
        _ffi_api.GenericFuncSetDefault(self, func, allow_override)

    def register(self, func, key_list, allow_override=False):
        """Register a specialization for this GenericFunc.

        Parameters
        ----------
        func : function
            The function to be registered.

        key : str or list of str
            The key to be registered.

        allow_override : bool, optional
            Whether to allow existing keys to be overridden.
        """
        key_list = [key_list] if isinstance(key_list, str) else key_list
        _ffi_api.GenericFuncRegisterFunc(self, func, key_list, allow_override)


def get_native_generic_func(name):
    """Get a generic function from the global registry. If no
    function is registered under the given name, a new generic
    function is created.

    Parameters
    ----------
    name : string
        The name of the generic function to get

    Returns
    -------
    func : GenericFunc
        The generic function for the given name
    """
    return _ffi_api.GenericFuncGetGlobal(name)


def override_native_generic_func(func_name):
    """Override a generic function defined in C++

    Generic function allows registration of further functions
    that can be dispatched on current target context.
    If no registered dispatch is matched, the fdefault will be called.

    Parameters
    ----------
    func_name : string
        The name of the generic func to be overridden

    Returns
    -------
    fgeneric : function
        A wrapped generic function.

    Example
    -------
    .. code-block:: python

      import tvm
      # wrap function as target generic
      @tvm.target.override_native_generic_func("my_func")
      def my_func(a):
          return a + 1
      # register specialization of my_func under target cuda
      @my_func.register("cuda")
      def my_func_cuda(a):
          return a + 2
      # displays 3, because my_func is called
      print(my_func(2))
      # displays 4, because my_func_cuda is called
      with tvm.target.cuda():
          print(my_func(2))
    """
    generic_func_node = get_native_generic_func(func_name)

    def fdecorate(fdefault):
        """Wrap a target generic function, overriding the previous
        default that was set for the generic function.

        Parameters
        ----------
        fdefault : function
            The default function.

        Returns
        -------
        fgeneric : function
            A wrapped generic function.

        """
        generic_func_node.set_default(fdefault, allow_override=True)

        def register(key, func=None, override=True):
            """Register function to be the dispatch function.

            Parameters
            ----------
            key : str or list of str
                The key to be registered.

            func : function
                The function to be registered.

            override : bool, optional
                Whether override existing registration.

            Returns
            -------
            The register function is necessary.
            """

            def _do_reg(myf):
                generic_func_node.register(myf, key, override)
                return myf

            if func:
                return _do_reg(func)
            return _do_reg

        def dispatch_func(func, *args, **kwargs):
            # pylint: disable=unused-argument
            """The wrapped dispath function"""
            if kwargs:
                raise RuntimeError(
                    "Keyword arguments cannot be used when invoking generic_func %s" % func_name
                )
            return generic_func_node(*args)

        fresult = decorate(fdefault, dispatch_func)
        fresult.fdefault = fdefault
        fresult.register = register
        fresult.generic_func_node = generic_func_node
        return fresult

    return fdecorate


def generic_func(fdefault):
    """Wrap a target generic function.

    Generic function allows registration of further functions
    that can be dispatched on current target context.
    If no registered dispatch is matched, the fdefault will be called.

    Parameters
    ----------
    fdefault : function
        The default function.

    Returns
    -------
    fgeneric : function
        A wrapped generic function.

    Example
    -------
    .. code-block:: python

      import tvm
      # wrap function as target generic
      @tvm.target.generic_func
      def my_func(a):
          return a + 1
      # register specialization of my_func under target cuda
      @my_func.register("cuda")
      def my_func_cuda(a):
          return a + 2
      # displays 3, because my_func is called
      print(my_func(2))
      # displays 4, because my_func_cuda is called
      with tvm.target.cuda():
          print(my_func(2))
    """
    dispatch_dict = {}
    func_name = fdefault.__name__

    def register(key, func=None, override=False):
        """Register function to be the dispatch function.

        Parameters
        ----------
        key : str or list of str
            The key to be registered.

        func : function
            The function to be registered.

        override : bool
            Whether override existing registration.

        Returns
        -------
        The register function is necessary.
        """

        def _do_reg(myf):
            key_list = [key] if isinstance(key, str) else key
            for k in key_list:
                if k in dispatch_dict and not override:
                    raise ValueError("Key is already registered for %s" % func_name)
                dispatch_dict[k] = myf
            return myf

        if func:
            return _do_reg(func)
        return _do_reg

    def dispatch_func(func, *args, **kwargs):
        """The wrapped dispath function"""
        target = Target.current()
        if target is None:
            return func(*args, **kwargs)
        for k in target.keys:
            if k in dispatch_dict:
                return dispatch_dict[k](*args, **kwargs)
        return func(*args, **kwargs)

    fdecorate = decorate(fdefault, dispatch_func)
    fdecorate.register = register
    fdecorate.fdefault = fdefault
    fdecorate.dispatch_dict = dispatch_dict
    return fdecorate
