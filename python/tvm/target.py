"""Target management API of TVM.

TVM's target string is in fomat ``<target_name> [-option=value]...``.

Note
----
The list of options include:

- **-device=<device name>**

   The device name.

- **-mtriple=<target triple>** or **-target**

   Specify the target triple, which is useful for cross
   compilation.

- **-mcpu=<cpuname>**

   Specify a specific chip in the current architecture to
   generate code for. By default this is infered from the
   target triple and autodetected to the current architecture.

- **-mattr=a1,+a2,-a3,...**

   Override or control specific attributes of the target,
   such as whether SIMD operations are enabled or not. The
   default set of attributes is set by the current CPU.

- **-system-lib**

   Build TVM system library module. System lib is a global module that contains
   self registered functions in program startup. User can get the module using
   :any:`tvm.module.system_lib`.
   It is useful in environments where dynamic loading api like dlopen is banned.
   The system lib will be available as long as the result code is linked by the program.

We can use :any:`tvm.target.create` to create a tvm.target.Target from the target string.
We can also use other specific function in this module to create specific targets.
"""
from __future__ import absolute_import

import warnings

from ._ffi.base import _LIB_NAME
from ._ffi.node import NodeBase, register_node
from . import _api_internal

try:
    from decorator import decorate
except ImportError as err_msg:
    # Allow decorator to be missing in runtime
    if _LIB_NAME != "libtvm_runtime.so":
        raise err_msg

def _merge_opts(opts, new_opts):
    """Helper function to merge options"""
    if isinstance(new_opts, str):
        new_opts = new_opts.split()
    if new_opts:
        opt_set = set(opts)
        new_opts = [opt for opt in new_opts if opt not in opt_set]
        return opts + new_opts
    return opts


@register_node
class Target(NodeBase):
    """Target device information, use through TVM API.

    Note
    ----
    Do not use class constructor, you can create target using the following functions

    - :any:`tvm.target.create` create target from string
    - :any:`tvm.target.arm_cpu` create arm_cpu target
    - :any:`tvm.target.cuda` create CUDA target
    - :any:`tvm.target.rocm` create ROCM target
    - :any:`tvm.target.mali` create Mali target
    - :any:`tvm.target.intel_graphics` create Intel Graphics target
    """
    def __new__(cls):
        # Always override new to enable class
        obj = NodeBase.__new__(cls)
        obj._keys = None
        obj._options = None
        obj._libs = None
        return obj

    @property
    def keys(self):
        if not self._keys:
            self._keys = [k.value for k in self.keys_array]
        return self._keys

    @property
    def options(self):
        if not self._options:
            self._options = [o.value for o in self.options_array]
        return self._options

    @property
    def libs(self):
        if not self._libs:
            self._libs = [l.value for l in self.libs_array]
        return self._libs

    @property
    def model(self):
        for opt in self.options_array:
            if opt.value.startswith('-model='):
                return opt.value[7:]
        return 'unknown'

    def __enter__(self):
        _api_internal._EnterTargetScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        _api_internal._ExitTargetScope()


@register_node
class GenericFunc(NodeBase):
    """GenericFunc node reference. This represents a generic function
    that may be specialized for different targets. When this object is
    called, a specialization is chosen based on the current target.

    Note
    ----
    Do not construct an instance of this object, it should only ever be
    used as a return value from calling into C++.
    """
    def __call__(self, *args):
        return _api_internal._GenericFuncCallFunc(self, *args)

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
        _api_internal._GenericFuncSetDefault(self, func, allow_override)

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
        _api_internal._GenericFuncRegisterFunc(self, func, key_list, allow_override)


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
    return _api_internal._GenericFuncGetGlobal(name)


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
            #pylint: disable=unused-argument
            """The wrapped dispath function"""
            if kwargs:
                raise RuntimeError(
                    "Keyword arguments cannot be used when invoking generic_func %s" % func_name)
            return generic_func_node(*args)
        fresult = decorate(fdefault, dispatch_func)
        fresult.fdefault = fdefault
        fresult.register = register
        return fresult
    return fdecorate

def generic_func(fdefault):
    """Wrap a target generic function.

    Generic function allows registeration of further functions
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
            Whether override existing registeration.

        Returns
        -------
        The register function is necessary.
        """
        def _do_reg(myf):
            key_list = [key] if isinstance(key, str) else key
            for k in key_list:
                if k in dispatch_dict and not override:
                    raise ValueError(
                        "Key is already registered for %s" % func_name)
                dispatch_dict[k] = myf
            return myf
        if func:
            return _do_reg(func)
        return _do_reg

    def dispatch_func(func, *args, **kwargs):
        """The wrapped dispath function"""
        target = current_target()
        if target is None:
            return func(*args, **kwargs)
        for k in target.keys:
            if k in dispatch_dict:
                return dispatch_dict[k](*args, **kwargs)
        return func(*args, **kwargs)
    fdecorate = decorate(fdefault, dispatch_func)
    fdecorate.register = register
    fdecorate.fdefault = fdefault
    return fdecorate


def cuda(model='unknown', options=None):
    """Returns a cuda target.

    Parameters
    ----------
    model: str
        The model of cuda device (e.g. 1080ti)
    options : str or list of str
        Additional options
    """
    opts = _merge_opts(['-model=%s' % model], options)
    return _api_internal._TargetCreate("cuda", *opts)


def rocm(model='unknown', options=None):
    """Returns a ROCM target.

    Parameters
    ----------
    model: str
        The model of this device
    options : str or list of str
        Additional options
    """
    opts = _merge_opts(["-model=%s" % model], options)
    return _api_internal._TargetCreate("rocm", *opts)


def mali(model='unknown', options=None):
    """Returns a ARM Mali GPU target.

    Parameters
    ----------
    model: str
        The model of this device
    options : str or list of str
        Additional options
    """
    opts = ["-device=mali", '-model=%s' % model]
    opts = _merge_opts(opts, options)
    return _api_internal._TargetCreate("opencl", *opts)


def intel_graphics(model='unknown', options=None):
    """Returns an Intel Graphics target.

    Parameters
    ----------
    model: str
        The model of this device
    options : str or list of str
        Additional options
    """
    opts = ["-device=intel_graphics", '-model=%s' % model]
    opts = _merge_opts(opts, options)
    return _api_internal._TargetCreate("opencl", *opts)


def opengl(options=None):
    """Returns a OpenGL target.

    Parameters
    ----------
    options : str or list of str
        Additional options
    """
    options = _merge_opts([], options)
    return _api_internal._TargetCreate("opengl", *options)


def arm_cpu(model='unknown', options=None):
    """Returns a ARM CPU target.
    This function will also download pre-tuned op parameters when there is none.

    Parameters
    ----------
    model: str
        SoC name or phone name of the arm board.
    options : str or list of str
        Additional options
    """
    trans_table = {
        "pixel2":    ["-model=snapdragon835", "-target=arm64-linux-android -mattr=+neon"],
        "mate10":    ["-model=kirin970", "-target=arm64-linux-android -mattr=+neon"],
        "mate10pro": ["-model=kirin970", "-target=arm64-linux-android -mattr=+neon"],
        "p20":       ["-model=kirin970", "-target=arm64-linux-android -mattr=+neon"],
        "p20pro":    ["-model=kirin970", "-target=arm64-linux-android -mattr=+neon"],
        "rasp3b":    ["-model=bcm2837", "-target=armv7l-linux-gnueabihf -mattr=+neon"],
        "rk3399":    ["-model=rk3399", "-target=aarch64-linux-gnu -mattr=+neon"],
        "pynq":      ["-model=pynq", "-target=armv7a-linux-eabi -mattr=+neon"],
        "ultra96":   ["-model=ultra96", "-target=aarch64-linux-gnu -mattr=+neon"],
    }
    pre_defined_opt = trans_table.get(model, ["-model=%s" % model])

    opts = ["-device=arm_cpu"] + pre_defined_opt
    opts = _merge_opts(opts, options)
    return _api_internal._TargetCreate("llvm", *opts)


def rasp(options=None):
    """Return a Raspberry 3b target.

    Parameters
    ----------
    options : str or list of str
        Additional options
    """
    warnings.warn('tvm.target.rasp() is going to be deprecated. '
                  'Please use tvm.target.arm_cpu("rasp3b")')
    return arm_cpu('rasp3b', options)


def create(target_str):
    """Get a target given target string.

    Parameters
    ----------
    target_str : str
        The target string.

    Returns
    -------
    target : Target
        The target object

    Note
    ----
    See the note on :any:`tvm.target` on target string format.
    """
    if isinstance(target_str, Target):
        return target_str
    if not isinstance(target_str, str):
        raise ValueError("target_str has to be string type")

    return _api_internal._TargetFromString(target_str)


def current_target(allow_none=True):
    """Returns the current target.

    Parameters
    ----------
    allow_none : bool
       Whether allow the current target to be none

    Raises
    ------
    ValueError if current target is not set.
    """
    return _api_internal._GetCurrentTarget(allow_none)
