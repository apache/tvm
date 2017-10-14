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
        return opts + new_opts
    return opts


class Target(object):
    """Target device information, use through TVM API.

    Parameters
    ----------
    target_name : {"llvm", "cuda", "opencl", "metal", "rocm", "stackvm", "ext_dev"}
        The major target name.

    options : list of str, optional
        Additional arguments appended to the target.

    Note
    ----
    Do not use class constructor, you can create target using the following functions

    - :any:`tvm.target.create` create target from string
    - :any:`tvm.target.rasp` create raspberry pi target
    - :any:`tvm.target.cuda` create CUDA target
    - :any:`tvm.target.rocm` create ROCM target
    """
    current = None

    def __init__(self,
                 target_name,
                 options=None):
        self.target_name = target_name
        self.options = _merge_opts([], options)
        self.device_name = ""
        # Parse device option
        for item in self.options:
            if item.startswith("-device="):
                self.device_name = item.split("=")[1]
        # Target query searchs device name first
        if self.device_name:
            self.keys = (self.device_name,)
        else:
            self.keys = ()
        # Target configuration handling
        self.thread_warp_size = 1
        if target_name in ("llvm", ):
            self.keys += ("cpu",)
        elif target_name in ("cuda", "nvptx"):
            self.keys += ("cuda", "gpu")
            self.max_num_threads = 512
            self.thread_warp_size = 32
        elif target_name in ("rocm", "opencl"):
            # For now assume rocm schedule for opencl
            self.keys += ("rocm", "gpu")
            self.max_num_threads = 256
        elif target_name in ("metal",):
            self.keys += ("gpu",)
            self.max_num_threads = 256
        elif target_name in ("stackvm", "ext_dev"):
            # Do not now class for stacvm or ext_dev
            pass
        else:
            raise ValueError("Unknown target name %s" % target_name)

    def __str__(self):
        return " ".join([self.target_name] + self.options)

    def __repr__(self):
        return self.__str__()

    def __enter__(self):
        self._old_target = Target.current
        if self._old_target is not None and str(self) != str(self._old_target):
            warnings.warn(
                "Override target '%s' with new target scope '%s'" % (
                    self._old_target, self))
        Target.current = self
        return self

    def __exit__(self, ptype, value, trace):
        Target.current = self._old_target


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
            return _do_reg(myf)
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
    return fdecorate


def cuda(options=None):
    """Returns a cuda target.

    Parameters
    ----------
    options : list of str
        Additional options
    """
    return Target("cuda", options)


def rocm(options=None):
    """Returns a ROCM target.

    Parameters
    ----------
    options : list of str
        Additional options
    """
    return Target("rocm", options)


def rasp(options=None):
    """Returns a rasp target.

    Parameters
    ----------
    options : list of str
        Additional options
    """
    opts = ["-device=rasp",
            "-mtriple=armv7l-none-linux-gnueabihf",
            "-mcpu=cortex-a53",
            "-mattr=+neon"]
    opts = _merge_opts(opts, options)
    return Target("llvm", opts)


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
    arr = target_str.split()
    # Parse device option
    device_name = ""
    for item in arr[1:]:
        if item.startswith("-device="):
            device_name = item.split("=")[1]
    if device_name == "rasp":
        return rasp(arr[1:])
    return Target(arr[0], arr[1:])


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
    if Target.current:
        return Target.current
    if not allow_none:
        raise RuntimeError(
            "Requires a current target in generic function, but it is not set. "
            "Please set it using `with TargetObject:`")
    return Target.current
