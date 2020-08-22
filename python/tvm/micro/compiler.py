import abc
import glob
import os
import re
import typing

from tvm.contrib import binutil
import tvm.target
from . import build
from . import class_factory
from . import debugger
from . import transport


class DetectTargetError(Exception):
  """Raised when no target comment was detected in the sources given."""


class NoDefaultToolchainMatchedError(Exception):
  """Raised when no default toolchain matches the target string."""


class Compiler(metaclass=abc.ABCMeta):
  """The compiler abstraction used with micro TVM."""

  TVM_TARGET_RE = re.compile(r'^// tvm target: (.*)$')

  @classmethod
  def _TargetFromSources(cls, sources):
    """Determine the target used to generate the given source files.

    Parameters
    ----------
    sources : List[str]
        The paths to source files to analyze.

    Returns
    -------
    tvm.target.Target :
        A Target instance reconstructed from the target string listed in the source files.
    """
    target_strs = set()

    for obj in sources:
      with open(obj) as obj_f:
        for line in obj_f:
          m = cls.TVM_TARGET_RE.match(line)
          if m:
            target_strs.add(m.group(1))

    if len(target_strs) != 1:
      raise DetectTargetError(
        'autodetecting cross-compiler: could not extract TVM target from C source; regex '
        f'{cls.TVM_TARGET_RE.pattern} does not match any line in sources: {", ".join(sources)}')

    target_str = next(iter(target_strs))
    return tvm.target.create(target_str)

  # Maps regexes identifying CPUs to the default toolchain prefix for that CPU.
  TOOLCHAIN_PREFIX_BY_CPU_REGEX = {
    r'cortex-[am].*': 'arm-none-eabi-',
    'x86[_-]64': '',
    'native': '',
  }

  def _AutodetectToolchainPrefix(self, target):
    matches = []
    for regex, prefix in self.TOOLCHAIN_PREFIX_BY_CPU_REGEX.items():
      if re.match(regex, target.attrs['mcpu']):
        matches.append(prefix)

    if matches:
      if len(matches) != 1:
        raise NoDefaultToolchainMatchedError(
          f'{opt} matched more than 1 default toolchain prefix: {", ".join(matches)}. Specify '
          f'cc.cross_compiler to create_micro_library()')

      return prefix

    raise NoDefaultToolchainMatchedError(f'target {str(target)} did not match any default toolchains')

  def _DefaultsFromTarget(self, target):
    """Determine the default compiler options from the target specified.

    Parameters
    ----------
    target : tvm.target.Target

    Returns
    -------
    List[str] :
        Default options used the configure the compiler for that target.
    """
    opts = []
    # TODO use march for arm(https://gcc.gnu.org/onlinedocs/gcc/ARM-Options.html)?
    if target.attrs.get('mcpu'):
      # opts.append(f'-mcpu={target.attrs["mcpu"]}')
      opts.append(f'-march={target.attrs["mcpu"]}')
    if target.attrs.get('mfpu'):
      opts.append(f'-mfpu={target.attrs["mfpu"]}')

    return opts

  # @classmethod
  # def _MergeOptions(cls, default, override, path=''):
  #   if isinstance(override, (str, bytes, int, float)):
  #     if isinstance(default, (str, bytes, int, float, None)):
  #       return override

  #     raise OptionOverrideError(
  #         f'overrriding {path}: trying to override {default!r} with primitive {override!r}')


  #   elif isinstance(override, (list, tuple)):
  #     if default is None:
  #       return override

  #     if not isinstance(default, (list, tuple)):
  #       raise OptionOverrideError(
  #           f'overrriding {path}: trying to override {default!r} with non-list {override!r}')

  #     new_list = list(default)
  #     for

  #   options = default_options
  #   for key, values in extensions.items():
  #     if isinstance(values, list):

  #     if isinstance(

  @abc.abstractmethod
  def Library(self, output, objects, options=None):
    """Build a library from the given source files.

    Parameters
    ----------
    output : str
        The path to the library that should be created. The containing directory
        is guaranteed to be empty and should be the base_dir for the returned
        Artifact.
    objects : List[str]
        A list of paths to source files that should be compiled.
    options : Optional[List[str]]
        If given, additional command-line flags to pass to the compiler.

    Returns
    -------
    MicroLibrary :
        The compiled library, as a MicroLibrary instance.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def Binary(self, output, objects, options=None, link_main=True, main_options=None):
    """Link a binary from the given object and/or source files.

    Parameters
    ----------
    output : str
        The path to the binary that should be created. The containing directory
        is guaranteed to be empty and should be the base_dir for the returned
        Artifact.
    objects : List[MicroLibrary]
        A list of paths to source files or libraries that should be compiled. The final binary should
        be statically-linked.
    options: Optional[List[str]]
        If given, additional command-line flags to pass to the compiler.
    link_main: Optional[bool]
        True if the standard main entry point for this Compiler should be included in the binary.
        False if a main entry point is provided in one of `objects`.
    main_options: Optional[List[str]]
        If given, additional command-line flags to pass to the compiler when compiling the main()
        library. In some cases, the main() may be compiled directly into the final binary along with
        `objects` for logistical reasons. In those cases, specifying main_options is an error and
        ValueError will be raised.

    Returns
    -------
    MicroBinary :
        The compiled binary, as a MicroBinary instance.
    """
    raise NotImplementedError()

  @property
  def flasher_factory(self):
    """Produce a FlasherFactory that can produce a Flasher instance suitable to be returned from Flasher()."""
    raise NotImplementedError("The Compiler base class doesn't define a flasher.")

  def Flasher(self, **kw):
    """Return a Flasher that can be used to program a produced MicroBinary onto the target."""
    return self.flasher_factory.override_kw(**kw).instantiate()


class IncompatibleTargetError(Exception):
  """Raised when source files specify a target that differs from the compiler target."""


class DefaultCompiler(Compiler):

  def __init__(self, target=None, *args, **kw):
    super(DefaultCompiler, self).__init__()
    self.target = target

  def Library(self, output, sources, options=None):
    options = options if options is not None else {}
    try:
      target = self._TargetFromSources(sources)
    except DetectTargetError:
      assert self.target is not None, (
        "Must specify target= to constructor when compiling sources which don't specify a target")

      target = self.target

    if self.target is not None and str(self.target) != str(target):
      raise IncompatibleTargetError(
        f'auto-detected target {target} differs from configured {self.target}')

    prefix = self._AutodetectToolchainPrefix(target)
    outputs = []
    for src in sources:
      src_base, src_ext = os.path.splitext(os.path.basename(src))

      compiler_name = {'.c': 'gcc', '.cc': 'g++', '.cpp': 'g++'}[src_ext]
      args = [prefix + compiler_name, '-g']
      args.extend(self._DefaultsFromTarget(target))

      args.extend(options.get(f'{src_ext[1:]}flags', []))

      for d in options.get('include_dirs', []):
        args.extend(['-I', d])

      output_filename = f'{src_base}.o'
      output_abspath = os.path.join(output, output_filename)
      binutil.run_cmd(args + ['-c', '-o', output_abspath, src])
      outputs.append(output_abspath)

    output_filename = f'{os.path.basename(output)}.a'
    output_abspath = os.path.join(output, output_filename)
    binutil.run_cmd([prefix + 'ar', '-r', output_abspath] + outputs)
    binutil.run_cmd([prefix + 'ranlib', output_abspath])

    return tvm.micro.MicroLibrary(output, [output_filename])

  def Binary(self, output, objects, options=None, link_main=True, main_options=None):
    assert self.target is not None, (
      'must specify target= to constructor, or compile sources which specify the target first')

    args = [self._AutodetectToolchainPrefix(self.target) + 'g++']
    args.extend(self._DefaultsFromTarget(self.target))
    if options is not None:
      args.extend(options.get('ldflags', []))

      for d in options.get('include_dirs', []):
        args.extend(['-I', d])

    output_filename = os.path.basename(output)
    output_abspath = os.path.join(output, output_filename)
    args.extend(['-g', '-o', output_abspath])

    if link_main:
      host_main_srcs = glob.glob(os.path.join(build.CRT_ROOT_DIR, 'host', '*.cc'))
      if main_options:
        main_lib = self.Library(os.path.join(output, 'host'), host_main_srcs, main_options)
        for lib_name in main_lib.library_files:
          args.append(main_lib.abspath(lib_name))
      else:
        args.extend(host_main_srcs)

    # for obj in objects:
    #   for lib_name in obj.library_files:
    #     args.append(obj.abspath(lib_name))
    for obj in objects:
      for lib_name in obj.library_files:
        args.append(obj.abspath(lib_name))

    binutil.run_cmd(args)
    return tvm.micro.MicroBinary(output, output_filename, [])

  @property
  def flasher_factory(self, **kw):
    return FlasherFactory(HostFlasher, [], kw)


class Flasher(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def Flash(self, micro_binary):
    """Flash a binary onto the device.

    Parameters
    ----------
    micro_binary : MicroBinary
        A MicroBinary instance.

    Returns
    -------
    transport.TransportContextManager :
        A ContextManager that can be used to create and tear down an RPC transport layer between
        this TVM instance and the newly-flashed binary.
    """
    raise NotImplementedError()


class FlasherFactory(class_factory.ClassFactory):

  SUPERCLASS = Flasher


class HostFlasher(Flasher):

  def __init__(self, debug=False):
    self.debug = debug

  def Flash(self, micro_binary):
    print('flash', self.debug)
    if self.debug:
      gdb_wrapper = debugger.GdbTransportDebugger([micro_binary.abspath(micro_binary.binary_file)])
      return transport.DebugWrapperTransport(
        debugger=gdb_wrapper, transport=gdb_wrapper.Transport())
    else:
      return transport.SubprocessTransport([micro_binary.abspath(micro_binary.binary_file)])
