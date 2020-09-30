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

"""Defines interfaces and default implementations for compiling and flashing code."""

import abc
import glob
import os
import re

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

    TVM_TARGET_RE = re.compile(r"^// tvm target: (.*)$")

    @classmethod
    def _target_from_sources(cls, sources):
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
                "autodetecting cross-compiler: could not extract TVM target from C source; regex "
                f"{cls.TVM_TARGET_RE.pattern} does not match any line in sources: "
                f'{", ".join(sources)}'
            )

        target_str = next(iter(target_strs))
        return tvm.target.create(target_str)

    # Maps regexes identifying CPUs to the default toolchain prefix for that CPU.
    TOOLCHAIN_PREFIX_BY_CPU_REGEX = {
        r"cortex-[am].*": "arm-none-eabi-",
        "x86[_-]64": "",
        "native": "",
    }

    def _autodetect_toolchain_prefix(self, target):
        matches = []
        for regex, prefix in self.TOOLCHAIN_PREFIX_BY_CPU_REGEX.items():
            if re.match(regex, target.attrs["mcpu"]):
                matches.append(prefix)

        if matches:
            if len(matches) != 1:
                raise NoDefaultToolchainMatchedError(
                    f'{opt} matched more than 1 default toolchain prefix: {", ".join(matches)}. '
                    "Specify cc.cross_compiler to create_micro_library()"
                )

            return matches[0]

        raise NoDefaultToolchainMatchedError(
            f"target {str(target)} did not match any default toolchains"
        )

    def _defaults_from_target(self, target):
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
        if target.attrs.get("mcpu"):
            opts.append(f'-march={target.attrs["mcpu"]}')
        if target.attrs.get("mfpu"):
            opts.append(f'-mfpu={target.attrs["mfpu"]}')

        return opts

    @abc.abstractmethod
    def library(self, output, sources, options=None):
        """Build a library from the given source files.

        Parameters
        ----------
        output : str
            The path to the library that should be created. The containing directory
            is guaranteed to be empty and should be the base_dir for the returned
            Artifact.
        sources : List[str]
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
    def binary(self, output, objects, options=None, link_main=True, main_options=None):
        """Link a binary from the given object and/or source files.

        Parameters
        ----------
        output : str
            The path to the binary that should be created. The containing directory
            is guaranteed to be empty and should be the base_dir for the returned
            Artifact.
        objects : List[MicroLibrary]
            A list of paths to source files or libraries that should be compiled. The final binary
            should be statically-linked.
        options: Optional[List[str]]
            If given, additional command-line flags to pass to the compiler.
        link_main: Optional[bool]
            True if the standard main entry point for this Compiler should be included in the
            binary. False if a main entry point is provided in one of `objects`.
        main_options: Optional[List[str]]
            If given, additional command-line flags to pass to the compiler when compiling the
            main() library. In some cases, the main() may be compiled directly into the final binary
            along with `objects` for logistical reasons. In those cases, specifying main_options is
            an error and ValueError will be raised.

        Returns
        -------
        MicroBinary :
            The compiled binary, as a MicroBinary instance.
        """
        raise NotImplementedError()

    @property
    def flasher_factory(self):
        """Produce a FlasherFactory for a Flasher instance suitable for this Compiler."""
        raise NotImplementedError("The Compiler base class doesn't define a flasher.")

    def flasher(self, **kw):
        """Return a Flasher that can be used to program a produced MicroBinary onto the target."""
        return self.flasher_factory.override_kw(**kw).instantiate()


class IncompatibleTargetError(Exception):
    """Raised when source files specify a target that differs from the compiler target."""


class DefaultCompiler(Compiler):
    """A Compiler implementation that attempts to use the system-installed GCC."""

    def __init__(self, target=None):
        super(DefaultCompiler, self).__init__()
        self.target = target
        if isinstance(target, str):
            self.target = tvm.target.create(target)

    def library(self, output, sources, options=None):
        options = options if options is not None else {}
        try:
            target = self._target_from_sources(sources)
        except DetectTargetError:
            assert self.target is not None, (
                "Must specify target= to constructor when compiling sources which don't specify a "
                "target"
            )

            target = self.target

        if self.target is not None and str(self.target) != str(target):
            raise IncompatibleTargetError(
                f"auto-detected target {target} differs from configured {self.target}"
            )

        prefix = self._autodetect_toolchain_prefix(target)
        outputs = []
        for src in sources:
            src_base, src_ext = os.path.splitext(os.path.basename(src))

            compiler_name = {".c": "gcc", ".cc": "g++", ".cpp": "g++"}[src_ext]
            args = [prefix + compiler_name, "-g"]
            args.extend(self._defaults_from_target(target))

            args.extend(options.get(f"{src_ext[1:]}flags", []))

            for include_dir in options.get("include_dirs", []):
                args.extend(["-I", include_dir])

            output_filename = f"{src_base}.o"
            output_abspath = os.path.join(output, output_filename)
            binutil.run_cmd(args + ["-c", "-o", output_abspath, src])
            outputs.append(output_abspath)

        output_filename = f"{os.path.basename(output)}.a"
        output_abspath = os.path.join(output, output_filename)
        binutil.run_cmd([prefix + "ar", "-r", output_abspath] + outputs)
        binutil.run_cmd([prefix + "ranlib", output_abspath])

        return tvm.micro.MicroLibrary(output, [output_filename])

    def binary(self, output, objects, options=None, link_main=True, main_options=None):
        assert self.target is not None, (
            "must specify target= to constructor, or compile sources which specify the target "
            "first"
        )

        args = [self._autodetect_toolchain_prefix(self.target) + "g++"]
        args.extend(self._defaults_from_target(self.target))
        if options is not None:
            args.extend(options.get("ldflags", []))

            for include_dir in options.get("include_dirs", []):
                args.extend(["-I", include_dir])

        output_filename = os.path.basename(output)
        output_abspath = os.path.join(output, output_filename)
        args.extend(["-g", "-o", output_abspath])

        if link_main:
            host_main_srcs = glob.glob(os.path.join(build.CRT_ROOT_DIR, "host", "*.cc"))
            if main_options:
                main_lib = self.library(os.path.join(output, "host"), host_main_srcs, main_options)
                for lib_name in main_lib.library_files:
                    args.append(main_lib.abspath(lib_name))
            else:
                args.extend(host_main_srcs)

        for obj in objects:
            for lib_name in obj.library_files:
                args.append(obj.abspath(lib_name))

        binutil.run_cmd(args)
        return tvm.micro.MicroBinary(output, output_filename, [])

    @property
    def flasher_factory(self):
        return FlasherFactory(HostFlasher, [], {})


class Flasher(metaclass=abc.ABCMeta):
    """An interface for flashing binaries and returning a transport factory."""

    @abc.abstractmethod
    def flash(self, micro_binary):
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
    """A ClassFactory for Flasher instances."""

    SUPERCLASS = Flasher


class HostFlasher(Flasher):
    """A Flasher implementation that spawns a subprocess on the host."""

    def __init__(self, debug=False):
        self.debug = debug

    def flash(self, micro_binary):
        if self.debug:
            gdb_wrapper = debugger.GdbTransportDebugger(
                [micro_binary.abspath(micro_binary.binary_file)]
            )
            return transport.DebugWrapperTransport(
                debugger=gdb_wrapper, transport=gdb_wrapper.Transport()
            )

        return transport.SubprocessTransport([micro_binary.abspath(micro_binary.binary_file)])
