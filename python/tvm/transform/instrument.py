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
"""Concrete tools for inspecting and timing IR transformations."""

import re
import shutil
from pathlib import Path

from . import _ffi_instrument_api
from .core import PassInstrument, pass_instrument


class PassTimingInstrument(PassInstrument):
    """A wrapper to create a passes time instrument that implemented in C++"""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_instrument_api.MakePassTimingInstrument)

    @staticmethod
    def render():
        """Retrieve rendered time profile result
        Returns
        -------
        string : string
            The rendered string result of time profiles

        Examples
        --------

        .. code-block:: python

            timing_inst = PassTimingInstrument()
            with tvm.transform.PassContext(instruments=[timing_inst]):
                relax_mod = relax.transform.FuseOps()(relax_mod)
                # before exiting the context, get profile results.
                profiles = timing_inst.render()
        """
        return _ffi_instrument_api.RenderTimePassProfiles()


@pass_instrument
class PassPrintingInstrument:
    """A pass instrument to print if before or
    print ir after each element of a named pass."""

    def __init__(self, print_before_pass_names, print_after_pass_names):
        self.print_before_pass_names = print_before_pass_names
        self.print_after_pass_names = print_after_pass_names

    def run_before_pass(self, mod, pass_info):
        if pass_info.name in self.print_before_pass_names:
            print(f"Print IR before: {pass_info.name}\n{mod}\n\n")

    def run_after_pass(self, mod, pass_info):
        if pass_info.name in self.print_after_pass_names:
            print(f"Print IR after: {pass_info.name}\n{mod}\n\n")


@pass_instrument
class PrintAfterAll:
    """Print the name of the pass, the IR, only after passes execute."""

    def run_after_pass(self, mod, info):
        print(f"After Running Pass: {info}")
        print(mod)


@pass_instrument
class PrintBeforeAll:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print(f"Before Running Pass: {info}")
        print(mod)


@pass_instrument
class DumpIR:
    """Dump the IR after the pass runs."""

    def __init__(self, dump_dir: Path | str, refresh: bool = False):
        if isinstance(dump_dir, Path):
            self.dump_dir = dump_dir
        else:
            self.dump_dir = Path(dump_dir)
        self.counter = 0
        if refresh and self.dump_dir.is_dir():
            self._safe_remove_dump_dir()

    def _safe_remove_dump_dir(self):
        """Remove dump directory only if it contains only dumped IR files."""
        # Pattern for dumped files: {counter:03d}_{pass_name}.py
        dump_pattern = re.compile(r"^\d{3}_.*\.py$")

        # Check all files in the directory
        for item in self.dump_dir.iterdir():
            # If there's a subdirectory or a file that doesn't match the pattern, abort
            if item.is_dir() or not dump_pattern.match(item.name):
                print(
                    f"WARNING: Skipping removal of {self.dump_dir} as it contains "
                    f"non-dumped files or directories. Please clean it manually."
                )
                return

        # Safe to remove - only contains dumped files
        try:
            shutil.rmtree(self.dump_dir)
        except OSError as e:
            print(f"WARNING: Failed to remove directory {self.dump_dir}: {e}")

    def run_after_pass(self, mod, info):
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        try:
            sanitized_pass_name = re.sub(r'[<>:"/\\|?*]', "_", info.name)
            with open(self.dump_dir / f"{self.counter:03d}_{sanitized_pass_name}.py", "w") as f:
                f.write(mod.script())
        except Exception:  # pylint: disable=broad-exception-caught
            print(f"WARNING: Failed to dump IR for pass {info.name}")
        finally:
            self.counter += 1
