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

"""Defines an Artifact subclass that describes a compiled static library."""

from tvm.contrib import util
from . import artifact
from . import compiler


class MicroLibrary(artifact.Artifact):
    """An Artifact that describes a compiled static library."""

    ARTIFACT_TYPE = "micro_library"

    @classmethod
    def from_unarchived(cls, base_dir, labelled_files, metadata):
        library_files = labelled_files["library_files"]
        del labelled_files["library_files"]

        debug_files = None
        if "debug_files" in labelled_files:
            debug_files = labelled_files["debug_files"]
            del labelled_files["debug_files"]

        return cls(
            base_dir,
            library_files,
            debug_files=debug_files,
            labelled_files=labelled_files,
            metadata=metadata,
        )

    def __init__(
        self, base_dir, library_files, debug_files=None, labelled_files=None, metadata=None
    ):
        labelled_files = {} if labelled_files is None else dict(labelled_files)
        metadata = {} if metadata is None else dict(metadata)
        labelled_files["library_files"] = library_files
        if debug_files is not None:
            labelled_files["debug_files"] = debug_files

        super(MicroLibrary, self).__init__(base_dir, labelled_files, metadata)

        self.library_files = library_files
        self.debug_file = debug_files


def create_micro_library(output, objects, options=None):
    """Create a MicroLibrary using the default compiler options.

    Parameters
    ----------
    output : str
      Path to the output file, expected to end in .tar.
    objects : List[str]
      Paths to the source files to include in the library.
    options : Optional[List[str]]
      If given, additional command-line flags for the compiler.
    """
    temp_dir = util.tempdir()
    comp = compiler.DefaultCompiler()
    output = temp_dir.relpath("micro-library.o")
    comp.library(output, objects, options=options)

    with open(output, "rb") as output_f:
        elf_data = output_f.read()

    # TODO(areusch): Define a mechanism to determine compiler and linker flags for each lib
    # enabled by the target str, and embed here.
    micro_lib = MicroLibrary("", elf_data, {"target": comp.target.str()})
    micro_lib.save(output)
