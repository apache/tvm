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

"""Defines an Artifact implementation for representing compiled micro TVM binaries."""

from . import artifact


class MicroBinary(artifact.Artifact):
    """An Artifact that describes a compiled binary."""

    ARTIFACT_TYPE = "micro_binary"

    @classmethod
    def from_unarchived(cls, base_dir, labelled_files, metadata):
        binary_file = labelled_files["binary_file"][0]
        del labelled_files["binary_file"]

        debug_files = None
        if "debug_files" in labelled_files:
            debug_files = labelled_files["debug_files"]
            del labelled_files["debug_files"]

        return cls(
            base_dir,
            binary_file,
            debug_files=debug_files,
            labelled_files=labelled_files,
            metadata=metadata,
        )

    def __init__(self, base_dir, binary_file, debug_files=None, labelled_files=None, metadata=None):
        labelled_files = {} if labelled_files is None else dict(labelled_files)
        metadata = {} if metadata is None else dict(metadata)
        labelled_files["binary_file"] = [binary_file]
        if debug_files is not None:
            labelled_files["debug_files"] = debug_files

        super(MicroBinary, self).__init__(base_dir, labelled_files, metadata)

        self.binary_file = binary_file
        self.debug_files = debug_files
