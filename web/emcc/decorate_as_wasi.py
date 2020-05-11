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
"""Decorate emcc generated js to a WASI compatible API."""

import sys

template_head = """
function EmccWASI() {
"""

template_tail = """
    this.Module = Module;
    this.start = Module.wasmLibraryProvider.start;
    this.imports = Module.wasmLibraryProvider.imports;
    this.wasiImport = this.imports["wasi_snapshot_preview1"];
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = EmccWASI;
}
"""

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage <file-in> <file-out>")
    result = template_head + open(sys.argv[1]).read() + template_tail
    with open(sys.argv[2], "w") as fo:
        fo.write(result)
