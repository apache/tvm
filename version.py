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

"""
This is the global script that set the version information of TVM.
This script runs and update all the locations that related to versions

List of affected files:
- tvm-root/python/tvm/_ffi/libinfo.py
- tvm-root/include/tvm/runtime/c_runtime_api.h
- tvm-root/web/tvm_runtime.js
- tvm-root/conda/tvm/meta.yaml
- tvm-root/conda/tvm-libs/meta.yaml
"""
import os
import re
# current version
# We use the version of the incoming release for code
# that is under development
__version__ = "0.6.dev"

# Implementations
def update(file_name, pattern, repl):
    update = []
    hit_counter = 0
    need_update = False
    for l in open(file_name):
        result = re.findall(pattern, l)
        if result:
            assert len(result) == 1
            hit_counter += 1
            if result[0] != repl:
                l = re.sub(pattern, repl, l)
                need_update = True
                print("%s: %s->%s" % (file_name, result[0], repl))
            else:
                print("%s: version is already %s" % (file_name, repl))

        update.append(l)
    if hit_counter != 1:
        raise RuntimeError("Cannot find version in %s" % file_name)

    if need_update:
        with open(file_name, "w") as output_file:
            for l in update:
                output_file.write(l)


def main():
    proj_root = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    # python path
    update(os.path.join(proj_root, "python", "tvm", "_ffi", "libinfo.py"),
           r"(?<=__version__ = \")[.0-9a-z]+", __version__)
    # C++ header
    update(os.path.join(proj_root, "include", "tvm", "runtime", "c_runtime_api.h"),
           "(?<=TVM_VERSION \")[.0-9a-z]+", __version__)
    # conda
    for path in ["tvm", "tvm-libs"]:
        update(os.path.join(proj_root, "conda", path, "meta.yaml"),
               "(?<=version = \")[.0-9a-z]+", __version__)
    # web
    update(os.path.join(proj_root, "web", "tvm_runtime.js"),
           "(?<=@version )[.0-9a-z]+", __version__)

if __name__ == "__main__":
    main()
