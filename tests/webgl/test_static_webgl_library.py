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
"""Create a static WebGL library and run it in the browser."""

from __future__ import absolute_import, print_function

import os, shutil, SimpleHTTPServer, SocketServer
import tvm
from tvm.contrib import emscripten, util
import numpy as np

def try_static_webgl_library():
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

    # Change to lib/ which contains "libtvm_runtime.bc".
    os.chdir(os.path.join(curr_path, "../../lib"))

    # Create OpenGL module.
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A', dtype="float")
    B = tvm.compute((n,), lambda *i: A[i], name="B")

    s = tvm.create_schedule(B.op)
    s[B].opengl()

    target_host = "llvm -target=asmjs-unknown-emscripten -system-lib"
    f = tvm.build(s, [A, B], name="identity", target="opengl",
                  target_host=target_host)

    # Create a JS library that contains both the module and the tvm runtime.
    path_dso = "identity_static.js"
    f.export_library(path_dso, emscripten.create_js, options=[
        "-s", "USE_GLFW=3",
        "-s", "USE_WEBGL2=1",
        "-lglfw",
    ])

    # Create "tvm_runtime.js" and "identity_static.html" in lib/
    shutil.copyfile(os.path.join(curr_path, "../../web/tvm_runtime.js"),
                    "tvm_runtime.js")
    shutil.copyfile(os.path.join(curr_path, "test_static_webgl_library.html"),
                    "identity_static.html")

    port = 8080
    handler = SimpleHTTPServer.SimpleHTTPRequestHandler
    httpd = SocketServer.TCPServer(("", port), handler)
    print("Please open http://localhost:" + str(port) + "/identity_static.html")
    httpd.serve_forever()

if __name__ == "__main__":
    try_static_webgl_library()
