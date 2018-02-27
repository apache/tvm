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

    tempdir = util.tempdir()

    # Save temporary file "identity_static.bc".
    path_obj = tempdir.relpath("identity_static.bc")
    f.save(path_obj)

    # Write device module content into C++ array.
    path_cc = tempdir.relpath("devc.cc")
    with open(path_cc, "w") as fp:
        is_syslib = True
        fp.write(tvm.module._PackImportsToC(f, is_syslib))

    # Compile and link host and dev modules into a single JS file.
    files = [path_obj, path_cc]
    path_dso = "identity_static.js"
    emscripten.create_js(path_dso, files, side_module=False, options=[
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
