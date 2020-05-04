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
"""Utils to path."""
import os
from tvm._ffi import libinfo
from tvm.contrib import cc


def find_minrpc_server_libpath(server="posix_popen_server"):
    """Get the path of minrpc server libary.

    Parameters
    ----------
    server : str
        The kind of built in minrpc server.

    Returns
    -------
    path : str
        The path to the min server library.
    """
    curr_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.abspath(os.path.join(curr_dir, "..", "..", ".."))

    path = os.path.join(
        source_dir, "src", "runtime", "rpc", "minrpc", ("%s.cc" % server))

    candidates = [path]
    if not os.path.isfile(path):
        raise RuntimeError("Cannot find minserver %s, in candidates %s" % (server, candidates))
    return path


def with_minrpc(compile_func,
                server="posix_popen_server",
                runtime="libtvm"):
    """Attach the compiler function with minrpc related options.

    Parameters
    ----------
    compile_func : Union[str, Callable[[str, str, Optional[str]], None]]
        The compilation function to decorate.

    server : str
        The server type.

    runtime : str
        The runtime library.

    Returns
    -------
    fcompile : function
        The return compilation.
    """
    server_path = find_minrpc_server_libpath(server)
    runtime_path = libinfo.find_lib_path(
        [runtime, runtime + ".so", runtime + ".dylib"])[0]

    runtime_dir = os.path.abspath(os.path.dirname(runtime_path))
    options = ["-std=c++14"]
    # Make sure the rpath to the libtvm is set so we can do local tests.
    # Note that however, this approach won't work on remote.
    # Always recommend to to link statically.
    options += ["-Wl,-rpath=" + runtime_dir]
    options += ["-I" + path for path in libinfo.find_include_path()]
    fcompile = cc.cross_compiler(
        compile_func,
        options=options,
        add_files=[server_path, runtime_path])
    fcompile.__name__ = "with_minrpc"
    fcompile.need_system_lib = True
    return fcompile
