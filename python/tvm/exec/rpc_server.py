"""Start an RPC server"""
from __future__ import absolute_import

import logging
import argparse
import sys, os
import ctypes
from ..contrib import rpc

def find_lib_path(name):
    """Find dynamic library."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    base_path = os.path.join(curr_path, "../../../")
    apps_path = os.path.join(base_path, "apps/graph_executor/lib/")
    api_path = os.path.join(base_path, 'lib/')
    cmake_build_path = os.path.join(base_path, 'build/Release/')
    dll_path = [curr_path, base_path, apps_path, api_path, cmake_build_path]
    if os.name == 'nt':
        vs_configuration = 'Release'
        if platform.architecture()[0] == '64bit':
            dll_path.append(os.path.join(base_path, 'build', vs_configuration))
            dll_path.append(os.path.join(base_path, 'windows/x64', vs_configuration))
        else:
            dll_path.append(os.path.join(base_path, 'build', vs_configuration))
            dll_path.append(os.path.join(base_path, 'windows', vs_configuration))
    elif os.name == "posix" and os.environ.get('LD_LIBRARY_PATH', None):
        dll_path.extend([p.strip() for p in os.environ['LD_LIBRARY_PATH'].split(":")])
    dll_path = [os.path.abspath(x) for x in dll_path]
    lib_dll_path = [os.path.join(p, name) for p in dll_path]

    # try to find lib_dll_path
    lib_found = [p for p in lib_dll_path if os.path.exists(p) and os.path.isfile(p)]
    if not lib_found:
        raise RuntimeError('Cannot find the files.\n' +
                           'List of candidates:\n' +
                           str('\n'.join(lib_dll_path)))
    return lib_found


def main():
    """Main funciton"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='the hostname of the server')
    parser.add_argument('--port', type=int, default=9090,
                        help='The port of the PRC')
    parser.add_argument('--port_end', type=int, default=9199,
                        help='The end search port of the PRC')
    parser.add_argument('--with_executor', type=bool, default=False,
                        help="Whether to load executor runtime")
    args = parser.parse_args()

    if args.with_executor:
        lib_path = find_lib_path('libtvm_graph_exec.so')
        ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)

    logging.basicConfig(level=logging.INFO)
    server = rpc.Server(args.host, args.port, args.port_end)
    server.proc.join()

if __name__ == "__main__":
    main()
