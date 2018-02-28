"""Start an RPC server"""
from __future__ import absolute_import

import logging
import argparse
import os
import ctypes
from ..contrib import rpc
from .._ffi.libinfo import find_lib_path

def main():
    """Main funciton"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='the hostname of the server')
    parser.add_argument('--port', type=int, default=9090,
                        help='The port of the PRC')
    parser.add_argument('--port-end', type=int, default=9199,
                        help='The end search port of the PRC')
    parser.add_argument('--with-executor', type=bool, default=False,
                        help="Whether to load executor runtime")
    parser.add_argument('--load-library', type=str, default="",
                        help="Additional library to load")
    parser.add_argument('--exclusive', action='store_true',
                        help="If this is enabled, the server will kill old connection"
                             "when new connection comes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    load_library = [lib for lib in args.load_library.split(":") if len(lib) != 0]
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    apps_path = os.path.join(curr_path, "../../../apps/graph_executor/lib/")
    libs = []
    if args.with_executor:
        load_library += ["libtvm_graph_exec.so"]
    for file_name in load_library:
        file_name = find_lib_path(file_name, apps_path)[0]
        libs.append(ctypes.CDLL(file_name, ctypes.RTLD_GLOBAL))
        logging.info("Load additional library %s", file_name)

    server = rpc.Server(args.host, args.port, args.port_end, exclusive=args.exclusive)
    server.libs += libs
    server.proc.join()

if __name__ == "__main__":
    main()
