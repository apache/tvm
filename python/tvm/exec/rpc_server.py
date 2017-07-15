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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    server = rpc.Server(args.host, args.port, args.port_end)

    if args.with_executor:
        curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
        apps_path = os.path.join(curr_path, "../../../apps/graph_executor/lib/")
        lib_path = find_lib_path('libtvm_graph_exec.so', apps_path)
        server.libs.append(ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL))

    server.proc.join()

if __name__ == "__main__":
    main()
