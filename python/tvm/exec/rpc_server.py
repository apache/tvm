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
    parser.add_argument('--load-libary', type=str, default="",
                        help="Additional libary to load")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    load_libary = args.load_libary.split(":")
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    apps_path = os.path.join(curr_path, "../../../apps/graph_executor/lib/")
    libs = []
    if args.with_executor:
        load_libary += ["libtvm_graph_exec.so"]
    if load_libary:
        for file_name in args.load_libary.split(":"):
            file_name = find_lib_path(file_name, apps_path)[0]
            libs.append(ctypes.CDLL(file_name, ctypes.RTLD_GLOBAL))
            logging.info("Load additional libary %s", file_name)

    server = rpc.Server(args.host, args.port, args.port_end)
    server.libs += libs
    server.proc.join()

if __name__ == "__main__":
    main()
