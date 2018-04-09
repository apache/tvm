"""VTA customized TVM RPC Server

Provides additional runtime function and library loading.
"""
from __future__ import absolute_import

import logging
import argparse
import os
import ctypes
import tvm
from tvm.contrib import rpc, util, cc


@tvm.register_func("tvm.contrib.rpc.server.start", override=True)
def server_start():
    curr_path = os.path.dirname(
        os.path.abspath(os.path.expanduser(__file__)))
    dll_path = os.path.abspath(
        os.path.join(curr_path, "../../../lib/libvta_runtime.so"))
    runtime_dll = []
    _load_module = tvm.get_global_func("tvm.contrib.rpc.server.load_module")

    @tvm.register_func("tvm.contrib.rpc.server.load_module", override=True)
    def load_module(file_name):
        if not runtime_dll:
            runtime_dll.append(ctypes.CDLL(dll_path, ctypes.RTLD_GLOBAL))
        return _load_module(file_name)

    @tvm.register_func("tvm.contrib.rpc.server.shutdown", override=True)
    def server_shutdown():
        if runtime_dll:
            runtime_dll[0].VTARuntimeShutdown()
            runtime_dll.pop()

    @tvm.register_func("tvm.contrib.vta.reconfig_runtime", override=True)
    def reconfig_runtime(cflags):
        """Rebuild and reload runtime with new configuration.

        Parameters
        ----------
        cfg_json : str
        JSON string used for configurations.
        """
        if runtime_dll:
            raise RuntimeError("Can only reconfig in the beginning of session...")
        cflags = cflags.split()
        cflags += ["-O2", "-std=c++11"]
        lib_name = dll_path
        curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
        proj_root = os.path.abspath(os.path.join(curr_path, "../../../"))
        runtime_source = os.path.join(proj_root, "src/runtime.cc")
        cflags += ["-I%s/include" % proj_root]
        cflags += ["-I%s/nnvm/tvm/include" % proj_root]
        cflags += ["-I%s/nnvm/tvm/dlpack/include" % proj_root]
        cflags += ["-I%s/nnvm/dmlc-core/include" % proj_root]
        logging.info("Rebuild runtime dll with %s", str(cflags))
        cc.create_shared(lib_name, [runtime_source], cflags)


def main():
    """Main funciton"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='the hostname of the server')
    parser.add_argument('--port', type=int, default=9090,
                        help='The port of the PRC')
    parser.add_argument('--port-end', type=int, default=9199,
                        help='The end search port of the PRC')
    parser.add_argument('--key', type=str, default="",
                        help="RPC key used to identify the connection type.")
    parser.add_argument('--tracker', type=str, default="",
                        help="Report to RPC tracker")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    proj_root = os.path.abspath(os.path.join(curr_path, "../../../"))
    lib_path = os.path.abspath(os.path.join(proj_root, "lib/libvta.so"))

    libs = []
    for file_name in [lib_path]:
        libs.append(ctypes.CDLL(file_name, ctypes.RTLD_GLOBAL))
        logging.info("Load additional library %s", file_name)

    if args.tracker:
        url, port = args.tracker.split(":")
        port = int(port)
        tracker_addr = (url, port)
        if not args.key:
            raise RuntimeError(
                "Need key to present type of resource when tracker is available")
    else:
        tracker_addr = None

    server = rpc.Server(args.host,
                        args.port,
                        args.port_end,
                        key=args.key,
                        tracker_addr=tracker_addr)
    server.libs += libs
    server.proc.join()

if __name__ == "__main__":
    main()
