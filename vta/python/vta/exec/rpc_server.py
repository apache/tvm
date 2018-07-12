"""VTA customized TVM RPC Server

Provides additional runtime function and library loading.
"""
from __future__ import absolute_import

import logging
import argparse
import os
import ctypes
import json
import tvm
from tvm._ffi.base import c_str
from tvm import rpc
from tvm.contrib import cc

from ..environment import get_env
from ..pkg_config import PkgConfig
from ..libinfo import find_libvta


@tvm.register_func("tvm.rpc.server.start", override=True)
def server_start():
    """VTA RPC server extension."""
    # pylint: disable=unused-variable
    curr_path = os.path.dirname(
        os.path.abspath(os.path.expanduser(__file__)))
    proj_root = os.path.abspath(os.path.join(curr_path, "../../../../"))
    dll_path = find_libvta()[0]
    cfg_path = os.path.abspath(os.path.join(proj_root, "build/vta_config.json"))
    runtime_dll = []
    _load_module = tvm.get_global_func("tvm.rpc.server.load_module")

    def load_vta_dll():
        """Try to load vta dll"""
        if not runtime_dll:
            runtime_dll.append(ctypes.CDLL(dll_path, ctypes.RTLD_GLOBAL))
        logging.info("Loading VTA library: %s", dll_path)
        return runtime_dll[0]

    @tvm.register_func("tvm.rpc.server.load_module", override=True)
    def load_module(file_name):
        load_vta_dll()
        return _load_module(file_name)

    @tvm.register_func("device_api.ext_dev")
    def ext_dev_callback():
        load_vta_dll()
        return tvm.get_global_func("device_api.ext_dev")()

    @tvm.register_func("tvm.contrib.vta.init", override=True)
    def program_fpga(file_name):
        path = tvm.get_global_func("tvm.rpc.server.workpath")(file_name)
        load_vta_dll().VTAProgram(c_str(path))
        logging.info("Program FPGA with %s", file_name)

    @tvm.register_func("tvm.rpc.server.shutdown", override=True)
    def server_shutdown():
        if runtime_dll:
            runtime_dll[0].VTARuntimeShutdown()
            runtime_dll.pop()

    @tvm.register_func("tvm.contrib.vta.reconfig_runtime", override=True)
    def reconfig_runtime(cfg_json):
        """Rebuild and reload runtime with new configuration.

        Parameters
        ----------
        cfg_json : str
            JSON string used for configurations.
        """
        if runtime_dll:
            raise RuntimeError("Can only reconfig in the beginning of session...")
        env = get_env()
        cfg = json.loads(cfg_json)
        cfg["TARGET"] = env.TARGET
        pkg = PkgConfig(cfg, proj_root)
        # check if the configuration is already the same
        if os.path.isfile(cfg_path):
            old_cfg = json.loads(open(cfg_path, "r").read())
            if pkg.same_config(old_cfg):
                logging.info("Skip reconfig_runtime due to same config.")
                return
        cflags = ["-O2", "-std=c++11"]
        cflags += pkg.cflags
        ldflags = pkg.ldflags
        lib_name = dll_path
        source = pkg.lib_source
        logging.info("Rebuild runtime: output=%s, cflags=%s, source=%s, ldflags=%s",
                     dll_path, str(cflags), str(source), str(ldflags))
        cc.create_shared(lib_name, source, cflags + ldflags)
        with open(cfg_path, "w") as outputfile:
            outputfile.write(pkg.cfg_json)


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
    server.proc.join()

if __name__ == "__main__":
    main()
