import logging
import argparse
import os
import ctypes
import json
import tvm
from tvm import rpc
from tvm import micro

DEVICE_TYPE = 'openocd'
TOOLCHAIN_PREFIX = 'arm-none-eabi-'

@tvm.register_func("tvm.rpc.server.start", override=True)
def server_start():
    ## pylint: disable=unused-variable
    #curr_path = os.path.dirname(
    #    os.path.abspath(os.path.expanduser(__file__)))
    #proj_root = os.path.abspath(os.path.join(curr_path, "../../../../"))
    #dll_path = find_libvta("libvta")[0]
    #cfg_path = os.path.abspath(os.path.join(proj_root, "build/vta_config.json"))
    #runtime_dll = []
    session = micro.Session(DEVICE_TYPE, TOOLCHAIN_PREFIX)
    session._enter()
    _load_module = tvm.get_global_func("tvm.rpc.server.load_module")

    @tvm.register_func("tvm.rpc.server.load_module", override=True)
    def load_module(file_name):
        #load_vta_dll()
        print(f'BEFORE MOD LOAD for {file_name}')
        res = _load_module(file_name)
        print(f'AFTER MOD LOAD: {res}')
        return res


    #@tvm.register_func("device_api.ext_dev")
    #def ext_dev_callback():
    #    load_vta_dll()
    #    return tvm.get_global_func("device_api.ext_dev")()

    @tvm.register_func("tvm.rpc.server.shutdown", override=True)
    def server_shutdown():
        session._exit()


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='the hostname of the server')
    parser.add_argument('--port', type=int, default=9091,
                        help='The port of the RPC')
    parser.add_argument('--port-end', type=int, default=9199,
                        help='The end search port of the RPC')
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
