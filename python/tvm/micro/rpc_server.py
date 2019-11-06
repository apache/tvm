import logging
import argparse
import os
import ctypes
import json
import tvm
from tvm import rpc
from tvm import micro

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
    parser.add_argument('--dev-config', type=str, required=True,
                        help="JSON config file for the target device")
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

    with open(args.dev_config, 'r') as dev_conf_file:
        dev_config = json.load(dev_conf_file)
    #dev_config = micro.device.arm.stm32f746xx.default_config('127.0.0.1', 6666)
    #dev_config = micro.device.host.default_config()

    @tvm.register_func("tvm.rpc.server.start", override=True)
    def server_start():
        session = micro.Session(dev_config)
        session._enter()

        @tvm.register_func("tvm.rpc.server.shutdown", override=True)
        def server_shutdown():
            session._exit()

    server = rpc.Server(args.host,
                        args.port,
                        args.port_end,
                        key=args.key,
                        tracker_addr=tracker_addr)
    server.proc.join()


if __name__ == "__main__":
    main()
