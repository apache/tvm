"""Start an RPC server"""
from __future__ import absolute_import

import argparse
import logging
from ..contrib import rpc

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
    parser.add_argument('--load-library', type=str, default="",
                        help="Additional library to load")
    parser.add_argument('--tracker', type=str, default="",
                        help="Report to RPC tracker")
    args = parser.parse_args()

    if args.tracker:
        url, port = args.tracker.split(":")
        port = int(port)
        tracker_addr = (url, port)
        if not args.key:
            raise RuntimeError(
                "Need key to present type of resource when tracker is available")
    else:
        tracker_addr = None

    logging.basicConfig(level=logging.INFO)
    server = rpc.Server(args.host,
                        args.port,
                        args.port_end,
                        key=args.key,
                        tracker_addr=tracker_addr,
                        load_library=args.load_library)
    server.proc.join()

if __name__ == "__main__":
    main()
