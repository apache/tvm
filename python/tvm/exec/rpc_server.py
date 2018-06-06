# pylint: disable=redefined-outer-name, invalid-name
"""Start an RPC server"""
from __future__ import absolute_import

import argparse
import multiprocessing
import sys
import logging
from ..contrib import rpc


def main(args):
    """Main funciton"""

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
                        tracker_addr=tracker_addr,
                        load_library=args.load_library)
    server.proc.join()


if __name__ == "__main__":
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
    parser.add_argument('--no-fork', dest='fork', action='store_false',
                        help="Use spawn mode to avoid fork. This option \
                         is able to avoid potential fork problems with Metal, OpenCL \
                         and ROCM compilers.")
    parser.set_defaults(fork=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.fork is False:
        if sys.version_info[0] < 3:
            raise RuntimeError(
                "Python3 is required for spawn mode."
            )
        multiprocessing.set_start_method('spawn')
    else:
        logging.info("If you are running ROCM/Metal, \
        fork with cause compiler internal error. Try to launch with arg ```--no-fork```")
    main(args)
