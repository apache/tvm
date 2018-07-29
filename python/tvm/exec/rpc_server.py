# pylint: disable=redefined-outer-name, invalid-name
"""Start an RPC server"""
from __future__ import absolute_import

import argparse
import multiprocessing
import sys
import logging
from .. import rpc

def main(args):
    """Main function"""

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
                        load_library=args.load_library,
                        custom_addr=args.custom_addr,
                        silent=args.silent)
    server.proc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='the hostname of the server')
    parser.add_argument('--port', type=int, default=9090,
                        help='The port of the PRC')
    parser.add_argument('--port-end', type=int, default=9199,
                        help='The end search port of the PRC')
    parser.add_argument('--tracker', type=str,
                        help="The address of RPC tracker in host:port format. "
                             "e.g. (10.77.1.234:9190)")
    parser.add_argument('--key', type=str, default="",
                        help="The key used to identify the device type in tracker.")
    parser.add_argument('--silent', action='store_true',
                        help="Whether run in silent mode.")
    parser.add_argument('--load-library', type=str,
                        help="Additional library to load")
    parser.add_argument('--no-fork', dest='fork', action='store_false',
                        help="Use spawn mode to avoid fork. This option \
                         is able to avoid potential fork problems with Metal, OpenCL \
                         and ROCM compilers.")
    parser.add_argument('--custom-addr', type=str,
                        help="Custom IP Address to Report to RPC Tracker")

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
        if not args.silent:
            logging.info("If you are running ROCM/Metal, fork will cause "
                         "compiler internal error. Try to launch with arg ```--no-fork```")
    main(args)
