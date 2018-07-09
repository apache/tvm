# pylint: disable=redefined-outer-name, invalid-name
"""RPC web proxy, allows redirect to websocket based RPC servers(browsers)"""
from __future__ import absolute_import

import logging
import argparse
import multiprocessing
import sys
import os
from ..rpc.proxy import Proxy


def find_example_resource():
    """Find resource examples."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    base_path = os.path.join(curr_path, "../../../")
    index_page = os.path.join(base_path, "web/example_rpc.html")
    js_files = [
        os.path.join(base_path, "web/tvm_runtime.js"),
        os.path.join(base_path, "lib/libtvm_web_runtime.js"),
        os.path.join(base_path, "lib/libtvm_web_runtime.js.mem")
    ]
    for fname in [index_page] + js_files:
        if not os.path.exists(fname):
            raise RuntimeError("Cannot find %s" % fname)
    return index_page, js_files


def main(args):
    """Main funciton"""
    if args.tracker:
        url, port = args.tracker.split(":")
        port = int(port)
        tracker_addr = (url, port)
    else:
        tracker_addr = None

    if args.example_rpc:
        index, js_files = find_example_resource()
        prox = Proxy(args.host,
                     port=args.port,
                     web_port=args.web_port,
                     index_page=index,
                     resource_files=js_files,
                     tracker_addr=tracker_addr)
    else:
        prox = Proxy(args.host,
                     port=args.port,
                     web_port=args.web_port,
                     tracker_addr=tracker_addr)
    prox.proc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='the hostname of the server')
    parser.add_argument('--port', type=int, default=9090,
                        help='The port of the PRC')
    parser.add_argument('--web-port', type=int, default=8888,
                        help='The port of the http/websocket server')
    parser.add_argument('--example-rpc', type=bool, default=False,
                        help='Whether to switch on example rpc mode')
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
