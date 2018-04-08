"""RPC web proxy, allows redirect to websocket based RPC servers(browsers)"""
from __future__ import absolute_import

import logging
import argparse
import os
from ..contrib.rpc.proxy import Proxy

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


def main():
    """Main funciton"""
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

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
    main()
