"""Start an RPC server"""
from __future__ import absolute_import

import logging
import argparse
from ..contrib import rpc

def main():
    """Main funciton"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='the hostname of the server')
    parser.add_argument('--port', type=int, default=9090,
                        help='The port of the PRC')
    parser.add_argument('--port_end', type=int, default=9199,
                        help='The end search port of the PRC')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    server = rpc.Server(args.host, args.port, args.port_end)
    server.proc.join()

if __name__ == "__main__":
    main()
