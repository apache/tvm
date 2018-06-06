"""Tool to start RPC tracker"""
from __future__ import absolute_import

import logging
import argparse
import multiprocessing
import sys
from ..contrib.rpc.tracker import Tracker

def main():
    """Main funciton"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='the hostname of the tracker')
    parser.add_argument('--port', type=int, default=9190,
                        help='The port of the PRC')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if sys.version_info[0] < 3:
        logging.warning("Python3 is recommended for RPC service!")
    tracker = Tracker(args.host, port=args.port)
    tracker.proc.join()

if __name__ == "__main__":
    if sys.version_info[0] >= 3:
        multiprocessing.set_start_method('spawn')
    main()
