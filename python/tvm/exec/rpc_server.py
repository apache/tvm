# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=redefined-outer-name, invalid-name
"""Start an RPC server"""
import argparse
import logging
from .. import rpc


def main(args):
    """Main function

    Parameters
    ----------
    args : argparse.Namespace
        parsed args from command-line invocation
    """
    if args.tracker:
        url, port = args.tracker.rsplit(":", 1)
        port = int(port)
        tracker_addr = (url, port)
        if not args.key:
            raise RuntimeError("Need key to present type of resource when tracker is available")
    else:
        tracker_addr = None

    server = rpc.Server(
        args.host,
        args.port,
        args.port_end,
        key=args.key,
        tracker_addr=tracker_addr,
        load_library=args.load_library,
        custom_addr=args.custom_addr,
        silent=args.silent,
        no_fork=not args.fork,
    )
    server.proc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="The host IP address the tracker binds to"
    )
    parser.add_argument("--port", type=int, default=9090, help="The port of the RPC")
    parser.add_argument("--port-end", type=int, default=9199, help="The end search port of the RPC")
    parser.add_argument(
        "--tracker",
        type=str,
        help=("The address of RPC tracker in host:port format. " "e.g. (10.77.1.234:9190)"),
    )
    parser.add_argument(
        "--key", type=str, default="", help="The key used to identify the device type in tracker."
    )
    parser.add_argument("--silent", action="store_true", help="Whether run in silent mode.")
    parser.add_argument("--load-library", type=str, help="Additional library to load")
    parser.add_argument(
        "--no-fork",
        dest="fork",
        action="store_false",
        help="Use spawn mode to avoid fork. This option \
                        is able to avoid potential fork problems with Metal, OpenCL \
                        and ROCM compilers.",
    )
    parser.add_argument(
        "--custom-addr", type=str, help="Custom IP Address to Report to RPC Tracker"
    )

    parser.set_defaults(fork=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not args.fork is False and not args.silent:
        logging.info(
            "If you are running ROCM/Metal, fork will cause "
            "compiler internal error. Try to launch with arg ```--no-fork```"
        )
    main(args)
