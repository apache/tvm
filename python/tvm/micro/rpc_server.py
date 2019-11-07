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
"""RPC server for interacting with devices via MicroTVM"""

import logging
import argparse
import os
import ctypes
import json
import tvm
from tvm import rpc
from tvm import micro

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='the hostname of the server')
    parser.add_argument('--port', type=int, default=9091,
                        help='The port of the RPC')
    parser.add_argument('--port-end', type=int, default=9199,
                        help='The end search port of the RPC')
    parser.add_argument('--key', type=str, default="",
                        help='RPC key used to identify the connection type.')
    parser.add_argument('--tracker', type=str, default="",
                        help='Report to RPC tracker')
    parser.add_argument('--dev-config', type=str,
                        help='JSON config file for the target device')
    parser.add_argument('--dev-id', type=str,
                        help='Unique ID for the target device')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.tracker:
        url, port = args.tracker.split(':')
        port = int(port)
        tracker_addr = (url, port)
        if not args.key:
            raise RuntimeError(
                'Need key to present type of resource when tracker is available')
    else:
        tracker_addr = None

    if not (args.dev_config or args.dev_id):
        raise RuntimeError('must provide either --dev-config or --dev-id')
    if args.dev_config and args.dev_id:
        raise RuntimeError('only one of --dev-config and --dev-id allowed')

    if args.dev_config:
        with open(args.dev_config, 'r') as dev_conf_file:
            dev_config = json.load(dev_conf_file)
    else:
        dev_config = micro.device.get_config(args.dev_id)

    @tvm.register_func('tvm.rpc.server.start', override=True)
    def server_start():
        session = micro.Session(dev_config)
        session._enter()

        @tvm.register_func('tvm.rpc.server.shutdown', override=True)
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
