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
from __future__ import absolute_import

import argparse
import ast
import json
import multiprocessing
import sys
import logging
import tvm
from tvm import micro
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
            raise RuntimeError(
                'Need key to present type of resource when tracker is available')
    else:
        tracker_addr = None

    if args.utvm_dev_config or args.utvm_dev_id:
        init_utvm(args)

    server = rpc.Server(args.host,
                        args.port,
                        args.port_end,
                        key=args.key,
                        tracker_addr=tracker_addr,
                        load_library=args.load_library,
                        custom_addr=args.custom_addr,
                        silent=args.silent)
    server.proc.join()


def init_utvm(args):
    """MicroTVM-specific RPC initialization

    Parameters
    ----------
    args : argparse.Namespace
        parsed args from command-line invocation
    """
    if args.utvm_dev_config and args.utvm_dev_id:
        raise RuntimeError('only one of --utvm-dev-config and --utvm-dev-id allowed')

    if args.utvm_dev_config:
        with open(args.utvm_dev_config, 'r') as dev_conf_file:
            dev_config = json.load(dev_conf_file)
    else:
        dev_config_args = ast.literal_eval(args.utvm_dev_config_args)
        generate_config_func = micro.device.get_device_funcs(args.utvm_dev_id)['generate_config']
        dev_config = generate_config_func(*dev_config_args)

    if args.utvm_dev_config or args.utvm_dev_id:
        # add MicroTVM overrides
        @tvm.register_func('tvm.rpc.server.start', override=True)
        def server_start():
            # pylint: disable=unused-variable
            session = micro.Session(dev_config)
            session._enter()

            @tvm.register_func('tvm.rpc.server.shutdown', override=True)
            def server_shutdown():
                session._exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='the hostname of the server')
    parser.add_argument('--port', type=int, default=9090,
                        help='The port of the RPC')
    parser.add_argument('--port-end', type=int, default=9199,
                        help='The end search port of the RPC')
    parser.add_argument('--tracker', type=str,
                        help=("The address of RPC tracker in host:port format. "
                              "e.g. (10.77.1.234:9190)"))
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
    parser.add_argument('--utvm-dev-config', type=str,
                        help=('JSON config file for the target device (if using MicroTVM). '
                              'This file should contain serialized output similar to that returned '
                              "from the device module's generate_config. Can't be specified when "
                              '--utvm-dev-config-args is specified.'))
    parser.add_argument('--utvm-dev-config-args', type=str,
                        help=("Arguments to the device module's generate_config function. "
                              'Must be a python literal parseable by literal_eval. If specified, '
                              "the device configuration is generated using the device module's "
                              "generate_config. Can't be specified when --utvm-dev-config is "
                              "specified."))
    parser.add_argument('--utvm-dev-id', type=str,
                        help=('Unique ID for the target device (if using MicroTVM). Should '
                              'match the name of a module underneath tvm.micro.device).'))

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
