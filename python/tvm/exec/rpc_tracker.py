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
"""Tool to start RPC tracker"""
import logging
import argparse
from ..rpc.tracker import Tracker


def main(args):
    """Main function"""
    tracker = Tracker(args.host, port=args.port, port_end=args.port_end, silent=args.silent)
    tracker.proc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="The host IP address the tracker binds to"
    )
    parser.add_argument("--port", type=int, default=9190, help="The port of the RPC")
    parser.add_argument("--port-end", type=int, default=9199, help="The end search port of the RPC")
    parser.add_argument("--silent", action="store_true", help="Whether run in silent mode.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
