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
"""Tool to query RPC tracker status"""
from __future__ import absolute_import

import logging
import argparse
import os
from .. import rpc


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="", help="the hostname of the tracker")
    parser.add_argument("--port", type=int, default=None, help="The port of the RPC")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # default to local host or environment variable
    if not args.host:
        args.host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")

    if not args.port:
        args.port = int(os.environ.get("TVM_TRACKER_PORT", "9190"))

    conn = rpc.connect_tracker(args.host, args.port)
    # pylint: disable=superfluous-parens
    print("Tracker address %s:%d\n" % (args.host, args.port))
    print("%s" % conn.text_summary())


if __name__ == "__main__":
    main()
