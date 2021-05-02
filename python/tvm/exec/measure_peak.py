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
"""measure bandwidth and compute peak

e.g.
python3 -m tvm.exec.measure_peak --target cuda --rpc-host 127.0.0.1 --rpc-port 9090
python3 -m tvm.exec.measure_peak --target opencl --target-host "llvm -mtriple=aarch64-linux-gnu" \
        --rpc-host $TVM_OPENCL_DEVICE_HOST --rpc-port 9090
"""

import argparse
import logging

from tvm.target import Target
from ..contrib.peak import measure_peak_all


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="llvm", help="The build target")
    parser.add_argument(
        "--target-host", type=str, default=None, help="The host code compilation target"
    )
    parser.add_argument(
        "--rpc-host", type=str, default="127.0.0.1", help="the hostname of the server"
    )
    parser.add_argument("--rpc-port", type=int, default=9090, help="The port of the RPC")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    args.target, args.target_host = Target.check_and_update_host_consist(
        args.target, args.target_host
    )
    measure_peak_all(args.target, args.target_host, args.rpc_host, args.rpc_port)


if __name__ == "__main__":
    main()
