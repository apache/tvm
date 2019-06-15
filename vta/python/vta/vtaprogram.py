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
"""VTA specific bitstream program library."""
import os
import sys
import argparse

def main():
    """Main funciton"""
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=str, default="",
                        help="target")
    parser.add_argument("bitstream", type=str, default="",
                        help="bitstream path")
    args = parser.parse_args()

    if len(sys.argv) < 3:
        parser.print_help()
        return
    if args.target != 'pynq':
        return

    curr_path = os.path.dirname(
        os.path.abspath(os.path.expanduser(__file__)))
    path_list = [
        os.path.join(curr_path, "/{}".format(args.bitstream)),
        os.path.join('./', "{}".format(args.bitstream))
    ]
    ok_path_list = [p for p in path_list if os.path.exists(p)]
    if not ok_path_list:
        raise RuntimeError("Cannot find bitstream file in %s" % str(path_list))

    vta_bitstream_program(args.target, args.bitstream)

def vta_pynq_bitstream_program(bitstream_path):
    sys.path.append("/home/xilinx/")
    from pynq import Bitstream
    bitstream = Bitstream(bitstream_path)
    bitstream.download()

def vta_bitstream_program(target, bitstream):
    if target == 'pynq':
        vta_pynq_bitstream_program(bitstream)
    else:
        raise RuntimeError("{} is not support target \
                            for bitstream program"
                           .format(target))

if __name__ == "__main__":
    main()
