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
import argparse


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=str, default="", help="target")
    parser.add_argument("bitstream", type=str, default="", help="bitstream path")
    args = parser.parse_args()

    if args.target not in ("pynq", "ultra96", "de10nano", "sim", "tsim"):
        raise RuntimeError("Unknown target {}".format(args.target))

    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    path_list = [
        os.path.join(curr_path, "/{}".format(args.bitstream)),
        os.path.join("./", "{}".format(args.bitstream)),
    ]
    ok_path_list = [p for p in path_list if os.path.exists(p)]
    if not ok_path_list:
        raise RuntimeError("Cannot find bitstream file in %s" % str(path_list))

    bitstream_program(args.target, args.bitstream)


def pynq_bitstream_program(bitstream_path):
    # pylint: disable=import-outside-toplevel
    from pynq import Bitstream

    bitstream = Bitstream(bitstream_path)
    bitstream.download()


def de10nano_bitstream_program(bitstream_path):
    # pylint: disable=import-outside-toplevel
    from tvm import get_global_func

    program = get_global_func("vta.de10nano.program")
    program(bitstream_path)


def intelfocl_bitstream_program(bitstream_path, mem_size=4 * 1024 * 1024 * 1024):
    # pylint: disable=import-outside-toplevel
    from tvm import get_global_func

    program = get_global_func("vta.oclfpga.program")
    program(bitstream_path, mem_size)


def bitstream_program(target, bitstream, *args):
    """program bitstream to devices"""

    if target in ["pynq", "ultra96"]:
        pynq_bitstream_program(bitstream)
    elif target in ["de10nano"]:
        de10nano_bitstream_program(bitstream)
    elif target in ["sim", "tsim"]:
        # In simulation, bit stream programming is a no-op
        return
    elif target in ["intelfocl"]:
        intelfocl_bitstream_program(bitstream, *args)
    else:
        raise RuntimeError("Unknown target {}".format(target))


if __name__ == "__main__":
    main()
