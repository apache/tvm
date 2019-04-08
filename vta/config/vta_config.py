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
"""VTA config tool"""
import os
import sys
import json
import argparse

def get_pkg_config(cfg):
    """Get the pkg config object."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    proj_root = os.path.abspath(os.path.join(curr_path, "../../"))
    pkg_config_py = os.path.join(proj_root, "vta/python/vta/pkg_config.py")
    libpkg = {"__file__": pkg_config_py}
    exec(compile(open(pkg_config_py, "rb").read(), pkg_config_py, "exec"), libpkg, libpkg)
    PkgConfig = libpkg["PkgConfig"]
    return PkgConfig(cfg, proj_root)


def main():
    """Main funciton"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cfg", type=str, default="",
                        help="path to the config json")
    parser.add_argument("--cflags", action="store_true",
                        help="print the cflags")
    parser.add_argument("--defs", action="store_true",
                        help="print the macro defs")
    parser.add_argument("--sources", action="store_true",
                        help="print the source file paths")
    parser.add_argument("--update", action="store_true",
                        help="Print out the json option.")
    parser.add_argument("--ldflags", action="store_true",
                        help="print the cflags")
    parser.add_argument("--cfg-json", action="store_true",
                        help="print all the config json")
    parser.add_argument("--save-cfg-json", type=str, default="",
                        help="save config json to file")
    parser.add_argument("--target", action="store_true",
                        help="print the target")
    parser.add_argument("--cfg-str", action="store_true",
                        help="print the configuration string")
    parser.add_argument("--get-inpwidth", action="store_true",
                        help="returns log of input bitwidth")
    parser.add_argument("--get-wgtwidth", action="store_true",
                        help="returns log of weight bitwidth")
    parser.add_argument("--get-accwidth", action="store_true",
                        help="returns log of accum bitwidth")
    parser.add_argument("--get-outwidth", action="store_true",
                        help="returns log of output bitwidth")
    parser.add_argument("--get-batch", action="store_true",
                        help="returns log of tensor batch dimension")
    parser.add_argument("--get-blockin", action="store_true",
                        help="returns log of tensor block in dimension")
    parser.add_argument("--get-blockout", action="store_true",
                        help="returns log of tensor block out dimension")
    parser.add_argument("--get-uopbuffsize", action="store_true",
                        help="returns log of micro-op buffer size in B")
    parser.add_argument("--get-inpbuffsize", action="store_true",
                        help="returns log of input buffer size in B")
    parser.add_argument("--get-wgtbuffsize", action="store_true",
                        help="returns log of weight buffer size in B")
    parser.add_argument("--get-accbuffsize", action="store_true",
                        help="returns log of accum buffer size in B")
    parser.add_argument("--get-outbuffsize", action="store_true",
                        help="returns log of output buffer size in B")
    parser.add_argument("--get-fpgafreq", action="store_true",
                        help="returns FPGA frequency")
    parser.add_argument("--get-fpgaper", action="store_true",
                        help="returns HLS target clock period")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    curr_path = os.path.dirname(
        os.path.abspath(os.path.expanduser(__file__)))
    proj_root = os.path.abspath(os.path.join(curr_path, "../../"))
    path_list = [
        os.path.join(proj_root, "vta_config.json"),
        os.path.join(proj_root, "build", "vta_config.json"),
        os.path.join(proj_root, "vta/config/vta_config.json")
    ]
    if args.use_cfg:
        path_list = [args.use_cfg]
    ok_path_list = [p for p in path_list if os.path.exists(p)]
    if not ok_path_list:
        raise RuntimeError("Cannot find config in %s" % str(path_list))
    cfg = json.load(open(ok_path_list[0]))
    cfg["LOG_OUT_BUFF_SIZE"] = (
        cfg["LOG_ACC_BUFF_SIZE"] +
        cfg["LOG_OUT_WIDTH"] -
        cfg["LOG_ACC_WIDTH"])
    pkg = get_pkg_config(cfg)

    if args.target:
        print(pkg.target)

    if args.defs:
        print(" ".join(pkg.macro_defs))

    if args.sources:
        print(" ".join(pkg.lib_source))

    if args.cflags:
        cflags_str = " ".join(pkg.cflags)
        if cfg["TARGET"] == "pynq":
            cflags_str += " -DVTA_TARGET_PYNQ"
        print(cflags_str)

    if args.ldflags:
        print(" ".join(pkg.ldflags))

    if args.cfg_json:
        print(pkg.cfg_json)

    if args.save_cfg_json:
        with open(args.save_cfg_json, "w") as fo:
            fo.write(pkg.cfg_json)

    if args.cfg_str:
        # Needs to match the BITSTREAM string in python/vta/environment.py
        cfg_str = "{}x{}x{}_{}bx{}b_{}_{}_{}_{}_{}MHz_{}ns_v{}".format(
            (1 << cfg["LOG_BATCH"]),
            (1 << cfg["LOG_BLOCK_IN"]),
            (1 << cfg["LOG_BLOCK_OUT"]),
            (1 << cfg["LOG_INP_WIDTH"]),
            (1 << cfg["LOG_WGT_WIDTH"]),
            cfg["LOG_UOP_BUFF_SIZE"],
            cfg["LOG_INP_BUFF_SIZE"],
            cfg["LOG_WGT_BUFF_SIZE"],
            cfg["LOG_ACC_BUFF_SIZE"],
            cfg["HW_FREQ"],
            cfg["HW_CLK_TARGET"],
            cfg["HW_VER"].replace('.', '_'))
        print(cfg_str)

    if args.get_inpwidth:
        print(cfg["LOG_INP_WIDTH"])

    if args.get_wgtwidth:
        print(cfg["LOG_WGT_WIDTH"])

    if args.get_accwidth:
        print(cfg["LOG_ACC_WIDTH"])

    if args.get_outwidth:
        print(cfg["LOG_OUT_WIDTH"])

    if args.get_batch:
        print(cfg["LOG_BATCH"])

    if args.get_blockin:
        print(cfg["LOG_BLOCK_IN"])

    if args.get_blockout:
        print(cfg["LOG_BLOCK_OUT"])

    if args.get_uopbuffsize:
        print(cfg["LOG_UOP_BUFF_SIZE"])

    if args.get_inpbuffsize:
        print(cfg["LOG_INP_BUFF_SIZE"])

    if args.get_wgtbuffsize:
        print(cfg["LOG_WGT_BUFF_SIZE"])

    if args.get_outbuffsize:
        print(cfg["LOG_OUT_BUFF_SIZE"])

    if args.get_accbuffsize:
        print(cfg["LOG_ACC_BUFF_SIZE"])

    if args.get_fpgafreq:
        print(cfg["HW_FREQ"])

    if args.get_fpgaper:
        print(cfg["HW_CLK_TARGET"])

if __name__ == "__main__":
    main()
