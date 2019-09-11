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
                        help="print the ldflags")
    parser.add_argument("--cfg-json", action="store_true",
                        help="print all the config json")
    parser.add_argument("--save-cfg-json", type=str, default="",
                        help="save config json to file")
    parser.add_argument("--target", action="store_true",
                        help="print the target")
    parser.add_argument("--cfg-str", action="store_true",
                        help="print the configuration string")
    parser.add_argument("--get-inp-mem-banks", action="store_true",
                        help="returns number of input memory banks")
    parser.add_argument("--get-inp-mem-width", action="store_true",
                        help="returns input memory read/write port width")
    parser.add_argument("--get-inp-mem-depth", action="store_true",
                        help="returns input memory depth")
    parser.add_argument("--get-inp-mem-axi-ratio", action="store_true",
                        help="returns ratio between input element width and axi width")
    parser.add_argument("--get-wgt-mem-banks", action="store_true",
                        help="returns number of weight memory banks")
    parser.add_argument("--get-wgt-mem-width", action="store_true",
                        help="returns weight memory read/write port width")
    parser.add_argument("--get-wgt-mem-depth", action="store_true",
                        help="returns weight memory depth")
    parser.add_argument("--get-wgt-mem-axi-ratio", action="store_true",
                        help="returns ratio between weight element width and axi width")
    parser.add_argument("--get-out-mem-banks", action="store_true",
                        help="returns number of output memory banks")
    parser.add_argument("--get-out-mem-width", action="store_true",
                        help="returns output memory read/write port width")
    parser.add_argument("--get-out-mem-depth", action="store_true",
                        help="returns output memory depth")
    parser.add_argument("--get-out-mem-axi-ratio", action="store_true",
                        help="returns ratio between output element width and axi width")
    parser.add_argument("--get-axi-cache-bits", action="store_true",
                        help="returns AXI system ARCACHE/AWCACHE hardcoded bit value")
    parser.add_argument("--get-axi-prot-bits", action="store_true",
                        help="returns AXI system ARPROT/AWPROT hardcoded bit value")
    parser.add_argument("--get-ip-reg-map-range", action="store_true",
                        help="returns ip register map address range")
    parser.add_argument("--get-fetch-base-addr", action="store_true",
                        help="returns fetch module base address")
    parser.add_argument("--get-load-base-addr", action="store_true",
                        help="returns load module base address")
    parser.add_argument("--get-compute-base-addr", action="store_true",
                        help="returns compute module base address")
    parser.add_argument("--get-store-base-addr", action="store_true",
                        help="returns store module base address")
    parser.add_argument("--get-fpga-dev", action="store_true",
                        help="returns FPGA device target")
    parser.add_argument("--get-fpga-family", action="store_true",
                        help="returns FPGA device family")
    parser.add_argument("--get-fpga-freq", action="store_true",
                        help="returns FPGA frequency")
    parser.add_argument("--get-fpga-per", action="store_true",
                        help="returns HLS target clock period")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    curr_path = os.path.dirname(
        os.path.abspath(os.path.expanduser(__file__)))
    proj_root = os.path.abspath(os.path.join(curr_path, "../../"))
    path_list = [
        os.path.join(proj_root, "vta/config/vta_config.json")
    ]
    if args.use_cfg:
        path_list = [args.use_cfg]
    ok_path_list = [p for p in path_list if os.path.exists(p)]
    if not ok_path_list:
        raise RuntimeError("Cannot find config in %s" % str(path_list))
    cfg = json.load(open(ok_path_list[0]))

    pkg = get_pkg_config(cfg)

    if args.target:
        print(pkg.TARGET)

    if args.defs:
        print(" ".join(pkg.macro_defs))

    if args.sources:
        print(" ".join(pkg.lib_source))

    if args.cflags:
        cflags_str = " ".join(pkg.cflags)
        if pkg.TARGET == "pynq":
            cflags_str += " -DVTA_TARGET_PYNQ"
        elif pkg.TARGET == "de10nano":
            cflags_str += " -DVTA_TARGET_DE10_NANO"
        elif pkg.TARGET == "ultra96":
            cflags_str += " -DVTA_TARGET_ULTRA96"
        print(cflags_str)

    if args.ldflags:
        print(" ".join(pkg.ldflags))

    if args.cfg_json:
        print(pkg.cfg_json)

    if args.save_cfg_json:
        with open(args.save_cfg_json, "w") as fo:
            fo.write(pkg.cfg_json)

    if args.cfg_str:
        print(pkg.TARGET + "_" + pkg.bitstream)

    if args.get_inp_mem_banks:
        print(pkg.inp_mem_banks)

    if args.get_inp_mem_width:
        print(pkg.inp_mem_width)

    if args.get_inp_mem_depth:
        print(pkg.inp_mem_depth)

    if args.get_inp_mem_axi_ratio:
        print(pkg.inp_mem_axi_ratio)

    if args.get_wgt_mem_banks:
        print(pkg.wgt_mem_banks)

    if args.get_wgt_mem_width:
        print(pkg.wgt_mem_width)

    if args.get_wgt_mem_depth:
        print(pkg.wgt_mem_depth)

    if args.get_wgt_mem_axi_ratio:
        print(pkg.wgt_mem_axi_ratio)

    if args.get_out_mem_banks:
        print(pkg.out_mem_banks)

    if args.get_out_mem_width:
        print(pkg.out_mem_width)

    if args.get_out_mem_depth:
        print(pkg.out_mem_depth)

    if args.get_out_mem_axi_ratio:
        print(pkg.out_mem_axi_ratio)

    if args.get_axi_cache_bits:
        print(pkg.axi_cache_bits)

    if args.get_axi_prot_bits:
        print(pkg.axi_prot_bits)

    if args.get_ip_reg_map_range:
        print(pkg.ip_reg_map_range)

    if args.get_fetch_base_addr:
        print(pkg.fetch_base_addr)

    if args.get_load_base_addr:
        print(pkg.load_base_addr)

    if args.get_compute_base_addr:
        print(pkg.compute_base_addr)

    if args.get_store_base_addr:
        print(pkg.store_base_addr)

    if args.get_fpga_dev:
        print(pkg.fpga_device)

    if args.get_fpga_family:
        print(pkg.fpga_family)

    if args.get_fpga_freq:
        print(pkg.fpga_freq)

    if args.get_fpga_per:
        print(pkg.fpga_per)

if __name__ == "__main__":
    main()
