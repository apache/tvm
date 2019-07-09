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
"""VTA Package configuration module

This module is dependency free and can be used to configure package.
"""
from __future__ import absolute_import as _abs

import json
import glob

class PkgConfig(object):
    """Simple package config tool for VTA.

    This is used to provide runtime specific configurations.

    Parameters
    ----------
    cfg : dict
        The config dictionary

    proj_root : str
        Path to the project root
    """
    cfg_keys = [
        "TARGET",
        "HW_VER",
        "LOG_INP_WIDTH",
        "LOG_WGT_WIDTH",
        "LOG_ACC_WIDTH",
        "LOG_OUT_WIDTH",
        "LOG_BATCH",
        "LOG_BLOCK_IN",
        "LOG_BLOCK_OUT",
        "LOG_BUS_WIDTH",
        "LOG_UOP_BUFF_SIZE",
        "LOG_INP_BUFF_SIZE",
        "LOG_WGT_BUFF_SIZE",
        "LOG_ACC_BUFF_SIZE",
    ]

    def __init__(self, cfg, proj_root):
        # include path
        self.include_path = [
            "-I%s/include" % proj_root,
            "-I%s/vta/include" % proj_root,
            "-I%s/3rdparty/dlpack/include" % proj_root,
            "-I%s/3rdparty/dmlc-core/include" % proj_root
        ]
        # List of source files that can be used to build standalone library.
        self.lib_source = []
        self.lib_source += glob.glob("%s/vta/src/*.cc" % proj_root)
        self.lib_source += glob.glob("%s/vta/src/%s/*.cc" % (proj_root, cfg["TARGET"]))
        # macro keys
        self.macro_defs = []
        self.cfg_dict = {}
        for key in self.cfg_keys:
            self.macro_defs.append("-DVTA_%s=%s" % (key, str(cfg[key])))
            self.cfg_dict[key] = cfg[key]

        self.target = cfg["TARGET"]

        if self.target == "pynq":
            self.ldflags = [
                "-L/usr/lib",
                "-l:libcma.so"]
        else:
            self.ldflags = []

        # Derive bitstream config string.
        self.bitstream = "{}_{}x{}x{}_a{}w{}o{}s{}_{}_{}_{}_{}".format(
            cfg["TARGET"],
            (1 << cfg["LOG_BATCH"]),
            (1 << cfg["LOG_BLOCK_IN"]),
            (1 << cfg["LOG_BLOCK_OUT"]),
            (1 << cfg["LOG_INP_WIDTH"]),
            (1 << cfg["LOG_WGT_WIDTH"]),
            (1 << cfg["LOG_OUT_WIDTH"]),
            (1 << cfg["LOG_ACC_WIDTH"]),
            cfg["LOG_UOP_BUFF_SIZE"],
            cfg["LOG_INP_BUFF_SIZE"],
            cfg["LOG_WGT_BUFF_SIZE"],
            cfg["LOG_ACC_BUFF_SIZE"])

        # Derive FPGA parameters from target
        #   - device:           part number
        #   - freq:             PLL frequency
        #   - per:              clock period to achieve in HLS
        #                       (how aggressively design is pipelined)
        #   - axi_bus_width:    axi bus width used for DMA transactions
        #                       (property of FPGA memory interface)
        #   - axi_cache_bits:   ARCACHE/AWCACHE signals for the AXI bus
        #                       (e.g. 1111 is write-back read and write allocate)
        #   - axi_prot_bits:    ARPROT/AWPROT signals for the AXI bus
        #   - max_bus_width:    maximum bus width allowed
        #                       (property of FPGA vendor toolchains)
        if self.target == "ultra96":
            self.fpga_device = "xczu3eg-sbva484-1-e"
            self.fpga_freq = 333
            self.fpga_per = 2
            self.fpga_axi_bus_width = 128
            self.axi_cache_bits = '1111'
            self.axi_prot_bits = '010'
            fpga_max_bus_width = 1024
        else:
            # By default, we use the pynq parameters
            self.fpga_device = "xc7z020clg484-1"
            self.fpga_freq = 100
            self.fpga_per = 7
            self.fpga_axi_bus_width = 64
            self.axi_cache_bits = '1111'
            self.axi_prot_bits = '000'
            fpga_max_bus_width = 1024

        # Derive FPGA memory mapped registers map
        if self.target == "ultra96":
            self.fetch_base_addr = "0xA0001000"
            self.load_base_addr = "0xA0002000"
            self.compute_base_addr = "0xA0003000"
            self.store_base_addr = "0xA0004000"
        else:
            # By default, we use the pynq parameters
            self.fetch_base_addr = "0x43C00000"
            self.load_base_addr = "0x43C20000"
            self.compute_base_addr = "0x43C10000"
            self.store_base_addr = "0x43C30000"
        # Add to the macro defs
        self.macro_defs.append("-DVTA_FETCH_ADDR=%s" % (self.fetch_base_addr))
        self.macro_defs.append("-DVTA_LOAD_ADDR=%s" % (self.load_base_addr))
        self.macro_defs.append("-DVTA_COMPUTE_ADDR=%s" % (self.compute_base_addr))
        self.macro_defs.append("-DVTA_STORE_ADDR=%s" % (self.store_base_addr))

        # Derive SRAM parameters
        # The goal here is to determine how many memory banks are needed,
        # how deep and wide each bank needs to be. This is derived from
        # the size of each memory element (result of data width, and tensor shape),
        # and also how wide a memory can be as permitted by the FPGA tools.
        #
        # The mem axi ratio is a parameter used by HLS to resize memories
        # so memory read/write ports are the same size as the design axi bus width.
        #
        # Input memory
        inp_mem_bus_width = 1 << (cfg["LOG_INP_WIDTH"] + \
                                  cfg["LOG_BATCH"] + \
                                  cfg["LOG_BLOCK_IN"])
        self.inp_mem_size_B = 1 << cfg["LOG_INP_BUFF_SIZE"]
        self.inp_mem_banks = (inp_mem_bus_width + \
                              fpga_max_bus_width - 1) // \
                              fpga_max_bus_width
        self.inp_mem_width = min(inp_mem_bus_width, fpga_max_bus_width)
        self.inp_mem_depth = self.inp_mem_size_B * 8 // inp_mem_bus_width
        self.inp_mem_axi_ratio = self.inp_mem_width // self.fpga_axi_bus_width
        # Weight memory
        wgt_mem_bus_width = 1 << (cfg["LOG_WGT_WIDTH"] + \
                                  cfg["LOG_BLOCK_IN"] + \
                                  cfg["LOG_BLOCK_OUT"])
        self.wgt_mem_size_B = 1 << cfg["LOG_WGT_BUFF_SIZE"]
        self.wgt_mem_banks = (wgt_mem_bus_width + \
                              fpga_max_bus_width - 1) // \
                              fpga_max_bus_width
        self.wgt_mem_width = min(wgt_mem_bus_width, fpga_max_bus_width)
        self.wgt_mem_depth = self.wgt_mem_size_B * 8 // wgt_mem_bus_width
        self.wgt_mem_axi_ratio = self.wgt_mem_width // self.fpga_axi_bus_width
        # Output memory
        out_mem_bus_width = 1 << (cfg["LOG_OUT_WIDTH"] + \
                                  cfg["LOG_BATCH"] + \
                                  cfg["LOG_BLOCK_OUT"])
        self.out_mem_size_B = 1 << cfg["LOG_OUT_BUFF_SIZE"]
        self.out_mem_banks = (out_mem_bus_width + \
                              fpga_max_bus_width - 1) // \
                              fpga_max_bus_width
        self.out_mem_width = min(out_mem_bus_width, fpga_max_bus_width)
        self.out_mem_depth = self.out_mem_size_B * 8 // out_mem_bus_width
        self.out_mem_axi_ratio = self.out_mem_width // self.fpga_axi_bus_width

    @property
    def cflags(self):
        return self.include_path + self.macro_defs

    @property
    def cfg_json(self):
        return json.dumps(self.cfg_dict, indent=2)

    def same_config(self, cfg):
        """Compare if cfg is same as current config.

        Parameters
        ----------
        cfg : the configuration
            The configuration

        Returns
        -------
        equal : bool
            Whether the configuration is the same.
        """
        for k, v in self.cfg_dict.items():
            if k not in cfg:
                return False
            if cfg[k] != v:
                return False
        return True
