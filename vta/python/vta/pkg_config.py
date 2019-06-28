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
        "HW_FREQ",
        "HW_CLK_TARGET",
        "HW_VER",
        "LOG_INP_WIDTH",
        "LOG_WGT_WIDTH",
        "LOG_ACC_WIDTH",
        "LOG_OUT_WIDTH",
        "LOG_BATCH",
        "LOG_BLOCK_IN",
        "LOG_BLOCK_OUT",
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
