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
                "-lsds_lib",
                "-L/opt/python3.6/lib/python3.6/site-packages/pynq/drivers/",
                "-L/opt/python3.6/lib/python3.6/site-packages/pynq/lib/",
                "-l:libdma.so"]
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
