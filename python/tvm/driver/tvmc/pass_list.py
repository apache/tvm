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
# specific language
"""
TVMC Pass List Management
"""

import argparse

import tvm
from tvm._ffi import registry


def parse_pass_list_str(input_string):
    """Parse an input string for existing passes

    Parameters
    ----------
    input_string: str
        Possibly comma-separated string with the names of passes

    Returns
    -------
    list: a list of existing passes.
    """
    replacements = {
        "tir.": "tir.transform.",
        "qnn.": "relay.qnn._transform.",
        "": "relay._transform.",
    }

    def apply_prefix(p, reverse=False):
        p = p.strip()
        for short, long in replacements.items():
            if not reverse and p.startswith(short):
                if len(short) == 0:
                    p = long + p
                else:
                    p = p.replace(short, long, 1)
                break
            if reverse and p.startswith(long):
                p = p.replace(long, short, 1)
                break
        return p

    pass_list = input_string.split(",")
    missing_list = [
        p
        for p in pass_list
        if len(p.strip()) > 0 and tvm.get_global_func(apply_prefix(p), True) is None
    ]
    if len(missing_list) > 0:
        available_list = [
            apply_prefix(n, True)
            for n in registry.list_global_func_names()
            if any(n.startswith(pre) for pre in replacements.values())
        ]
        raise argparse.ArgumentTypeError(
            "Following passes are not registered within tvm: {}. Available: {}.".format(
                ", ".join(missing_list), ", ".join(sorted(available_list))
            )
        )
    return pass_list
