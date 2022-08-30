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
    _prefix = "relay._transform."
    pass_list = input_string.split(",")
    missing_list = [
        p.strip()
        for p in pass_list
        if len(p.strip()) > 0 and tvm.get_global_func(_prefix + p.strip(), True) is None
    ]
    if len(missing_list) > 0:
        available_list = [
            n[len(_prefix) :] for n in registry.list_global_func_names() if n.startswith(_prefix)
        ]
        raise argparse.ArgumentTypeError(
            "Following passes are not registered within tvm: {}. Available: {}.".format(
                ", ".join(missing_list), ", ".join(sorted(available_list))
            )
        )
    return pass_list
