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
"""tvm.contrib.msc.core.utils.info"""

import os
import json


def load_dict(str_dict: str, flavor: str = "json") -> dict:
    """Load the string/file to dict.

    Parameters
    ----------
    str_dict: string
        The file_path or string object.
    flavor: str
        The flavor for load.

    Returns
    -------
    dict_obj: dict
        The loaded dict.
    """

    if isinstance(str_dict, str) and os.path.isfile(str_dict):
        with open(str_dict, "r") as f:
            dict_obj = json.load(f)
    elif isinstance(str_dict, str):
        dict_obj = json.loads(str_dict)
    assert flavor == "json", "Unexpected flavor for load_dict: " + str(flavor)
    return dict_obj


def dump_dict(dict_obj: dict, flavor: str = "dmlc") -> str:
    """Dump the config to string.

    Parameters
    ----------
    src_dict: dict
        The source dict.
    flavor: str
        The flavor for dumps.

    Returns
    -------
    str_dict: string
        The dumped string.
    """

    if not dict_obj:
        return ""
    if flavor == "dmlc":
        return json.dumps({k: int(v) if isinstance(v, bool) else v for k, v in dict_obj.items()})
    return json.dumps(dict_obj)


def dict_equal(dict_a: dict, dict_b: dict) -> bool:
    """Check if two dicts are the same.

    Parameters
    ----------
    dict_a: dict
        The A dict.
    dict_b: dict
        The B dict.

    Returns
    -------
    equal: bool
        Whether two dicts are the same.
    """

    if not isinstance(dict_a, dict) or not isinstance(dict_b, dict):
        return False
    if dict_a.keys() != dict_b.keys():
        return False
    for k, v in dict_a.items():
        if not isinstance(v, type(dict_b[k])):
            return False
        if isinstance(v, dict) and not dict_equal(v, dict_b[k]):
            return False
        if v != dict_b[k]:
            return False
    return True
