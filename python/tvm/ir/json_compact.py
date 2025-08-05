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
"""Tool to upgrade json from historical versions."""
import json


def get_version(jgraph):
    """
    Get the tvm version from the json graph.

    Parameters
    ----------
    jgraph : dict
        The json graph.
    """
    return jgraph["metadata"]["tvm_version"]


def create_updater(node_map, from_ver, to_ver):
    """Create an updater to update json loaded data.

    Parameters
    ----------
    node_map : Map[str, Function]
        Map from type_key to updating function

    from_ver : str
        Prefix of version that we can accept,

    to_ver : str
        The target version.

    Returns
    -------
    fupdater : function
        The updater function
    """

    def _updater(data):
        assert get_version(data).startswith(from_ver)
        nodes = data["nodes"]
        for idx, item in enumerate(nodes):
            f = node_map.get(item["type"], None)
            if isinstance(f, list):
                for fpass in f:
                    item = fpass(item, nodes)
            elif f:
                item = f(item, nodes)
            nodes[idx] = item
        data["metadata"]["tvm_version"] = to_ver
        return data

    return _updater


def upgrade_json(json_str):
    """Update json from a historical version.

    Parameters
    ----------
    json_str : str
        A historical json file.

    Returns
    -------
    updated_json : str
        The updated version.
    """
    data = json.loads(json_str)
    if "metadata" not in data and "attrs" in data:
        raise ValueError("Legacy json graph format detected, we don't support it anymore.")
    return json.dumps(data, indent=2)
