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
        assert data["attrs"]["tvm_version"].startswith(from_ver)
        nodes = data["nodes"]
        for idx, item in enumerate(nodes):
            f = node_map.get(item["type_key"], None)
            if f:
                nodes[idx] = f(item, nodes)
        data["attrs"]["tvm_version"] = to_ver
        return data
    return _updater


def create_updater_06_to_07():
    """Create an update to upgrade json from v0.6 to v0.7

    Returns
    -------
    fupdater : function
        The updater function
    """
    def _ftype_var(item, nodes):
        vindex = int(item["attrs"]["var"])
        item["attrs"]["name_hint"] = nodes[vindex]["attrs"]["name"]
        # set vindex to null
        nodes[vindex]["type_key"] = ""
        del item["attrs"]["var"]
        return item

    node_map = {
        "relay.TypeVar": _ftype_var,
        "relay.GlobalTypeVar": _ftype_var,
    }
    return create_updater(node_map, "0.6", "0.7")


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
    from_version = data["attrs"]["tvm_version"]
    if from_version.startswith("0.6"):
        data = create_updater_06_to_07()(data)
    else:
        raise ValueError("Cannot update from version %s" % from_version)
    return json.dumps(data, indent=2)
