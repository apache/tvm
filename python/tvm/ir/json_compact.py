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
            if isinstance(f, list):
                for fpass in f:
                    item = fpass(item, nodes)
            elif f:
                item = f(item, nodes)
            nodes[idx] = item
        data["attrs"]["tvm_version"] = to_ver
        return data

    return _updater


def create_updater_16_to_17():
    """
    Create an update to upgrade json from v0.16 to v0.17

    Returns
    -------
    fupdater : function
        The updater function
    """

    def _update_predicate_argument(item, nodes):
        null_value_idx = 0
        null_value = nodes[null_value_idx]
        assert str(null_value) == "{'type_key': ''}", f"Expected a null value but got {null_value}"
        item["attrs"]["predicate"] = str(null_value_idx)
        return item

    node_map = {
        "tir.BufferLoad": _update_predicate_argument,
        "tir.BufferStore": _update_predicate_argument,
    }

    return create_updater(node_map, "0.16", "0.17")


def create_updater_15_to_16():
    """
    Create an update to upgrade json from v0.15 to v0.16

    Returns
    -------
    fupdater : function
        The updater function
    """

    def _update_lanes_obj(item, nodes):
        lanes = item["attrs"]["lanes"]
        new_idx = len(nodes)
        item["attrs"]["lanes"] = str(new_idx)
        lanes_node = {
            "type_key": "IntImm",
            "attrs": {"dtype": "int32", "span": "0", "value": lanes},
        }
        nodes.append(lanes_node)
        return item

    node_map = {"tir.Ramp": _update_lanes_obj, "tir.Broadcast": _update_lanes_obj}

    return create_updater(node_map, "0.15", "0.16")


def create_updater_13_to_14():
    """Create an update to upgrade json from v0.13 to v0.14 for TVM Unity"""

    def _update_vdevice(item, _):
        if "vdevice" not in item["attrs"]:
            item["attrs"]["vdevice"] = "0"
        return item

    node_map = {
        "relax.TensorStructInfo": _update_vdevice,
    }

    return create_updater(node_map, "0.13", "0.14")


def create_updater_08_to_09():
    """
    Create an update to upgrade json from v0.8 to v0.9

    Returns
    -------
    fupdater : function
        The updater function
    """

    def _initialize_virtual_device(item, _):
        if "virtual_device_" not in item["attrs"]:
            item["attrs"]["virtual_device_"] = "0"
        return item

    node_map = {
        # Base IR
        "GlobalVar": _initialize_virtual_device,
    }

    return create_updater(node_map, "0.8", "0.9")


def create_updater_07_to_08():
    """Create an update to upgrade json from v0.7 to v0.8"""

    def _initialize_module_attributes(item, _):
        assert item["type_key"] == "IRModule", "Only initialize the attributes for IRModules"
        if "attrs" not in item["attrs"]:
            item["attrs"]["attrs"] = "0"
        return item

    node_map = {"IRModule": _initialize_module_attributes}
    return create_updater(node_map, "0.7", "0.8")


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

    def _from_version(data):
        return data["attrs"]["tvm_version"]

    if _from_version(data).startswith("0.6"):
        data = create_updater_06_to_07()(data)
    if _from_version(data).startswith("0.7"):
        data = create_updater_07_to_08()(data)
    if _from_version(data).startswith("0.8"):
        data = create_updater_08_to_09()(data)
    if _from_version(data).startswith("0.9"):
        data = create_updater({}, "0.9", "0.10")(data)
    if _from_version(data).startswith("0.10"):
        data = create_updater({}, "0.10", "0.11")(data)
    if _from_version(data).startswith("0.11"):
        data = create_updater({}, "0.11", "0.12")(data)
    if _from_version(data).startswith("0.12"):
        data = create_updater({}, "0.12", "0.13")(data)
    if _from_version(data).startswith("0.13"):
        data = create_updater_13_to_14()(data)
    if _from_version(data).startswith("0.14"):
        data = create_updater({}, "0.14", "0.15")(data)
    if _from_version(data).startswith("0.15"):
        data = create_updater_15_to_16()(data)
    if _from_version(data).startswith("0.16"):
        data = create_updater_16_to_17()(data)

    return json.dumps(data, indent=2)
