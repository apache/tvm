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

_PRIM_TYPE_KEY_RENAMES = {
    "tirx.StringImm": "prim.StringImm",
    "tirx.Cast": "prim.Cast",
    "tirx.Add": "prim.Add",
    "tirx.Sub": "prim.Sub",
    "tirx.Mul": "prim.Mul",
    "tirx.Div": "prim.Div",
    "tirx.Mod": "prim.Mod",
    "tirx.FloorDiv": "prim.FloorDiv",
    "tirx.FloorMod": "prim.FloorMod",
    "tirx.Min": "prim.Min",
    "tirx.Max": "prim.Max",
    "tirx.EQ": "prim.EQ",
    "tirx.NE": "prim.NE",
    "tirx.LT": "prim.LT",
    "tirx.LE": "prim.LE",
    "tirx.GT": "prim.GT",
    "tirx.GE": "prim.GE",
    "tirx.And": "prim.And",
    "tirx.Or": "prim.Or",
    "tirx.Not": "prim.Not",
    "tirx.Select": "prim.Select",
    "tirx.Let": "prim.Let",
    "tirx.Ramp": "prim.Ramp",
    "tirx.Broadcast": "prim.Broadcast",
    "tirx.Shuffle": "prim.Shuffle",
    "tirx.CommReducer": "te.CommReducer",
    "tirx.Reduce": "te.Reduce",
}
_PRIM_TYPE_KEY_RENAMES.update(
    {
        f"ir.{target}": target
        for target in _PRIM_TYPE_KEY_RENAMES.values()
        if target.startswith("prim.")
    }
)

_PRIM_OP_RENAMES = {
    "ir.prim.likely": "prim.likely",
    "ir.prim.bitwise_and": "prim.bitwise_and",
    "ir.prim.bitwise_or": "prim.bitwise_or",
    "ir.prim.bitwise_xor": "prim.bitwise_xor",
    "ir.prim.bitwise_not": "prim.bitwise_not",
    "ir.prim.shift_left": "prim.shift_left",
    "ir.prim.shift_right": "prim.shift_right",
    "ir.prim.if_then_else": "prim.if_then_else",
    "ir.prim.vscale": "prim.vscale",
    "tirx.ceil": "prim.ceil",
    "tirx.log2": "prim.log2",
}


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

    # `ir.Var` is the sole runtime variable node.  Keep `tvm.ir.load_json`
    # compatible with the pre-unification Relax/TIRx schemas and with graphs
    # written before the canonical Var field was renamed to `name`.  Rewriting
    # nodes in place preserves node indices and shared references.
    for node in data.get("nodes", []):
        node["type"] = _PRIM_TYPE_KEY_RENAMES.get(node.get("type"), node.get("type"))
        if node.get("type") == "ir.Op":
            node["data"] = _PRIM_OP_RENAMES.get(node["data"], node["data"])
        if node.get("type") == "relax.expr.Var":
            node["type"] = "ir.Var"
        elif node.get("type") == "tirx.Var":
            node["type"] = "ir.Var"
        if node.get("type") in ("ir.Var", "relax.expr.DataflowVar"):
            fields = node.get("data", {})
            if "name_hint" in fields and "name" not in fields:
                fields["name"] = fields.pop("name_hint")
    return json.dumps(data, indent=2)
