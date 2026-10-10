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
    "arith.Analyzer": "sym.Analyzer",
    "arith.CanonicalExpr": "sym.CanonicalExpr",
    "arith.ConstIntBound": "sym.ConstIntBound",
    "arith.IntervalSet": "sym.IntervalSet",
    "arith.IterMapExpr": "sym.IterMapExpr",
    "arith.IterMapResult": "sym.IterMapResult",
    "arith.IterMark": "sym.IterMark",
    "arith.IterSplitExpr": "sym.IterSplitExpr",
    "arith.IterSumExpr": "sym.IterSumExpr",
    "arith.ModularSet": "sym.ModularSet",
    "arith.PresburgerSet": "sym.PresburgerSet",
    "arith.RewriteSimplifierStats": "sym.RewriteSimplifierStats",
    "arith.SplitExpr": "sym.SplitExpr",
    "arith.SumExpr": "sym.SumExpr",
    "tirx.BufferRegion": "ir.TensorRegion",
    "tirx.BufferRegionType": "ir.TensorRegionType",
    "tirx.IterVar": "s_tir.IterVar",
    "tirx.SBlock": "s_tir.SBlock",
    "tirx.SBlockRealize": "s_tir.SBlockRealize",
    "tirx.MatchBufferRegion": "s_tir.MatchTensorRegion",
    "s_tir.MatchBufferRegion": "s_tir.MatchTensorRegion",
    "tirx.TensorIntrin": "s_tir.TensorIntrin",
    "tirx.Cast": "ir.prim.Cast",
    "tirx.Add": "ir.prim.Add",
    "tirx.Sub": "ir.prim.Sub",
    "tirx.Mul": "ir.prim.Mul",
    "tirx.Div": "ir.prim.Div",
    "tirx.Mod": "ir.prim.Mod",
    "tirx.FloorDiv": "ir.prim.FloorDiv",
    "tirx.FloorMod": "ir.prim.FloorMod",
    "tirx.Min": "ir.prim.Min",
    "tirx.Max": "ir.prim.Max",
    "tirx.EQ": "ir.prim.EQ",
    "tirx.NE": "ir.prim.NE",
    "tirx.LT": "ir.prim.LT",
    "tirx.LE": "ir.prim.LE",
    "tirx.GT": "ir.prim.GT",
    "tirx.GE": "ir.prim.GE",
    "tirx.And": "ir.prim.And",
    "tirx.Or": "ir.prim.Or",
    "tirx.Not": "ir.prim.Not",
    "tirx.Select": "ir.prim.Select",
    "tirx.Let": "ir.prim.Let",
    "tirx.Ramp": "ir.prim.Ramp",
    "tirx.Broadcast": "ir.prim.Broadcast",
    "tirx.Shuffle": "ir.prim.Shuffle",
    "tirx.CommReducer": "te.CommReducer",
    "tirx.Reduce": "te.Reduce",
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
    nodes = data.get("nodes", [])
    tensor_region_type = None
    renamed_metadata = {}
    for node in nodes:
        if node.get("type") == "tirx.BufferRegion":
            fields = node.get("data")
            if not isinstance(fields, dict) or "buffer" not in fields:
                raise ValueError("Legacy tirx.BufferRegion requires a buffer field")
            fields["source"] = fields.pop("buffer")
            # Typed BufferRegion already carries type/loc.  Before it became
            # an Expr, it had only buffer/region; supply that form's defaults
            # by appending a type node so existing graph indices stay intact.
            if "ty" not in fields:
                if tensor_region_type is None:
                    tensor_region_type = len(nodes)
                    nodes.append({"type": "ir.TensorRegionType", "data": {}})
                fields["ty"] = tensor_region_type
        node["type"] = _PRIM_TYPE_KEY_RENAMES.get(node.get("type"), node.get("type"))
        # S-TIR tensor declarations keep their historical graph references while
        # migrating the type key and tensor-valued fields together.
        if node.get("type") == "s_tir.MatchTensorRegion":
            fields = node.get("data", {})
            if "buffer" in fields:
                fields["tensor"] = fields.pop("buffer")
        elif node.get("type") == "s_tir.SBlock":
            fields = node.get("data", {})
            for old, new in (
                ("alloc_buffers", "alloc_tensors"),
                ("match_buffers", "match_tensors"),
            ):
                if old in fields:
                    fields[new] = fields.pop(old)
        # Rename only keys in owning IR metadata maps. Strings and maps may be
        # shared with unrelated payloads, so retain the original graph nodes.
        fields = node.get("data", {})
        metadata_field = None
        if node.get("type") == "s_tir.SBlock":
            metadata_field = "annotations"
            key_renames = {
                "s_tir.buffer_allocated_addr": "s_tir.tensor_allocated_addr",
                "buffer_dim_align": "tensor_dim_align",
            }
        elif node.get("type") == "tirx.Function":
            metadata_field = "attrs"
            key_renames = {"layout_free_buffers": "layout_free_tensors"}
        elif node.get("type") == "ir.Call" and nodes[fields["op"]] == {
            "type": "ir.Op",
            "data": "tirx.alloc_tensor",
        }:
            metadata_field = "attrs"
            key_renames = {
                "buffer_data_alignment": "tensor_data_alignment",
                "buffer_dim_align": "tensor_dim_align",
            }
        if metadata_field and isinstance(fields, dict) and metadata_field in fields:
            metadata_index = fields[metadata_field]
            policy = (metadata_index, tuple(key_renames.items()))
            if policy in renamed_metadata:
                fields[metadata_field] = renamed_metadata[policy]
                continue
            metadata = nodes[metadata_index]
            map_index = metadata_index
            if metadata.get("type") == "ir.DictAttrs":
                map_index = metadata["data"]["__dict__"]
            mapping = nodes[map_index]
            if mapping.get("type") == "ffi.Map":
                entries = list(mapping["data"])
                changed = False
                for index in range(0, len(entries), 2):
                    key = nodes[entries[index]]
                    if key.get("type") == "ffi.String" and key["data"] in key_renames:
                        entries[index] = len(nodes)
                        nodes.append({"type": "ffi.String", "data": key_renames[key["data"]]})
                        changed = True
                if changed:
                    replacement = len(nodes)
                    nodes.append({"type": "ffi.Map", "data": entries})
                    if metadata.get("type") == "ir.DictAttrs":
                        nodes.append({"type": "ir.DictAttrs", "data": {"__dict__": replacement}})
                        replacement = len(nodes) - 1
                    fields[metadata_field] = replacement
                    renamed_metadata[policy] = replacement
        if node.get("type") == "relax.expr.Var":
            node["type"] = "ir.Var"
        elif node.get("type") == "tirx.Var":
            node["type"] = "ir.Var"
        if node.get("type") in ("ir.Var", "relax.expr.DataflowVar"):
            fields = node.get("data", {})
            if "name_hint" in fields and "name" not in fields:
                fields["name"] = fields.pop("name_hint")
    # IterVar became a primitive-typed OpaqueExpr.  Its value type is the
    # contained variable type, including for older metadata-only graphs.
    for node in nodes:
        if node.get("type") == "s_tir.IterVar":
            fields = node.get("data", {})
            if "ty" not in fields:
                fields["ty"] = nodes[fields["var"]]["data"]["ty"]
    return json.dumps(data, indent=2)
