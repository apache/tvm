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
    for node in nodes:
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
