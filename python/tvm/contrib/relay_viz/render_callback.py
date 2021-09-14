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
"""Default render callback rules"""
import tvm
from tvm import relay

UNKNOWN_TYPE = "unknown"


class RenderCallback:
    """This is the default callback rules, which is also the _bokeh backend drawing way"""

    def __init__(self):
        self.render_rules = {}
        self.build_rules()

    def var_node(self, node, relay_param, node_to_id):
        """Render rule for a relay var node"""
        node_id = node_to_id[node]
        name_hint = node.name_hint
        node_detail = ""
        node_type = "Var(Param)" if name_hint in relay_param else "Var(Input)"
        if node.type_annotation is not None:
            if hasattr(node.type_annotation, "shape"):
                shape = tuple(map(int, node.type_annotation.shape))
                dtype = node.type_annotation.dtype
                node_detail = "name_hint: {}\nshape: {}\ndtype: {}".format(name_hint, shape, dtype)
            else:
                node_detail = "name_hint: {}\ntype_annotation: {}".format(
                    name_hint, node.type_annotation
                )
        graph_info = [node_id, node_type, node_detail]
        edge_info = []
        return graph_info, edge_info

    def function_node(self, node, relay_param, node_to_id):  # pylint: disable=unused-argument
        node_id = node_to_id[node]
        graph_info = [node_id, "Func", str(node.params)]
        edge_info = [[node_to_id[node.body], node_id]]
        return graph_info, edge_info

    def call_node(self, node, relay_param, node_to_id):  # pylint: disable=unused-argument
        """Render rule for a relay call node"""
        node_id = node_to_id[node]
        op_name = UNKNOWN_TYPE
        node_detail = []
        if isinstance(node.op, tvm.ir.Op):
            op_name = node.op.name
            if node.attrs:
                node_detail = ["{}: {}".format(k, node.attrs.get_str(k)) for k in node.attrs.keys()]
        elif isinstance(node.op, relay.Function):
            func_attrs = node.op.attrs
            op_name = "Anonymous Func"
            if func_attrs:
                node_detail = ["{}: {}".format(k, func_attrs.get_str(k)) for k in func_attrs.keys()]
                # "Composite" might from relay.transform.MergeComposite
                if "Composite" in func_attrs.keys():
                    op_name = func_attrs["Composite"]
        elif isinstance(node.op, relay.GlobalVar):
            op_name = "GlobalVar"
            node_detail = [f"GlobalVar.name_hint: {node.op.name_hint}"]
        else:
            op_name = str(type(node.op)).split(".")[-1].split("'")[0]

        graph_info = [node_id, f"Call {op_name}", "\n".join(node_detail)]
        args = [node_to_id[arg] for arg in node.args]
        edge_info = [[arg, node_id] for arg in args]
        return graph_info, edge_info

    def let_node(self, node, relay_param, node_to_id):  # pylint: disable=unused-argument
        node_id = node_to_id[node]
        graph_info = [node_id, "Let", ""]
        edge_info = [[node_to_id[node.value], node_id]]
        edge_info.append([node_id, node_to_id[node.var]])
        return graph_info, edge_info

    def global_var_node(self, node, relay_param, node_to_id):  # pylint: disable=unused-argument
        graph_info = []
        edge_info = []
        return graph_info, edge_info

    def if_node(self, node, relay_param, node_to_id):  # pylint: disable=unused-argument
        graph_info = []
        edge_info = []
        return graph_info, edge_info

    def tuple_node(self, node, relay_param, node_to_id):  # pylint: disable=unused-argument
        node_id = node_to_id[node]
        graph_info = [node_id, "Tuple", ""]
        edge_info = [[node_to_id[field], node_id] for field in node.fields]
        return graph_info, edge_info

    def tuple_get_item_node(self, node, relay_param, node_to_id):  # pylint: disable=unused-argument
        node_id = node_to_id[node]
        graph_info = [node_id, "TupleGetItem", "idx: {}".format(node.index)]
        edge_info = [[node_to_id[node.tuple_value], node_id]]
        return graph_info, edge_info

    def constant_node(self, node, relay_param, node_to_id):  # pylint: disable=unused-argument
        node_id = node_to_id[node]
        node_detail = "shape: {}, dtype: {}".format(node.data.shape, node.data.dtype)
        graph_info = [node_id, "Const", "\n".join(node_detail)]
        edge_info = []
        return graph_info, edge_info

    def op_node(self, node, relay_param, node_to_id):  # pylint: disable=unused-argument
        graph_info = []
        edge_info = []
        return graph_info, edge_info

    def build_rules(self):
        self.render_rules = {
            tvm.relay.Function: self.function_node,
            tvm.relay.expr.Call: self.call_node,
            tvm.relay.expr.Let: self.let_node,
            tvm.relay.expr.Var: self.var_node,
            tvm.relay.expr.GlobalVar: self.global_var_node,
            tvm.relay.expr.If: self.if_node,
            tvm.relay.expr.Tuple: self.tuple_node,
            tvm.relay.expr.TupleGetItem: self.tuple_get_item_node,
            tvm.relay.expr.Constant: self.constant_node,
            tvm.ir.Op: self.op_node,
        }

    def get_rules(self):
        return self.render_rules
