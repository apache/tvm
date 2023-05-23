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
# pylint: disable=import-outside-toplevel, nested-min-max
"""Tensor Expression Debug Display (TEDD), visualizing Tensor Expression"""
import html
import json
import warnings
from graphviz import Digraph
from graphviz import Source
import tvm

TVMDD_TABLE_BODY_WIDTH = 30
# Must match enum IterVarType defined in include/tvm/expr.h
ITERVAR_TYPE_STRING_MAP = {
    0: ("kDataPar", "#FFFFFF"),
    1: ("kThreadIndex", "#2980B9"),
    2: ("kCommReduce", "#FAD7A0"),
    3: ("kOrdered", "#D35400"),
    4: ("kOpaque", "#ABB2B9"),
    5: ("kUnrolled", "#D2B4DE"),
    6: ("kVectorized", "#AED6F1"),
    7: ("kParallelized", "#F5B7B1"),
    8: ("kTensorized", "#A9DFBF"),
}

PALETTE = {
    0: "#000000",
    1: "#922B21",
    2: "#76448A",
    3: "#1F618D",
    4: "#148F77",
    5: "#B7950B",
    6: "#AF601A",
    7: "#F5B7B1",
    8: "#A9DFBF",
}

PALETTE_SIZE = 9


def dom_path_to_string(dom_path, prefix=""):
    path_string = prefix
    for index in dom_path:
        path_string = path_string + "_" + str(index)
    return path_string


def insert_dot_id(sch):
    """Insert unique ID for each node in the DOM tree.
    They are used as Dot node ID.
    """
    for stage_idx, stage in enumerate(sch["stages"]):
        dom_path = [stage_idx]
        stage["id"] = dom_path_to_string(dom_path, stage["type"])
        for itervar_idx, itervar in enumerate(stage["all_itervars"]):
            dom_path = [stage_idx, itervar_idx]
            itervar["id"] = dom_path_to_string(dom_path, itervar["type"])
        for rel_idx, rel in enumerate(stage["relations"]):
            dom_path = [stage_idx, rel_idx]
            rel["id"] = dom_path_to_string(dom_path, rel["type"])
        for tensor_idx, tensor in enumerate(stage["output_tensors"]):
            dom_path = [stage_idx, tensor_idx]
            tensor["id"] = dom_path_to_string(dom_path, tensor["type"])
    return sch


def itervar_equal(iv_a, iv_b):
    """A helper method that compares the equality of two iterative variables"""
    # Adopt the following method to assure the equality between two itervars.
    # The plain comparison might fail (i.e. iv_a == iv_b) after the change of
    # domain bounds from InferBound.
    def _var_equal(v_a, v_b):
        condtions = [
            v_a.name == v_b.name,
            v_a.dtype == v_b.dtype,
            v_a.type_annotation == v_b.type_annotation,
        ]
        return all(c for c in condtions)

    condtions = [
        _var_equal(iv_a.var, iv_b.var),
        iv_a.iter_type == iv_b.iter_type,
        iv_a.thread_tag == iv_b.thread_tag,
    ]
    return all(c for c in condtions)


class ObjectManager:
    """A helper class tracking schedule objects, e.g. stage, IterVar,
    relationship, and tensor, to their DOM path."""

    def __init__(self, sch):
        self.dict = {}
        for stage_idx, stage in enumerate(sch.stages):
            self.dict[stage] = [stage_idx]
            for itervar_idx, itervar in enumerate(stage.all_iter_vars):
                self.dict[itervar] = [stage_idx, itervar_idx]
                # the itervars of leaf should also be mapped to the original one
                for leaf_iv in stage.leaf_iter_vars:
                    if itervar_equal(leaf_iv, itervar):
                        self.dict[leaf_iv] = [stage_idx, itervar_idx]
            for rel_idx, rel in enumerate(stage.relations):
                self.dict[rel] = [stage_idx, rel_idx]
            for tensor_idx in range(stage.op.num_outputs):
                self.dict[frozenset({stage.op.name, tensor_idx})] = [stage_idx, tensor_idx]

    def get_dom_path(self, obj):
        if obj is None:
            return None
        assert obj in self.dict, "Node is no found."
        return self.dict[obj]


def get_or_create_dot_id(obj, prefix="", assert_on_missing=False):
    """If obj's ID has been registered, return it.
    If not, either assert or create a unique and legal ID, register and
    return it, according to assert_on_missing.
    ID must be a unique and legal Dotty ID.

     Parameters
     ----------
     obj : objet
                 Serve as the key to the ID.

     prefix : string
                 Prefix to attach to the ID.  Usually use obj's non-unique
                 name as prefix.

     assert_on_missing : bool
                 Assert or not if object doesn't have a registered ID.
    """
    prefix = prefix.replace(".", "_")
    if not hasattr(get_or_create_dot_id, "obj_id_dict"):
        get_or_create_dot_id.obj_id_dict = {}
    if obj not in get_or_create_dot_id.obj_id_dict:
        if assert_on_missing:
            assert False, "dot_id " + str(obj) + " has not been registered."
        else:
            get_or_create_dot_id.obj_id_dict[obj] = prefix + hex(id(obj))
    return get_or_create_dot_id.obj_id_dict[obj]


def get_port_id(is_input, index):
    return "I_" + str(index) if is_input else "O_" + str(index)


def get_itervar_type_info(iter_type):
    assert iter_type < len(ITERVAR_TYPE_STRING_MAP), "Unknown IterVar type: " + str(iter_type)
    return ITERVAR_TYPE_STRING_MAP[iter_type]


def get_itervar_label_color(itervar, iv_type):
    type_info = get_itervar_type_info(iv_type)
    return (
        linebrk(str(itervar["name"]) + "(" + type_info[0] + ")", TVMDD_TABLE_BODY_WIDTH),
        type_info[1],
    )


def linebrk(s, n):
    """Break input string s with <br/> for every n charactors."""
    result = ""
    j = 0
    for i, c in enumerate(s):
        if j == n and i != len(s) - 1:
            result = result + "\n"
            j = 0
        j = j + 1
        result = result + c
    result = html.escape(str(result), quote=True)
    result = result.replace("\n", "<br/>")
    return result


def create_graph(name="", rankdir="BT"):
    graph = Digraph(name=name)
    graph.graph_attr["rankdir"] = rankdir
    return graph


def itervar_label(itervar, index, index_color, label):
    return (
        '<TR><TD PORT="'
        + itervar["id"]
        + '" BGCOLOR="'
        + index_color
        + '">'
        + str(index)
        + '</TD><TD BGCOLOR="white" PORT="itervar">'
        + label
        + "<br/>"
        + str(itervar["properties"]["range"])
        + "</TD></TR>"
    )


def stage_label(stage):
    return stage["name"] + "<br/>Scope: " + stage["properties"]["scope"]


def legend_label():
    """Generate legend labels."""
    label = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'
    for iter_type in ITERVAR_TYPE_STRING_MAP:
        name, color = ITERVAR_TYPE_STRING_MAP[iter_type]
        label += (
            '<TR><TD BGCOLOR="' + color + '"></TD>' + '<TD BGCOLOR="white">' + name + "</TD></TR>"
        )
    label += "</TABLE>>"
    return label


def leaf_itervars(stage):
    filtered = filter(lambda x: (x["index"] >= 0), stage["all_itervars"])
    return sorted(filtered, key=lambda x: x["index"])


def legend_dot(g):
    with g.subgraph(name="cluster_legend") as subgraph:
        subgraph.attr(label="Legend")
        label = legend_label()
        subgraph.node("legend", label, shape="none", margin="0")


def extract_dom_for_viz(sch, need_range=True):
    json_str = dump_json(sch, need_range)
    s = json.loads(json_str)
    s = insert_dot_id(s)
    return s


def dump_graph(dot_string, show_svg=True, dot_file_path="", output_dot_string=False):
    """Output dot_string in various formats."""
    if dot_file_path:
        try:
            dot_file = open(dot_file_path, "w+")
            dot_file.write(dot_string)
            dot_file.close()
        except IOError:
            print("Cannot open file: " + dot_file_path)
    if show_svg:
        from IPython.display import display
        from IPython.display import SVG

        src = Source(dot_string)
        display(SVG(src.pipe(format="svg")))
    if output_dot_string:
        return dot_string
    return None


def dump_json(sch, need_range):
    """Serialize data for visualization from a schedule in JSON format.

    Parameters
    ----------
    sch : schedule
                The schedule object to serialize

    Returns
    -------
    json : string
        Serialized JSON string
    """

    def encode_itervar(itervar, stage, index, range_map):
        """Extract and encode IterVar visualization data to a dictionary"""
        ivrange = range_map[itervar] if range_map is not None and itervar in range_map else None
        bind_thread = None
        tensor_intrin = None
        if itervar in stage.iter_var_attrs:
            attr = stage.iter_var_attrs[itervar]
            iv_type = attr.iter_type
            # binding
            bind_thread = str(attr.bind_thread.var) if attr.bind_thread is not None else None
            # tensorization
            if attr.tensor_intrin is not None:
                tensor_intrin = str(attr.tensor_intrin.body)
                # remove the final \n
                tensor_intrin = tensor_intrin[0:-1] if tensor_intrin[-1] == "\n" else tensor_intrin
            else:
                tensor_intrin = None
        else:
            iv_type = itervar.iter_type
        itervar_dict = {
            "type": "IterVar",
            "index": index,
            "name": str(itervar.var),
            "itervar_type": iv_type,
            "properties": {
                "thread": bind_thread,
                "intrin": tensor_intrin,
                "range": str(ivrange) if ivrange is not None else "range(N/A)",
            },
        }
        return itervar_dict

    def encode_itervars(stage, range_map):
        """Extract and encode IterVars visualization data from a stage to a dictionary"""

        def get_leaf_itervar_index(itervar, leaf_iv):
            for leaf_index, ivar in enumerate(leaf_iv):
                if itervar_equal(ivar, itervar):
                    return leaf_index
            return -1

        itervars = []
        for itervar in stage.all_iter_vars:
            leaf_index = get_leaf_itervar_index(itervar, stage.leaf_iter_vars)
            itervars.append(encode_itervar(itervar, stage, leaf_index, range_map))
        return itervars

    def encode_itervar_relation(obj_manager, rel):
        """Extract and encode IterVar Relationship visualization data to a dictionary"""
        rel_type = type(rel)
        if rel_type is tvm.te.schedule.Split:
            node_type = "Split_Relation"
            rel_dict = {
                "type": node_type,
                "parent": obj_manager.get_dom_path(rel.parent),
                "outer": obj_manager.get_dom_path(rel.outer),
                "inner": obj_manager.get_dom_path(rel.inner),
            }
        elif rel_type is tvm.te.schedule.Fuse:
            node_type = "Fuse_Relation"
            rel_dict = {
                "type": node_type,
                "fused": obj_manager.get_dom_path(rel.fused),
                "outer": obj_manager.get_dom_path(rel.outer),
                "inner": obj_manager.get_dom_path(rel.inner),
            }
        elif rel_type is tvm.te.schedule.Singleton:
            node_type = "Singleton_Relation"
            rel_dict = {
                "type": node_type,
                "iter": obj_manager.get_dom_path(rel.iter),
            }
        else:
            return None
        return rel_dict

    def encode_itervar_relations(obj_manager, stage):
        relations = []
        for i in range(len(stage.relations)):
            rel = encode_itervar_relation(obj_manager, stage.relations[i])
            if rel is not None:
                relations.append(rel)
        return relations

    def encode_tensor(obj_manager, tensor, stage):
        """Extract and encode tensor visualization data to a dictionary"""
        tensor_dict = {
            "type": "Tensor",
            "source": obj_manager.get_dom_path(stage),
            "value_index": tensor.value_index,
            "shape": str(tensor.op.output(tensor.value_index).shape),
            "data_type": tensor.op.output(tensor.value_index).dtype,
        }
        return tensor_dict

    def encode_tensors(obj_manager, stage):
        tensors = []
        for i in range(stage.op.num_outputs):
            tensor = stage.op.output(i)
            tensors.append(encode_tensor(obj_manager, tensor, stage))
        tensors.sort(key=lambda tensor: tensor["value_index"])
        return tensors

    def encode_stage(obj_manager, stage, range_map):
        """Extract and encode stage visualization data to a dictionary"""
        stage_dict = {
            "type": "Stage",
            "name": stage.op.name,
            "attaching_to": obj_manager.get_dom_path(stage.attach_ivar),
            "compute": str(stage.op.body) if hasattr(stage.op, "body") else None,
            "properties": {
                "scope": stage.scope,
            },
            "all_itervars": encode_itervars(stage, range_map),
            "relations": encode_itervar_relations(obj_manager, stage),
            "input_tensors": [
                obj_manager.get_dom_path(frozenset({tensor.op.name, tensor.value_index}))
                for tensor in stage.op.input_tensors
            ],
            "output_tensors": encode_tensors(obj_manager, stage),
        }
        return stage_dict

    def encode_schedule(sch, need_range):
        """Extract and encode data from a schedule for visualization to a nested dictionary.
        It is useful for JSON to serialize schedule.

            Parameters
            ----------
            sch : schedule
                        The schedule object to extract

            Returns
            -------
            dict : dictionary
                A nested dictionary
        """
        assert isinstance(
            sch, tvm.te.schedule.Schedule
        ), "Input is not a tvm.te.schedule.Schedule object."
        range_map = None
        if need_range:
            try:
                range_map = tvm.te.schedule.InferBound(sch)
            except tvm._ffi.base.TVMError as expt:
                warnings.warn(
                    "Ranges are not available, because InferBound fails with the following error:\n"
                    + str(expt)
                )

        obj_manager = ObjectManager(sch)
        stages = []
        for stage in sch.stages:
            stages.append(encode_stage(obj_manager, stage, range_map))
        return {
            "type": "Schedule",
            "stages": stages,
        }

    return json.dumps(sch, default=lambda s: encode_schedule(s, need_range))


def viz_schedule_tree(sch, show_svg=False, dot_file_path="", output_dot_string=False):
    """Top level API to render schedule tree

    Parameters
    ----------
    sch : schedule
                The schedule object to visualize

    show_svg : bool
                Display graph as SVG, useful for Jupyter notebooks.

    dot_file_path : string
                Dot file to save the graph.

    output_dot_string : bool
                Return dot file content or an empty string.

    Returns
    -------
    dot_string : string
        Dot file content or an empty string according to output_dot_string

    Examples
    --------
    The following code writes a schedule tree to a dot file.

    .. code-block:: python
        tedd.viz_schedule_tree(s, dot_file_path = '/tmp/example.dot')

    Use the following code to render a SVG graph in a Jupyter notebook.

    .. code-block:: python
        tedd.viz_schedule_tree(s, show_svg = True)
    """

    def create_schedule_tree_graph(name=""):
        return create_graph(name=name, rankdir="BT")

    def root_dot(g):
        g.node("ROOT", "ROOT", shape="oval", margin="0")

    def stage_node_dot(g, stage):
        node_label = stage_node_label(stage)
        g.node(stage["id"], node_label, shape="none", margin="0")

    def stage_node_label(stage):
        """Return a html format label for the given stage."""
        label = (
            '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" '
            'CELLPADDING="4"> <TR><TD BGCOLOR="lightgrey" '
            'COLSPAN="2" PORT="stage">' + stage_label(stage) + "</TD></TR>"
        )

        for leafiv in leaf_itervars(stage):
            iv_type = leafiv["itervar_type"]
            var_attr_label = ""
            if "thread" in leafiv["properties"] and leafiv["properties"]["thread"] is not None:
                var_attr_label = (
                    var_attr_label
                    + '<br/><font color="#2980B9">('
                    + str(leafiv["properties"]["thread"])
                    + ")</font>"
                )
            if "intrin" in leafiv["properties"] and leafiv["properties"]["intrin"] is not None:
                var_attr_label = (
                    var_attr_label
                    + "<br/>"
                    + linebrk(
                        "(tensor_intrin:" + str(leafiv["properties"]["intrin"]) + ")",
                        TVMDD_TABLE_BODY_WIDTH,
                    )
                )
            var_label, color = get_itervar_label_color(leafiv, iv_type)
            label += itervar_label(leafiv, leafiv["index"], color, var_label + var_attr_label)
        if stage["compute"] is not None:
            label += (
                '<TR><TD COLSPAN="2">'
                + linebrk(str(stage["compute"]), TVMDD_TABLE_BODY_WIDTH)
                + "</TD></TR>"
            )
        label += "</TABLE>>"
        return label

    def compute_at_dot(g, stage):
        """If the given stage attaches to another stage, create an edge from it
        stage to its attach point; otherwise, create an edge to the ROOT.
        """
        src = stage["id"]
        dst = (
            dom_path_to_string([stage["attaching_to"][0]], "Stage")
            + ":"
            + dom_path_to_string(stage["attaching_to"], "IterVar")
            if stage["attaching_to"] is not None
            else "ROOT"
        )
        color = (
            PALETTE[stage["attaching_to"][1] + 1]
            if stage["attaching_to"] is not None and stage["attaching_to"][1] < PALETTE_SIZE - 1
            else PALETTE[0]
        )
        g.edge(src, dst, color=color)

    graph = create_schedule_tree_graph("Schedule Tree")
    s = extract_dom_for_viz(sch)
    legend_dot(graph)
    for stage in s["stages"]:
        stage_node_dot(graph, stage)
    for stage in s["stages"]:
        compute_at_dot(graph, stage)
    root_dot(graph)
    return dump_graph(graph.source, show_svg, dot_file_path, output_dot_string)


def viz_itervar_relationship_graph(sch, show_svg=False, dot_file_path="", output_dot_string=False):
    """Top level API to render IterVar relationship graph

    Parameters
    ----------
    sch : schedule
                The schedule object to visualize

    show_svg : bool
                Display graph as SVG, useful for Jupyter notebooks.

    dot_file_path : string
                Dot file to save the graph.

    output_dot_string : bool
                Return dot file content or an empty string.

    Examples
    --------
    The following code writes Ian tervar relationship graph to a dot file.

    .. code-block:: python
        tedd.viz_def viz_itervar_relationship_graph(sch,
            (s, dot_file_path = '/tmp/example.dot')

    Use the following code to render a SVG graph in a Jupyter notebook.

    .. code-block:: python
        tedd.viz_def viz_itervar_relationship_graph(sch,
            (s, show_svg = True)
    """

    def create_itervar_relation_graph(name=""):
        return create_graph(name=name, rankdir="TB")

    def itervar_node_dot(g, itervar, iv_type, index):
        label = itervar_node_label(itervar, iv_type, index)
        g.node(itervar["id"], label, shape="none", margin="0")

    def itervar_node_label(itervar, iv_type, index):
        label = (
            '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" '
            'CELLPADDING="4">'
            + itervar_label(
                itervar,
                index,
                get_itervar_label_color(itervar, iv_type)[1],
                get_itervar_label_color(itervar, iv_type)[0],
            )
            + "</TABLE>>"
        )
        return label

    def itervar_relation_node_dot(g, node_id, node_label, input_ports, output_ports):
        label = itervar_relation_node_label(node_label, input_ports, output_ports)
        g.node(node_id, label, shape="none", margin="0")

    def itervar_relation_node_label(node_label, input_ports, output_ports):
        """Return a html format label for an itervar relationship node
        including node_label and input/output ports.
        """
        label = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" ' 'CELLPADDING="4">' + "<TR>"
        max_port_num = max(len(input_ports), len(output_ports))
        for i in range(max_port_num):
            if i < len(input_ports):
                input_port = input_ports[i]
                label += '<TD BGCOLOR="lightgrey" PORT="' + input_port + '">' + input_port + "</TD>"
            else:
                label += '<TD BGCOLOR="white"></TD>'
        label += "</TR>"
        label += (
            '<TR><TD BGCOLOR="white" COLSPAN="'
            + str(max_port_num)
            + '" PORT="relation">'
            + node_label
            + "</TD></TR>"
        )
        label += "<TR>"
        for i in range(max_port_num):
            if i < len(output_ports):
                output_port = output_ports[i]
                label += (
                    '<TD BGCOLOR="lightgrey" PORT="' + output_port + '">' + output_port + "</TD>"
                )
            else:
                label += '<TD BGCOLOR="white"></TD>'
        label += "</TR>"
        label += "</TABLE>>"
        return label

    def itervar_relation_dot(g, node, node_id):
        """Create an itervar relationship node."""
        node_type = node["type"]
        if node_type == "Split_Relation":
            node_type = "Split"
            itervar_relation_node_dot(g, node_id, node_type, ["Input"], ["Outer", "Inner"])
            parent = dom_path_to_string(node["parent"], "IterVar")
            outer = dom_path_to_string(node["outer"], "IterVar")
            inner = dom_path_to_string(node["inner"], "IterVar")
            g.edge(parent + ":itervar", node_id + ":Input")
            g.edge(node_id + ":Outer", outer + ":itervar")
            g.edge(node_id + ":Inner", inner + ":itervar")
        elif node_type == "Fuse_Relation":
            node_type = "Fuse"
            itervar_relation_node_dot(g, node_id, node_type, ["Outer", "Inner"], ["Fused"])
            fused = dom_path_to_string(node["fused"], "IterVar")
            outer = dom_path_to_string(node["outer"], "IterVar")
            inner = dom_path_to_string(node["inner"], "IterVar")
            g.edge(outer + ":itervar", node_id + ":Outer")
            g.edge(inner + ":itervar", node_id + ":Inner")
            g.edge(node_id + ":Fused", fused + ":itervar")
        elif node_type == "Singleton_Relation":
            node_type = "Singleton"
            itervar_relation_node_dot(g, node_id, node_type, [], ["Iter"])
            itervar = dom_path_to_string(node["inner"], "IterVar")
            g.edge(node_id + ":Iter", itervar + ":itervar")
        else:
            assert False, "Unknown IterVarRelationNode: " + node_type

    def stage_node_dot(g, stage):
        """Create a stage node."""
        with g.subgraph(name="cluster_" + stage["id"]) as subgraph:
            subgraph.attr(label=stage["name"])
            if stage["all_itervars"]:
                for itervar in stage["all_itervars"]:
                    iv_type = itervar["itervar_type"]
                    itervar_node_dot(subgraph, itervar, iv_type, itervar["index"])
                for rel in stage["relations"]:
                    node_id = rel["id"]
                    itervar_relation_dot(subgraph, rel, node_id)
            else:
                subgraph.node(stage["name"] + "_placeholder", style="invis")

    graph = create_itervar_relation_graph("IterVar Relationship Graph")
    s = extract_dom_for_viz(sch)
    legend_dot(graph)
    for stage in s["stages"]:
        stage_node_dot(graph, stage)

    return dump_graph(graph.source, show_svg, dot_file_path, output_dot_string)


def viz_dataflow_graph(sch, show_svg=False, dot_file_path="", output_dot_string=False):
    """Top level API to render dataflow graph

    Parameters
    ----------
    sch : schedule
                The schedule object to visualize

    show_svg : bool
                Display graph as SVG, useful for Jupyter notebooks.

    dot_file_path : string
                Dot file to save the graph.

    output_dot_string : bool
                Return dot file content or an empty string.

    Examples
    --------
    The following code writes a dataflow graph to a dot file.

    .. code-block:: python
        tedd.viz_dataflow_graph(s, dot_file_path = '/tmp/example.dot')

    Use the following code to render a SVG graph in a Jupyter notebook.

    .. code-block:: python
        tedd.viz_dataflow_graph(s, show_svg = True)"""

    def create_dataflow_graph(name=""):
        return create_graph(name=name, rankdir="LR")

    def tensor_node_dot(g, tensor):
        """Create a tensor node."""
        label = tensor_node_label(tensor)
        g.node(tensor["id"], label, shape="oval", margin="0")

    def tensor_node_label(tensor):
        """Return a html format label for the given tensor."""
        label = str(tensor["shape"]) + "\n" + str(tensor["data_type"])
        return label

    def stage_node_dot(g, stage):
        """Create a stage node."""
        label = stage_node_label(stage)
        g.node(stage["id"], label, shape="none", margin="0")

    def stage_node_label(stage):
        """Return a html format label for the given stage."""
        rows = max(1, max(len(stage["output_tensors"]), len(stage["input_tensors"])))
        label = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" ' 'CELLPADDING="4">'
        for i in range(rows):
            label += "<TR>"
            if i < len(stage["input_tensors"]):
                port_id = get_port_id(True, i)
                label += (
                    '<TD BGCOLOR="lightgrey" COLSPAN="2" PORT="' + port_id + '">' + str(i) + "</TD>"
                )
            else:
                label += '<TD BGCOLOR="white" COLSPAN="2"></TD>'
            if i == 0:
                label += (
                    '<TD BGCOLOR="white" COLSPAN="2" ROWSPAN="'
                    + str(rows)
                    + '">'
                    + stage_label(stage)
                    + "</TD>"
                )
            if i < len(stage["output_tensors"]):
                port_id = get_port_id(False, i)
                label += (
                    '<TD BGCOLOR="lightgrey" COLSPAN="2" PORT="' + port_id + '">' + str(i) + "</TD>"
                )
            else:
                label += '<TD BGCOLOR="white" COLSPAN="2"></TD>'
            label += "</TR>"
        label += "</TABLE>>"
        return label

    def dfg_dot(g, sch):
        """Create edges among stages."""
        stages = sch["stages"]
        for stage in stages:
            for i in range(len(stage["input_tensors"])):
                src = dom_path_to_string(stage["input_tensors"][i], "Tensor")
                dst = stage["id"] + ":" + get_port_id(True, i)
                g.edge(src, dst)
            for i in range(len(stage["output_tensors"])):
                src = stage["id"] + ":" + get_port_id(False, i)
                dst = stage["output_tensors"][i]["id"]
                g.edge(src, dst)

    graph = create_dataflow_graph("Dataflow Graph")
    s = extract_dom_for_viz(sch, need_range=False)
    for stage in s["stages"]:
        stage_node_dot(graph, stage)
        for tensor in stage["output_tensors"]:
            tensor_node_dot(graph, tensor)

    dfg_dot(graph, s)

    return dump_graph(graph.source, show_svg, dot_file_path, output_dot_string)
