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
"""Tensor Expression Debug Display (TEDD), visualizing Tensor Expression"""
import tvm
from graphviz import Digraph
from graphviz import Source
from IPython.display import display
from IPython.display import SVG
import html

TVMDD_TABLE_BODY_WIDTH = 30
# Must match enum IterVarType defined in include/tvm/expr.h
ITERVAR_TYPE_STRING_MAP = {
    0: ('kDataPar', '#FFFFFF'),
    1: ('kThreadIndex', '#2980B9'),
    2: ('kCommReduce', '#FAD7A0'),
    3: ('kOrdered', '#D35400'),
    4: ('kOpaque', '#ABB2B9'),
    5: ('kUnrolled', '#D2B4DE'),
    6: ('kVectorized', '#AED6F1'),
    7: ('kParallelized', '#F5B7B1'),
    8: ('kTensorized', '#A9DFBF'),
}


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
                    Asserts or not if object doesn't have a registered ID.
    """
    prefix = prefix.replace('.', '_')
    if not hasattr(get_or_create_dot_id, "obj_id_dict"):
        get_or_create_dot_id.obj_id_dict = {}
    if obj not in get_or_create_dot_id.obj_id_dict:
        if (assert_on_missing):
            assert False, 'dot_id ' + str(obj) + ' has not been registered.'
        else:
            get_or_create_dot_id.obj_id_dict[obj] = prefix + hex(id(obj))
    return get_or_create_dot_id.obj_id_dict[obj]


def get_port_id(stage, is_input, index):
    return 'I_' + str(index) if is_input else 'O_' + str(index)


def get_itervar_type_info(type):
    assert type < len(
        ITERVAR_TYPE_STRING_MAP), 'Unknown IterVar type: ' + str(type)
    return ITERVAR_TYPE_STRING_MAP[type]


def get_itervar_label_color(itervar, iv_type):
    type_info = get_itervar_type_info(iv_type)
    return str(itervar.var) + '(' + type_info[0] + ')', type_info[1]


def linebrk(s, n):
    ss = ''
    j = 0
    for i in range(len(s)):
        if j == n and i != len(s) - 1:
            ss = ss + '\n'
            j = 0
        j = j + 1
        ss = ss + s[i]
    ss = html.escape(str(ss), quote=True)
    ss = ss.replace('\n', '<br/>')
    return ss


def create_graph(name="", rankdir='BT'):
    graph = Digraph(name=name)
    graph.graph_attr['rankdir'] = rankdir
    return graph


def itervar_label(itervar, index, index_color, label):
    return '<TR><TD PORT="' + get_or_create_dot_id(itervar, str(
        itervar.var)) + '" BGCOLOR="' + index_color + '">' + str(
            index
        ) + '</TD><TD BGCOLOR="white" PORT="itervar">' + label + '</TD></TR>'


def stage_label(stage):
    return stage.op.name + '<br/>Scope: ' + stage.scope


def legend_label():
    label = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'
    for iter_type in ITERVAR_TYPE_STRING_MAP:
        name, color = ITERVAR_TYPE_STRING_MAP[iter_type]
        label += '<TR><TD BGCOLOR="' + color + '"></TD>' \
                + '<TD BGCOLOR="white">' + name + '</TD></TR>'
    label += '</TABLE>>'
    return label


def legend_dot(g):
    with g.subgraph(name='cluster_legend') as subgraph:
        subgraph.attr(label='Legend')
        label = legend_label()
        subgraph.node('legend', label, shape='none', margin='0')


def dump_graph(DotString, showdot=True, dotfilepath=''):
    if dotfilepath:
        try:
            DotFile = open(dotfilepath, "w+")
            DotFile.write(DotString)
            DotFile.close()
        except Exception:
            print('Cannot open file: ' + dotfilepath)
    if showdot:
        src = Source(DotString)
        display(SVG(src.pipe(format='svg')))


def viz_schedule_tree(sch, showdot=False, dotfilepath=''):
    """Top level API to render schedule tree

        Parameters
        ----------
        sch : schedule
                    The schedule object to visualize

        showdot : bool
                    Display graph as SVG, useful for Jupyter notebooks.

        dotfilepath : string
                    Dotty file to save the graph.
    """
    def create_schedule_tree_graph(name=""):
        return create_graph(name=name, rankdir='BT')

    def root_dot(g):
        g.node('ROOT', 'ROOT', shape='oval', margin='0')

    def stage_node_dot(g, stage):
        node_label = stage_node_label(stage)
        g.node(get_or_create_dot_id(stage.op, stage.op.name),
               node_label,
               shape='none',
               margin='0')

    def stage_node_label(stage):
        label = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" ' \
            'CELLPADDING="4"> <TR><TD BGCOLOR="lightgrey" ' \
            'COLSPAN="2" PORT="stage">' + stage_label(stage) + '</TD></TR>'

        for index in range(len(stage.leaf_iter_vars)):
            leafiv = stage.leaf_iter_vars[index]
            iv_type = leafiv.iter_type
            var_attr_label = ''
            if leafiv in stage.iter_var_attrs:
                # binding
                bind_thread = stage.iter_var_attrs[leafiv].bind_thread
                if bind_thread != None:
                    var_attr_label = var_attr_label + " (" + str(
                        bind_thread.var) + ")"
                # tensorization
                tensor_intrin = stage.iter_var_attrs[leafiv].tensor_intrin
                if tensor_intrin != None:
                    var_attr_label = var_attr_label + \
                        " (tensor_intrin:" + \
                        str(stage.iter_var_attrs[leafiv].tensor_intrin.name) + ")"
                iv_type = stage.iter_var_attrs[leafiv].iter_type
            var_label, color = get_itervar_label_color(leafiv, iv_type)
            label += itervar_label(leafiv, index, color,
                                   var_label + var_attr_label)
        if hasattr(stage.op, 'body'):
            label += '<TR><TD COLSPAN="2">' + linebrk(str(
                stage.op.body), TVMDD_TABLE_BODY_WIDTH) + '</TD></TR>'
        label += '</TABLE>>'
        return label

    def compute_at_dot(g, stage):
        if stage.attach_type == 4:
            src = get_or_create_dot_id(stage.op, stage.op.name,
                                       True) + ':stage'
            dst = get_or_create_dot_id(
                stage.attach_stage.op,
                stage.attach_stage.op.name, True) + ':' + get_or_create_dot_id(
                    stage.attach_ivar, str(stage.attach_ivar.var), True)
        else:
            src = get_or_create_dot_id(stage.op, stage.op.name,
                                       True) + ':stage'
            dst = 'ROOT'
        g.edge(src, dst)

    graph = create_schedule_tree_graph("Schedule Tree")
    legend_dot(graph)
    for stage in sch.stages:
        stage_node_dot(graph, stage)
    for stage in sch.stages:
        compute_at_dot(graph, stage)
    root_dot(graph)
    dump_graph(graph.source, showdot, dotfilepath)
    return graph.source


def viz_itervar_relationship_graph(sch, showdot=False, dotfilepath=''):
    """Top level API to render IterVar relationship graph

        Parameters
        ----------
        sch : schedule
                    The schedule object to visualize
                    
        showdot : bool
                    Display graph as SVG, useful for Jupyter notebooks.

        dotfilepath : string
                    Dotty file to save the graph.
    """
    def create_itervar_relation_graph(name=""):
        return create_graph(name=name, rankdir='TB')

    def itervar_node_dot(g, itervar, iv_type, index):
        label = itervar_node_label(itervar, iv_type, index)
        var_name = str(itervar.var)
        g.node(get_or_create_dot_id(itervar, var_name),
               label,
               shape='none',
               margin='0')

    def itervar_node_label(itervar, iv_type, index):
        label = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" ' \
            'CELLPADDING="4">' + itervar_label(
            itervar, index,
            get_itervar_label_color(itervar, iv_type)[1],
            get_itervar_label_color(itervar, iv_type)[0]) + '</TABLE>>'
        return label

    def itervar_relation_node_dot(g, node_type, node_id, node_label, inputs,
                                  outputs):
        label = itervar_relation_node_label(node_type, node_id, node_label,
                                            inputs, outputs)
        g.node(node_id, label, shape='none', margin='0')

    def itervar_relation_node_label(node_type, node_id, node_label, inputs,
                                    outputs):
        label = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" ' \
            'CELLPADDING="4">' + '<TR>'
        max_port_num = max(len(inputs), len(outputs))
        for i in range(max_port_num):
            if (i < len(inputs)):
                input = inputs[i]
                label += '<TD BGCOLOR="lightgrey" PORT="' + input + '">' \
                        + input + '</TD>'
            else:
                label += '<TD BGCOLOR="white"></TD>'
        label += '</TR>'
        label += '<TR><TD BGCOLOR="white" COLSPAN="' + str(
            max_port_num) + '" PORT="relation">' + node_label + '</TD></TR>'
        label += '<TR>'
        for i in range(max_port_num):
            if (i < len(outputs)):
                output = outputs[i]
                label += '<TD BGCOLOR="lightgrey" PORT="' + output + '">' \
                    + output + '</TD>'
            else:
                label += '<TD BGCOLOR="white"></TD>'
        label += '</TR>'
        label += '</TABLE>>'
        return label

    def itervar_relation_dot(g, node, node_id):
        node_type = type(node)
        if node_type is tvm.schedule.Split:
            node_type = 'Split'
            itervar_relation_node_dot(g, node_type, node_id, node_type,
                                      ['Input'], ['Outer', 'Inner'])
            parent = get_or_create_dot_id(node.parent, str(node.parent.var),
                                          True)
            outer = get_or_create_dot_id(node.outer, str(node.outer.var), True)
            inner = get_or_create_dot_id(node.inner, str(node.inner.var), True)
            g.edge(parent + ':itervar', node_id + ':Input')
            g.edge(node_id + ':Outer', outer + ':itervar')
            g.edge(node_id + ':Inner', inner + ':itervar')
        elif node_type is tvm.schedule.Fuse:
            node_type = 'Fuse'
            itervar_relation_node_dot(g, node_type, node_id, node_type,
                                      ['Outer', 'Inner'], ['Fused'])
            fused = get_or_create_dot_id(node.fused, str(node.fused.var), True)
            outer = get_or_create_dot_id(node.outer, str(node.outer.var), True)
            inner = get_or_create_dot_id(node.inner, str(node.inner.var), True)
            g.edge(outer + ':itervar', node_id + ':Outer')
            g.edge(inner + ':itervar', node_id + ':Inner')
            g.edge(node_id + ':Fused', fused + ':itervar')
        elif node_type is tvm.schedule.Singleton:
            node_type = 'Singleton'
            itervar_relation_node_dot(g, node_type, node_id, node_type, [],
                                      ['Iter'])
            iter = get_or_create_dot_id(node.iter, str(node.iter.var), True)
            g.edge(node_id + ':Iter', iter + ':itervar')
        else:
            assert False, 'Unknown IterVarRelationNode: ' + node_type

    def get_leaf_itervars_index(stage, itervar):
        for i in range(len(stage.leaf_iter_vars)):
            if itervar == stage.leaf_iter_vars[i]:
                return i
        return -1

    def stage_node_dot(g, stage):
        with g.subgraph(
                name='cluster_' +
                get_or_create_dot_id(stage.op, stage.op.name)) as subgraph:
            subgraph.attr(label=stage.op.name)
            if (len(stage.all_iter_vars) > 0):
                for i in range(len(stage.all_iter_vars)):
                    iv = stage.all_iter_vars[i]
                    if iv in stage.iter_var_attrs:
                        iv_type = stage.iter_var_attrs[iv].iter_type
                    else:
                        iv_type = iv.iter_type
                    itervar_node_dot(subgraph, iv, iv_type,
                                     get_leaf_itervars_index(stage, iv))
                for i in range(len(stage.relations)):
                    node_id = get_or_create_dot_id(
                        stage.relations[i], stage.op.name + "_rel_" + str(i))
                    itervar_relation_dot(subgraph, stage.relations[i], node_id)
            else:
                subgraph.node(stage.op.name + '_placeholder', style='invis')

    graph = create_itervar_relation_graph("IterVar Relationship Graph")
    legend_dot(graph)
    for stage in sch.stages:
        stage_node_dot(graph, stage)

    dump_graph(graph.source, showdot, dotfilepath)
    return graph.source


def viz_dataflow_graph(sch, showdot=False, dotfilepath=''):
    """Top level API to render dataflow graph

        Parameters
        ----------
        sch : schedule
                    The schedule object to visualize
                    
        showdot : bool
                    Display graph as SVG, useful for Jupyter notebooks.

        dotfilepath : string
                    Dotty file to save the graph.
    """
    def create_dataflow_graph(name=""):
        return create_graph(name=name, rankdir='LR')

    def stage_node_dot(g, stage):
        op = stage.op
        label = stage_node_label(stage)
        g.node(get_or_create_dot_id(op, op.name),
               label,
               shape='none',
               margin='0')

    def stage_node_label(stage):
        op = stage.op
        rows = max(1, max(op.num_outputs, len(op.input_tensors)))
        label = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" ' \
            'CELLPADDING="4">'
        for i in range(rows):
            label += '<TR>'
            if (i < len(op.input_tensors)):
                port_id = get_port_id(op, True, i)
                label += '<TD BGCOLOR="lightgrey" COLSPAN="2" PORT="' \
                    + port_id + '">' + str(
                    i) + ':' + str(op.input_tensors[i].shape) + '::' + str(
                        op.input_tensors[i].dtype) + '</TD>'
            else:
                label += '<TD BGCOLOR="white" COLSPAN="2"></TD>'
            if (i == 0):
                label += '<TD BGCOLOR="white" COLSPAN="2" ROWSPAN="' + str(
                    rows) + '">' + stage_label(stage) + '</TD>'
            if (i < op.num_outputs):
                port_id = get_port_id(op, False, i)
                label += '<TD BGCOLOR="lightgrey" COLSPAN="2" PORT="' \
                    + port_id + '">' + str(
                    i) + ':' + str(op.output(0).shape) + '::' + str(
                        op.output(0).dtype) + '</TD>'
            else:
                label += '<TD BGCOLOR="white" COLSPAN="2"></TD>'
            label += '</TR>'
        label += '</TABLE>>'
        return label

    def dfg_dot(g, sch):
        stages = sch.stages
        for stage in stages:
            dest_op = stage.op
            for i in range(len(stage.op.input_tensors)):
                source_tensor = stage.op.input_tensors[i]
                source_op = source_tensor.op
                src = get_or_create_dot_id(source_op, source_op.name,
                                           True) + ':' + get_port_id(
                                               source_op, False, 0)
                dst = get_or_create_dot_id(dest_op, dest_op.name,
                                           True) + ':' + get_port_id(
                                               dest_op, True, i)
                g.edge(src, dst)

    graph = create_dataflow_graph("Dataflow Graph")
    for stage in sch.stages:
        stage_node_dot(graph, stage)

    dfg_dot(graph, sch)

    dump_graph(graph.source, showdot, dotfilepath)
    return graph.source
