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
"""Graph transformation specific to accelerator.

This module provide specific NNVM graph transformations
to transform a generic NNVM graph to a version that can
be executed on accelerator.
"""

import nnvm

from nnvm.compiler import graph_attr, graph_util


def _pack_batch_channel(data, dshape, bfactor, cfactor):
    """Pack the data channel dimension.
    """
    assert dshape[0] % bfactor == 0
    assert dshape[1] % cfactor == 0
    data = nnvm.sym.reshape(data,
                            shape=(dshape[0] // bfactor, bfactor,
                                   dshape[1] // cfactor, cfactor,
                                   dshape[2], dshape[3]))
    data = nnvm.sym.transpose(
        data, axes=(0, 2, 4, 5, 1, 3))
    return data


def _unpack_batch_channel(data, old_shape):
    """Unpack the data channel dimension.
    """
    data = nnvm.sym.transpose(data, axes=(0, 4, 1, 5, 2, 3))
    data = nnvm.sym.reshape(data, shape=old_shape)
    return data


def _pack_weight(data, dshape, cfactor):
    """Pack the weight into packed format.
    """
    assert len(dshape) == 4
    assert dshape[0] % cfactor == 0
    assert dshape[1] % cfactor == 0
    data = nnvm.sym.reshape(data,
                            shape=(dshape[0] // cfactor, cfactor,
                                   dshape[1] // cfactor, cfactor,
                                   dshape[2], dshape[3]))
    data = nnvm.sym.transpose(
        data, axes=(0, 2, 4, 5, 1, 3))
    return data


def _pack_bias(data, dshape, bfactor, cfactor):
    """Pack the bias parameter.
    """
    assert len(dshape) == 3
    assert dshape[0] % cfactor == 0
    data = nnvm.sym.reshape(data,
                            shape=(dshape[0] // cfactor,
                                   cfactor, dshape[1],
                                   dshape[2], 1))
    data = nnvm.sym.transpose(
        data, axes=(0, 2, 3, 4, 1))
    # broadcast batch dimension to bfactor
    data = nnvm.sym.broadcast_to(
        data,
        shape=(dshape[0] // cfactor, dshape[1], dshape[2], bfactor, cfactor))
    return data


def _get_shape(sym, shape_dict):
    """Get the shape of a node.
    """
    return graph_util.infer_shape(
        nnvm.graph.create(sym), **shape_dict)[1][0]

def clean_conv_fuse(graph):
    """Cleanup the convolution's later fuse stages

    Parameters
    ----------
    graph : Graph
        Input graph

    Returns
    -------
    graph : Graph
        Optimized graph
    """
    def _clean_entry(entry):
        node, flag = entry
        if flag:
            node = nnvm.symbol.clip(node, a_max=127, a_min=-127)
            node = nnvm.symbol.cast(node, dtype="int8")
            # Use copy as a hint to block conv2d schedules
            node = nnvm.symbol.copy(node)
            flag = False
        return node, flag

    gidx = graph.index
    ref_count = {}
    # count reference of each node
    for nid, node in enumerate(gidx.nodes):
        ref_count[nid] = 0
        for elem in node["inputs"]:
            ref_count[elem[0]] += 1
    # construction remap
    # entry_id->(new_node, conv_fuse)
    # need_fold: bool indicates if we need fold
    node_map = {}

    for nid, node in enumerate(gidx.nodes):
        children = [node_map[e[0]] for e in node["inputs"]]
        attrs = node.get("attrs", {})
        node_name = node["name"]
        op_name = node["op"]
        get_clone = lambda c, o_n, n_n, a: getattr(nnvm.symbol, o_n)(
            *c, name=n_n, **a)

        new_entry = None
        if op_name == "null":
            new_entry = (nnvm.symbol.Variable(node_name), False)
        elif op_name in ("cast", "clip"):
            if children[0][1]:
                new_entry = children[0]
            else:
                new_entry = (
                    get_clone([children[0][0]], op_name, node_name, attrs),
                    False)
        elif op_name == "conv2d" and attrs["out_dtype"] == "int32":
            data, weight = children
            data = _clean_entry(data)
            new_node = nnvm.sym.conv2d(
                data[0], weight[0], name=node_name, **attrs)
            new_entry = (new_node, True)
        elif op_name in ("__lshift_scalar__", "__rshift_scalar__", "relu"):
            new_entry = (
                get_clone([children[0][0]], op_name, node_name, attrs),
                children[0][1])
        elif op_name in ("broadcast_add", "broadcast_mul"):
            rhs = children[1][0]
            lhs, _ = _clean_entry(children[0])
            lhs = nnvm.sym.cast(lhs, dtype="int32")
            rhs = nnvm.sym.cast(rhs, dtype="int32")
            new_entry = (
                get_clone([lhs, rhs], op_name, node_name, attrs),
                False)

        if new_entry is None:
            inputs = [_clean_entry(x) for x in children]
            new_entry = (
                get_clone([x[0] for x in inputs], op_name, node_name, attrs),
                False)
        if ref_count[nid] > 1:
            new_entry = _clean_entry(new_entry)
        node_map[nid] = new_entry

    assert len(graph.index.output_entries) == 1
    ret = node_map[graph.index.output_entries[0][0]][0]
    ret = nnvm.graph.create(ret)
    return ret

def clean_cast(graph):
    """
    Move the casts to early part of graph,
    remove uncessary clip operations when possible.
    """
    gidx = graph.index
    node_map = {}

    def _clean_cast(node, target_type):
        op_name = node.attr("op_name")
        if op_name == "cast":
            return _clean_cast(node.get_children(), target_type)
        if op_name == "relu":
            data, has_clip = _clean_cast(
                node.get_children(), target_type)
            data = nnvm.sym.relu(data)
            return data, has_clip
        return nnvm.sym.cast(node, dtype=target_type), False

    for nid, node in enumerate(gidx.nodes):
        children = [node_map[e[0]] for e in node["inputs"]]
        attrs = node.get("attrs", {})
        node_name = node["name"]
        op_name = node["op"]
        get_clone = lambda c, o_n, n_n, a: getattr(nnvm.symbol, o_n)(
            *c, name=n_n, **a)

        if op_name == "null":
            new_node = nnvm.symbol.Variable(node_name)
        elif op_name == "cast":
            dtype = attrs["dtype"]
            new_node, _ = _clean_cast(children[0], dtype)
        elif op_name == "conv2d" and attrs["out_dtype"] == "int32":
            data, weight = children
            data, _ = _clean_cast(data, "int8")
            weight, _ = _clean_cast(weight, "int8")
            new_node = nnvm.sym.conv2d(
                data, weight, name=node_name, **attrs)
        elif op_name == "elemwise_add":
            lhs, rhs = children
            rhs = nnvm.sym.cast(rhs, dtype="int8")
            new_node = nnvm.sym.elemwise_add(lhs, rhs)
        else:
            new_node = get_clone(children, op_name, node_name, attrs)
        node_map[nid] = new_node

    assert len(graph.index.output_entries) == 1
    ret = node_map[graph.index.output_entries[0][0]]
    ret = nnvm.graph.create(ret)
    return ret


def pack(graph, shape_dict, bfactor, cfactor, start_name=None):
    """Pack the graph into batch&channel packed format.

    Parameters
    ----------
    graph : Graph
       The input graph.

    shape_dict : dict of str to shapex
       The input shape.

    bfactor : int
       The packing factor in batch

    cfactor : int
       The packing factor in channel

    start_name: str, optional
       Start name start packing from certain known node.

    Returns
    -------
    graph : Graph
        The transformed graph.
    """
    graph = graph_attr.set_shape_inputs(graph, shape_dict)
    graph = graph.apply("InferShape")
    shape = graph.json_attr("shape")
    gidx = graph.index
    node_map = {}
    dset = set()
    counter = 0
    start_pack = False

    for nid, node in enumerate(gidx.nodes):
        children = [node_map[e[0]] for e in node["inputs"]]
        ishape = [shape[gidx.entry_id(e)] for e in node["inputs"]]
        oshape = shape[gidx.entry_id(nid, 0)]
        attrs = node.get("attrs", {})
        node_name = node["name"]
        op_name = node["op"]
        get_clone = lambda c, o_n, n_n, a: getattr(nnvm.symbol, o_n)(
            *c, name=n_n, **a)

        if op_name == "null":
            new_node = nnvm.symbol.Variable(node_name)
            if start_name and node_name == start_name:
                start_pack = True
                new_node = _pack_batch_channel(new_node, oshape, bfactor, cfactor)
        elif op_name == "max_pool2d":
            assert not start_pack
            start_pack = True
            new_node = get_clone(children, op_name, node_name, attrs)
            new_node = _pack_batch_channel(new_node, oshape, bfactor, cfactor)
        elif op_name == "global_avg_pool2d":
            if start_pack:
                start_pack = False
                children[0] = _unpack_batch_channel(children[0], ishape[0])
                new_node = getattr(nnvm.symbol, op_name)(
                    *children, name=node_name, **attrs)
            else:
                new_node = get_clone(children, op_name, node_name, attrs)
        elif op_name == "conv2d" and attrs["out_dtype"] == "int32":
            if start_pack:
                attrs["layout"] = "NCHW%dn%dc" % (bfactor, cfactor)
                attrs["kernel_layout"] = "OIHW%do%di" % (cfactor, cfactor)
                data, weight = children
                weight = _pack_weight(weight, ishape[1], cfactor)
                new_node = nnvm.sym.conv2d(
                    data, weight, name=node_name, **attrs)
            elif counter == 1:
                attrs["layout"] = "NCHW%dn%dc" % (bfactor, cfactor)
                attrs["kernel_layout"] = "OIHW%do%di" % (cfactor, cfactor)
                data, weight = children
                data = _pack_batch_channel(data, ishape[0], bfactor, cfactor)
                weight = _pack_weight(weight, ishape[1], cfactor)
                new_node = nnvm.sym.conv2d(
                    data, weight, name=node_name, **attrs)
                new_node = _unpack_batch_channel(new_node, oshape)
                counter = counter + 1
            else:
                new_node = get_clone(children, op_name, node_name, attrs)
        elif op_name.startswith("broadcast"):
            if start_pack:
                assert len(ishape[1]) == 3
                children[1] = _pack_bias(children[1], ishape[1], bfactor, cfactor)
                new_node = getattr(nnvm.symbol, op_name)(
                    *children, name=node_name, **attrs)
            else:
                new_node = get_clone(children, op_name, node_name, attrs)
        elif op_name.startswith("elementwise_add"):
            new_node = get_clone(children, op_name, node_name, attrs)
        else:
            new_node = get_clone(children, op_name, node_name, attrs)
            dset.add(op_name)
        node_map[nid] = new_node

    assert len(graph.index.output_entries) == 1
    ret = node_map[graph.index.output_entries[0][0]]
    if start_pack:
        oshape = shape[graph.index.output_entries[0][0]]
        ret = _unpack_batch_channel(ret, oshape)
    graph = nnvm.graph.create(ret)
    graph = graph_attr.set_shape_inputs(graph, shape_dict)
    graph = graph.apply("InferShape")
    return graph
