"""Graph defenition class which is used to exchange the nodes information between tvm and CLI."""
from __future__ import absolute_import as _abs


class GraphDef(object):
    """The class which is used to exhange the nodes inforamtion between TVM and CLI.
       This class contains the list of nodes."""
    def __init__(self, ctx, json_nodes):
        self._node = []
        for node in json_nodes['nodes']:
            self._node.append(Node(ctx, node))

    @property
    def node(self):
        return self._node


class Node(object):
    """The class which is used to store a node inforamtion.
       This class contains the node information like name, ops, context,
       inputs and other attributes.
       Both the arguments and operation is represented in the same node"""
    def __init__(self, ctx, node):

        name = node['name']
        op = node['op']
        device = "/job:localhost/replica:0/task:0/device:" + ctx
        input_lst = []
        attr = {}
        if 'inputs' in node:
            input_lst = node['inputs']
        if 'attrs' in node:
            attr = node['attrs']

        self._name = name
        self._op = op
        self._device = device
        self._input = input_lst
        self._attr = attr

    @property
    def device(self):
        """Returns the device context"""
        return self._device

    @property
    def attr(self):
        """Returns the attributes of a node"""
        return self._attr

    @property
    def name(self):
        """Returns the name of a node"""
        return self._name

    @property
    def op(self):
        """Returns the optpe of a node"""
        return self._op

    @property
    def input(self):
        """Returns the inputs of an node which is not an argument"""
        return self._input
