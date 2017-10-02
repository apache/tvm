NNVM Graph JSON Specification
=============================

NNVM uses JSON for graph serialization. This allows NNVM graph to be
exported to any backend either natively supported or by third-party
without any dependency such as protobuf.

Getting started
---------------

A serialized NNVM graph in JSON format can be deserialized by any JSON
parser.

.. code:: python

    # python
    import json
    with open('model.json', 'r') as f:
      graph = json.loads(f.read())
    print(graph.keys())

``['nodes', 'arg_nodes', 'heads', 'node_row_ptr']``

Actually, the following keys are valid in JSON graph.

+--------------------------------------+------------+-----------------------------------+
| Keys                                 | Required   | Description                       |
+======================================+============+===================================+
| `nodes <#nodes>`__                   | Yes        | The nodes in graph.               |
+--------------------------------------+------------+-----------------------------------+
| `arg\_nodes <#arg_nodes>`__          | Yes        | Indices of input nodes.           |
+--------------------------------------+------------+-----------------------------------+
| `heads <#heads>`__                   | Yes        | Indices of output nodes.          |
+--------------------------------------+------------+-----------------------------------+
| `node\_row\_ptr <#node_row_ptr>`__   | Optional   | Depth first search row indices.   |
+--------------------------------------+------------+-----------------------------------+
| `attr <#attr>`__                     | Optional   | Additional information.           |
+--------------------------------------+------------+-----------------------------------+

nodes
-----

Explained by the name itself, ``nodes`` are either placeholders or
computational nodes in NNVM graph. The ``nodes`` are stored in list.

.. code:: python

    nodes = graph['nodes']
    print(len(nodes))
    print(nodes[0])
    print(nodes[3])

::

    53
    {'inputs': [], 'name': 'data', 'op': 'null'}
    {'inputs': [[0, 0, 0], [1, 0, 0], [2, 0, 0]], 'attrs': {'channels': '64',
    'padding': '(1, 1)', 'layout': 'NCHW', 'kernel_size': '[3, 3]', 'groups': '1',
    'strides': '(1, 1)', 'use_bias': 'True', 'dilation': '(1, 1)'},
    'name': 'conv1_1', 'op': 'conv2d'}

The following keys are valid in each node:

+----------------+------------------+----------+
| Keys           | Required         | Descript |
|                |                  | ion      |
+================+==================+==========+
| op             | Yes              | The      |
|                |                  | operator |
|                |                  | type     |
|                |                  | name,    |
|                |                  | 'null'   |
|                |                  | is used  |
|                |                  | if it's  |
|                |                  | a        |
|                |                  | placehol |
|                |                  | der/vari |
|                |                  | able/inp |
|                |                  | ut.      |
+----------------+------------------+----------+
| name           | Yes              | The      |
|                |                  | given    |
|                |                  | name of  |
|                |                  | the      |
|                |                  | node,    |
|                |                  | defined  |
|                |                  | by user  |
|                |                  | composin |
|                |                  | g        |
|                |                  | the      |
|                |                  | network. |
+----------------+------------------+----------+
| inputs         | Yes              | List of  |
|                |                  | Entry    |
|                |                  | of the   |
|                |                  | input    |
|                |                  | nodes,   |
|                |                  | can be   |
|                |                  | empty    |
|                |                  | list []. |
|                |                  | Entry is |
|                |                  | a list   |
|                |                  | of       |
|                |                  | [nose\_i |
|                |                  | d,       |
|                |                  | index,   |
|                |                  | version] |
+----------------+------------------+----------+
| attrs          | Optional         | Extra    |
|                |                  | attribut |
|                |                  | es       |
|                |                  | for the  |
|                |                  | specific |
|                |                  | operator |
|                |                  | .        |
+----------------+------------------+----------+
| control\_deps  | Optional         | Control  |
|                |                  | dependen |
|                |                  | cies,    |
|                |                  | left     |
|                |                  | blank    |
|                |                  | unless   |
|                |                  | specific |
|                |                  | ally     |
|                |                  | used.    |
+----------------+------------------+----------+

``attrs`` for operators is a dictionary. Key-value pair examples:

+----------------+------------------+----------+----------+
| Keys           | Value            | Operator | Descript |
|                |                  |          | ion      |
+================+==================+==========+==========+
| 'channels'     | '64'             | conv2d   | Output   |
|                |                  |          | channels |
|                |                  |          | for 2d   |
|                |                  |          | convolut |
|                |                  |          | ion.     |
+----------------+------------------+----------+----------+
| 'kernel\_size' | '[3, 3]'         | conv2d   | Convolut |
|                |                  |          | ion      |
|                |                  |          | filter   |
|                |                  |          | kernel   |
|                |                  |          | size in  |
|                |                  |          | (h, w),  |
|                |                  |          | list and |
|                |                  |          | tuple    |
|                |                  |          | both     |
|                |                  |          | works.   |
+----------------+------------------+----------+----------+
| 'use\_bias'    | '1'              | conv2d   | Whether  |
|                |                  |          | use bias |
|                |                  |          | such     |
|                |                  |          | that     |
|                |                  |          | `y = w   |
|                |                  |          | * x + b` |
|                |                  |          | .        |
+----------------+------------------+----------+----------+

.. note::

    Tips for parsing key-value pair:

    * Both key and value are stored as strings.

    * Boolean values need extra attention, convert to int is recommended since `bool('0') == True` in python.

    * For a full list of operator attributes, please refer to the core operator `documentation <top.html>`__.

arg\_nodes
----------

``arg_nodes`` is a list of indices of nodes which is
placeholder/variable/input to the graph.

.. code:: python

    print(graph['arg_nodes'])

::

    [0, 1, 2, 6, 7, 11, 12, 15, 16, 20, 21, 24, 25, 29, 30, 33, 34, 39, 40, 44, 45, 49, 50]

For example, ``nodes[3]`` is not in ``arg_nodes`` because it's an
internal node.

heads
-----

``heads`` is a list of entries as the outlet/output of the graph.

.. code:: python

    print(graph['heads'])

::

    [[52, 0, 0]]

This example indicating that there's only one output in the graph, with
index 52.

node\_row\_ptr
--------------

``node_row_ptr`` stores the history of forward path, so you can skip
constructing the entire graph in inference tasks.

attrs
-----

``attrs`` can contain version numbers or similar helpful informations.
