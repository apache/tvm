"""Classes and methods for processing debugger-decorated graphs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def parse_node_or_tensor_name(name):
    """Get the node name from a string that can be node or tensor name.

    Args:
      name: An input node name (e.g., "node_a") or tensor name (e.g.,
        "node_a:0"), as a str.

    Returns:
      1) The node name, as a str. If the input name is a tensor name, i.e.,
        consists of a colon, the final colon and the following output slot
        will be stripped.
      2) If the input name is a tensor name, the output slot, as an int. If
        the input name is not a tensor name, None.
    """

    if ":" in name and not name.endswith(":"):
        node_name = name[:name.rfind(":")]
        output_slot = int(name[name.rfind(":") + 1:])

        return node_name, output_slot
    return name, None


def get_node_name(element_name):
    """Get the node name from a string that can be node or tensor name.

    Args:
      element_name: An input node name (e.g., "node_a") or tensor name (e.g.,
        "node_a:0"), as a str.

    Returns:
      The node name, as a str. If the input name is a tensor name, i.e.,
        consists of a colon, the final colon and the following output slot
        will be stripped.
    """
    node_name, _ = parse_node_or_tensor_name(element_name)
    return node_name


def get_output_slot(element_name):
    """Get the output slot number from the name of a graph element.

    If element_name is a node name without output slot at the end, 0 will be
    assumed.

    Args:
      element_name: (`str`) name of the graph element in question.

    Returns:
      (`int`) output slot number.
    """
    _, output_slot = parse_node_or_tensor_name(element_name)
    return output_slot if output_slot is not None else 0


def is_copy_node(node_name):
    """Determine whether a node name is that of a debug Copy node.

    Such nodes are inserted by TVM core upon request in
    RunOptions.debug_options.debug_tensor_watch_opts.

    Args:
      node_name: Name of the node.

    Returns:
      A bool indicating whether the input argument is the name of a debug Copy
      node.
    """
    return node_name.startswith("__copy_")


def is_debug_node(node_name):
    """Determine whether a node name is that of a debug node.

    Such nodes are inserted by TVM core upon request in
    RunOptions.debug_options.debug_tensor_watch_opts.

    Args:
      node_name: Name of the node.

    Returns:
      A bool indicating whether the input argument is the name of a debug node.
    """
    return node_name.startswith("__dbg_")

class GraphTracingReachedDestination(Exception):
    """Graph Tracing Reached Destination Exception."""
    pass

def _infer_device_name(graph_def):
    """Infer device name from a partition GraphDef."""
    device_name = None
    for node in graph_def.node:
        if node.device:
            device_name = node.device
            break
    if device_name is None:
        print(
            "Failed to infer device name from partition GraphDef: none of the "
            "nodes of the GraphDef has a non-empty device name.")
    return device_name


class DebugGraph(object):
    """Represents a debugger-decorated graph."""

    def __init__(self, graph_dump_def, device_name=None):
        """Constructor of _DebugGraph.

        Args:
          graph_dump_def: The debugger-decorated `tvm.GraphDef`, with the
            debugger-inserted Copy* and Debug* nodes.
          device_name: (str) name of the device.

        Raises:
          ValueError: If duplicate node names are encountered.
        """
        self._graph_dump_def = graph_dump_def
        self._non_graph_dump_def = None

        self._node_attributes = {}
        self._node_inputs = {}
        self._node_reversed_ref_inputs = {}
        self._node_ctrl_inputs = {}
        self._node_recipients = {}
        self._node_ctrl_recipients = {}
        self._node_devices = {}
        self._node_op_types = {}
        self._copy_send_nodes = []

        self._device_name = device_name
        if not self._device_name:
            self._device_name = _infer_device_name(graph_dump_def)

        for node in graph_dump_def.node:
            self._process_graph_dump_node(node)

        self._prune_nctl_edges_of_dbg_ops()
        self._prune_ctl_edges_of_dbg_ops()
        self._prune_nodes_ins_resp_maps(self._get_copy_nodes())

        self._populate_recipient_maps()

    def _process_graph_dump_node(self, node):
        """Process a node from the debug GraphDef.

        Args:
          node: (NodeDef) A partition-graph node to be processed.

        Raises:
          ValueError: If duplicate node names are encountered.
        """
        if is_debug_node(node.name):
            # This is a debug node. Parse the node name and retrieve the
            # information about debug watches on tensors. But do not include
            # the node in the graph.
            return

        if node.name in self._node_inputs:
            raise ValueError("Duplicate node name on device %s: '%s'" %
                             (self._device_name, node.name))

        self._node_attributes[node.name] = node.attr

        self._node_inputs[node.name] = []
        self._node_ctrl_inputs[node.name] = []
        self._node_recipients[node.name] = []
        self._node_ctrl_recipients[node.name] = []

        if node.name not in self._node_devices:
            self._node_devices[node.name] = set()
        self._node_devices[node.name].add(
            node.device if node.device else self._device_name)
        self._node_op_types[node.name] = node.op

        for inp in node.input:
            if is_copy_node(inp) and (node.op == "_Send" or node.op == "_Retval"):
                self._copy_send_nodes.append(node.name)

            if inp.startswith("^"):
                cinp = inp[1:]
                self._node_ctrl_inputs[node.name].append(cinp)
            else:
                self._node_inputs[node.name].append(inp)

    def _get_copy_nodes(self):
        """Find all Copy nodes in the loaded graph."""
        copy_nodes = []
        for node in self._node_inputs:
            if is_copy_node(node):
                copy_nodes.append(node)
        return copy_nodes

    def _prune_nctl_edges_of_dbg_ops(self):
        """Prune (non-control) edges related to debug ops.

        Prune the Copy ops and associated _Send ops inserted by the debugger out
        from the non-control inputs and output recipients map. Replace the inputs
        and recipients with original ones.
        """
        for node in self._node_inputs:
            inputs = self._node_inputs[node]

            len_inputs = len(inputs)
            for i in range(len_inputs):
                inp = inputs[i]
                if is_copy_node(inp):
                    # Find the input to the Copy node, which should be the original
                    # input to the node.
                    orig_inp = self._node_inputs[inp][0]
                    inputs[i] = orig_inp

    def _prune_ctl_edges_of_dbg_ops(self):
        """Prune control edges related to the debug ops."""
        for node in self._node_ctrl_inputs:
            ctrl_inputs = self._node_ctrl_inputs[node]
            debug_op_inputs = []
            for ctrl_inp in ctrl_inputs:
                if is_debug_node(ctrl_inp):
                    debug_op_inputs.append(ctrl_inp)
            for debug_op_inp in debug_op_inputs:
                ctrl_inputs.remove(debug_op_inp)

    def _populate_recipient_maps(self):
        """Populate the map from node name to recipient(s) of its output(s).

        This method also populates the input map based on reversed ref edges.
        """
        for node in self._node_inputs:
            inputs = self._node_inputs[node]
            for inp in inputs:
                inp = get_node_name(inp)
                if inp not in self._node_recipients:
                    self._node_recipients[inp] = []
                self._node_recipients[inp].append(node)

        for node in self._node_ctrl_inputs:
            ctrl_inputs = self._node_ctrl_inputs[node]
            for ctrl_inp in ctrl_inputs:
                if ctrl_inp in self._copy_send_nodes:
                    continue

                if ctrl_inp not in self._node_ctrl_recipients:
                    self._node_ctrl_recipients[ctrl_inp] = []
                self._node_ctrl_recipients[ctrl_inp].append(node)

    def _prune_nodes_ins_resp_maps(self, nodes_to_prune):
        """Prune nodes out of input and recipient maps.

        Args:
          nodes_to_prune: (`list` of `str`) Names of the nodes to be pruned.
        """
        for node in nodes_to_prune:
            del self._node_inputs[node]
            del self._node_ctrl_inputs[node]
            del self._node_recipients[node]
            del self._node_ctrl_recipients[node]

    def _reconstruct_ndbg_graph_def(self):
        """Reconstruct non-debug GraphDef.

        Non-debug GraphDef means the original GraphDef without the Copy* and Debug
        nodes inserted by the debugger.
        """
        if self._non_graph_dump_def:
            return

    @property
    def device_name(self):
        """Name of the device that the tensor belongs to.

        Returns:
          (`str`) device name.
        """
        return self._device_name

    @property
    def graph_dump_def(self):
        """The debugger-decorated GraphDef."""
        return self._graph_dump_def

    @property
    def non_graph_dump_def(self):
        """The GraphDef without the Copy* and Debug* nodes added by the debugger."""
        self._reconstruct_ndbg_graph_def()
        return self._non_graph_dump_def

    @property
    def node_devices(self):
        """Device that belong to tensor.

        Returns:
          (`str`) node device name.
        """
        return self._node_devices

    @property
    def node_op_types(self):
        """Type of device that the tensor belongs to.

        Returns:
          (`str`) node tensor type.
        """
        return self._node_op_types

    @property
    def node_attributes(self):
        """Attributes of device that the tensor belongs to.

        Returns:
          (`str`) node tensor attributes.
        """
        return self._node_attributes

    @property
    def node_inputs(self):
        """Inputes of device that the tensor belongs to.

        Returns:
          (`Dict`) inputs of node tensor.
        """
        return self._node_inputs

    @property
    def node_ctrl_inputs(self):
        """Control inputes of device that the tensor belongs to.

        Returns:
          (`Dict`) control inputs of node tensor.
        """
        return self._node_ctrl_inputs

    @property
    def node_reversed_ref_inputs(self):
        """Reversed inputs of device that the tensor belongs to.

        Returns:
          (`Dict`) reversed inputs of node tensor.
        """
        return self._node_reversed_ref_inputs

    @property
    def node_recipients(self):
        """Recipient of device that the tensor belongs to.

        Returns:
          (`Dict`) recipient of node tensor.
        """
        return self._node_recipients

    @property
    def node_ctrl_recipients(self):
        """Control recipients of device that the tensor belongs to.

        Returns:
          (`Dict`) control recipients of node tensor.
        """
        return self._node_ctrl_recipients


def reconstruct_non_graph_dump_def(graph_dump_def):
    """Reconstruct original (non-debugger-decorated) partition GraphDef.

    This method strips the input `tvm.GraphDef` of the Copy* and Debug*-type nodes
    inserted by the debugger.

    The reconstructed partition graph is identical to the original (i.e.,
      non-debugger-decorated) partition graph except in the following respects:
        1) The exact names of the runtime-inserted internal nodes may differ.
           These include _Send, _Recv, _HostSend, _HostRecv, _Retval ops.
        2) As a consequence of 1, the nodes that receive input directly from such
           send- and recv-type ops will have different input names.
        3) The parallel_iteration attribute of while-loop Enter ops are set to 1.

    Args:
      graph_dump_def: The debugger-decorated `tvm.GraphDef`, with the
        debugger-inserted Copy* and Debug* nodes.

    Returns:
      The reconstructed `tvm.GraphDef` stripped of the debugger-inserted nodes.
    """
    return DebugGraph(graph_dump_def).non_graph_dump_def
