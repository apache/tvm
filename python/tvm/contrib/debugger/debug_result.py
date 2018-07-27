"""Graph debug results dumping class."""
import os
import json
import nnvm

GRAPH_DUMP_FILE_NAME = '_tvmdbg_graph_dump.json'

class DebugResult():
    """Graph debug data module.

    Data dump module manage all the debug data formatting.
    Output data and input graphs are formatted and the dump to files.
    Frontend read these data and graph for visualization.

    Parameters
    ----------
    graph_json : str
        The graph to be deployed in json format output by nnvm graph. The graph can only contain
        one operator(tvm_op) that points to the name of PackedFunc in the libmod.

    dump_path : str
        Output data path is read/provided from frontend
    """

    def __init__(self, graph_json, dump_path):
        self._dump_path = dump_path
        self._time_list = []
        self._parse_graph(graph_json)
        # dump the json information
        self.dump_graph_json(graph_json)

    def _parse_graph(self, graph_json):
        """Parse and extract the NNVM graph and update the nodes, shapes and dltype.

        Parameters
        ----------
        graph_json : str or graph class
           The graph to be deployed in json format output by nnvm graph.
        """
        json_obj = json.loads(graph_json)
        self._nodes_list = json_obj['nodes']
        self._shapes_list = json_obj['attrs']['shape']
        self._dtype_list = json_obj['attrs']['dltype']
        self._update_graph_json()

    def _update_graph_json(self):
        """update the nodes_list with name, shape and data type,
        for temporarily storing the output.
        """

        nodes_len = len(self._nodes_list)
        for i in range(nodes_len):
            node = self._nodes_list[i]
            input_list = []
            for input_node in node['inputs']:
                input_list.append(self._nodes_list[input_node[0]]['name'])
            node['inputs'] = input_list
            dtype = str("type: " + self._dtype_list[1][i])
            if 'attrs' not in node:
                node['attrs'] = {}
                node['op'] = "param"
            else:
                node['op'] = node['attrs']['func_name']
            node['name'] = node['name'].replace("/", "_")
            node['attrs'].update({"T": dtype})
            node['shape'] = self._shapes_list[1][i]

    def get_graph_nodes(self):
        """Return the nodes list
        """
        return self._nodes_list

    def get_graph_node_shapes(self):
        """Return the nodes shapes list
        """
        return self._shapes_list

    def get_graph_node_dtypes(self):
        """Return the nodes dtype list
        """
        return self._dtype_list

    def set_time(self, time):
        """set the timestamp to the timelist, the list is appended in the same order as node list

        Parameters
        ----------
        time : float
            The time for a particular operation, added to the list
        """
        self._time_list.append(time)

    def dump_output_tensor(self, out_stats):
        """Dump the outputs to a temporary folder, the tensors is in numpy format

        Parameters
        ----------
        out_stats: list
            Contains the list of output tensors
        """
        #cleanup existing tensors before dumping
        self._cleanup_tensors()
        eid = 0
        order = 0
        output_tensors = {}
        for node, time in zip(self._nodes_list, self._time_list):
            num_outputs = 1 if node['op'] == 'param' \
                            else int(node['attrs']['num_outputs'])
            for j in range(num_outputs):
                order += time
                key = node['name'].replace("/", "_") + "_" + str(j) + "__" + str(order)
                output_tensors[key] = out_stats[eid]
                eid += 1

        with open(os.path.join(self._dump_path, "output_tensors.params"), "wb") as param_f:
            param_f.write(nnvm.compiler.save_param_dict(output_tensors))

    def dump_graph_json(self, graph):
        """Dump json formatted graph.

        Parameters
        ----------
        graph : json format
            json formatted NNVM graph contain list of each node's
            name, shape and type.
        """
        graph_dump_file_name = GRAPH_DUMP_FILE_NAME
        with open(os.path.join(self._dump_path, graph_dump_file_name), 'w') as outfile:
            json.dump(graph, outfile, indent=2, sort_keys=False)

    def _cleanup_tensors(self):
        """Remove the tensor dumps files(grah wont be removed)
        """
        for filename in os.listdir(self._dump_path):
            if os.path.isfile(filename) and not filename.endswith(".json"):
                os.remove(filename)
