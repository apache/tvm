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
"""Graph debug results dumping class."""
import collections
import json
import os
import numpy as np
import tvm

GRAPH_DUMP_FILE_NAME = '_tvmdbg_graph_dump.json'
CHROME_TRACE_FILE_NAME = "_tvmdbg_execution_trace.json"

ChromeTraceEvent = collections.namedtuple(
    'ChromeTraceEvent',
    ['ts', 'tid', 'pid', 'name', 'ph']
)


class DebugResult(object):
    """Graph debug data module.

    Data dump module manage all the debug data formatting.
    Output data and input graphs are formatted and dumped to file.
    Frontend read these data and graph for visualization.

    Parameters
    ----------
    graph_json : str
        The graph to be deployed in json format output by nnvm graph. Each operator (tvm_op)
        in the graph will have a one to one mapping with the symbol in libmod which is used
        to construct a "PackedFunc" .

    dump_path : str
        Output data path is read/provided from frontend
    """

    def __init__(self, graph_json, dump_path):
        self._dump_path = dump_path
        self._output_tensor_list = []
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
            node['attrs'].update({"T": dtype})
            node['shape'] = self._shapes_list[1][i]

    def _cleanup_tensors(self):
        """Remove the tensor dump file (graph wont be removed)
        """
        for filename in os.listdir(self._dump_path):
            if os.path.isfile(filename) and not filename.endswith(".json"):
                os.remove(filename)

    def get_graph_nodes(self):
        """Return the nodes list
        """
        return self._nodes_list

    def get_graph_node_shapes(self):
        """Return the nodes shapes list
        """
        return self._shapes_list

    def get_graph_node_output_num(self, node):
        """Return the number of outputs of a node
        """
        return 1 if node['op'] == 'param' else int(node['attrs']['num_outputs'])

    def get_graph_node_dtypes(self):
        """Return the nodes dtype list
        """
        return self._dtype_list

    def get_output_tensors(self):
        """Dump the outputs to a temporary folder, the tensors are in numpy format
        """
        eid = 0
        order = 0
        output_tensors = {}
        for node, time in zip(self._nodes_list, self._time_list):
            num_outputs = self.get_graph_node_output_num(node)
            for j in range(num_outputs):
                order += time[0]
                key = node['name'] + "_" + str(j)
                output_tensors[key] = self._output_tensor_list[eid]
                eid += 1
        return output_tensors

    def dump_output_tensor(self):
        """Dump the outputs to a temporary folder, the tensors are in numpy format
        """
        #cleanup existing tensors before dumping
        self._cleanup_tensors()
        eid = 0
        order = 0
        output_tensors = {}
        for node, time in zip(self._nodes_list, self._time_list):
            num_outputs = self.get_graph_node_output_num(node)
            for j in range(num_outputs):
                order += time[0]
                key = node['name'] + "_" + str(j) + "__" + str(order)
                output_tensors[key] = self._output_tensor_list[eid]
                eid += 1

        with open(os.path.join(self._dump_path, "output_tensors.params"), "wb") as param_f:
            param_f.write(save_tensors(output_tensors))

    def dump_chrome_trace(self):
        """Dump the trace to the Chrome trace.json format.
        """
        def s_to_us(t):
            return t * 10 ** 6

        starting_times = np.zeros(len(self._time_list) + 1)
        starting_times[1:] = np.cumsum([times[0] for times in self._time_list])

        def node_to_events(node, times, starting_time):
            return [
                ChromeTraceEvent(
                    ts=s_to_us(starting_time),
                    tid=1,
                    pid=1,
                    ph='B',
                    name=node['name'],
                ),
                ChromeTraceEvent(
                    # Use start + duration instead of end to ensure precise timings.
                    ts=s_to_us(times[0] + starting_time),
                    tid=1,
                    pid=1,
                    ph='E',
                    name=node['name'],
                ),
            ]
        events = [
            e for (node, times, starting_time) in zip(
                self._nodes_list, self._time_list, starting_times)
            for e in node_to_events(node, times, starting_time)]
        result = dict(
            displayTimeUnit='ns',
            traceEvents=[e._asdict() for e in events]
        )

        with open(os.path.join(self._dump_path, CHROME_TRACE_FILE_NAME), "w") as trace_f:
            json.dump(result, trace_f)

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
            json.dump(graph, outfile, indent=4, sort_keys=False)

    def display_debug_result(self, sort_by_time=True):
        """Displays the debugger result"
        """
        header = ["Node Name", "Ops", "Time(us)", "Time(%)", "Shape", "Inputs", "Outputs"]
        lines = ["---------", "---", "--------", "-------", "-----", "------", "-------"]
        eid = 0
        data = []
        total_time = sum(time[0] for time in self._time_list)
        for node, time in zip(self._nodes_list, self._time_list):
            num_outputs = self.get_graph_node_output_num(node)
            for j in range(num_outputs):
                op = node['op']
                if node['op'] == 'param':
                    eid += 1
                    continue
                name = node['name']
                shape = str(self._output_tensor_list[eid].shape)
                time_us = round(time[0] * 1000000, 3)
                time_percent = round(((time[0] / total_time) * 100), 3)
                inputs = str(node['attrs']['num_inputs'])
                outputs = str(node['attrs']['num_outputs'])
                node_data = [name, op, time_us, time_percent, shape, inputs, outputs]
                data.append(node_data)
                eid += 1

        if sort_by_time:
            # Sort on the basis of execution time. Prints the most expensive ops in the start.
            data = sorted(data, key=lambda x: x[2], reverse=True)
            # Insert a row for total time at the end.
            rounded_total_time = round(total_time * 1000000, 3)
            data.append(["Total_time", "-", rounded_total_time, "-", "-", "-", "-", "-"])

        fmt = ""
        for i, _ in enumerate(header):
            max_len = len(header[i])
            for j, _ in enumerate(data):
                item_len = len(str(data[j][i]))
                if item_len > max_len:
                    max_len = item_len
            fmt = fmt + "{:<" + str(max_len + 2) + "}"
        print(fmt.format(*header))
        print(fmt.format(*lines))
        for row in data:
            print(fmt.format(*row))

def save_tensors(params):
    """Save parameter dictionary to binary bytes.

    The result binary bytes can be loaded by the
    GraphModule with API "load_params".

    Parameters
    ----------
    params : dict of str to NDArray
        The parameter dictionary.

    Returns
    -------
    param_bytes: bytearray
        Serialized parameters.
    """
    _save_tensors = tvm.get_global_func("_save_param_dict")

    args = []
    for k, v in params.items():
        args.append(k)
        args.append(tvm.nd.array(v))
    return _save_tensors(*args)
