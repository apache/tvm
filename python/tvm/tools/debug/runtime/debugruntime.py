"""Debug runtime functions."""

import os
import json
import numpy as np
from tvm import ndarray as nd
from tvm.tools.debug.wrappers import ui_wrapper as tvmdbg

class DebugGraphModule(object):
    """Wrapper debug runtime module.

    This is a thin wrapper of the debug for TVM runtime.

    Parameters
    ----------
    nodes_list : list
        The list of all the graph nodes.

    cli_obj : Object
        The context of CLI object

    """
    def __init__(self, nodes_list, cli_obj, dbg_out_buffer_list):
        self.nodes_list = nodes_list
        self.cli_obj = cli_obj
        self.dbg_out_buffer_list = dbg_out_buffer_list

    def get_run_command(self):
        return self.cli_obj.get_run_command()

    def run_end(self, run_cli_session, retvals):
        self.cli_obj.run_end(run_cli_session, retvals)

    def get_debug_buffer_count(self):
        return len(self.dbg_out_buffer_list)

    def get_debug_buffer(self, eid):
        return self.dbg_out_buffer_list[eid]

    def set_input(self, key=None, value=None, **params):
        """Set inputs to the module via kwargs

        Parameters
        ----------

        key : int or str
           The input key

        value : the input value.
           The input key

        params : dict of str to NDArray
           Additonal arguments
        """
        params = params
        if key:
            self.cli_obj.set_input(key.replace("/", "_"), value)

    def dump_output(self):
        """Dump the outputs to a temporary folder

        Parameters
        ----------

        cli_obj: obj
            The CLI object

        """
        eid = 0
        for node in self.nodes_list:
            num_outputs = 1 if node['op'] == 'param' else int(node['attrs']['num_outputs'])
            for j in range(num_outputs):
                ndbuffer = self.dbg_out_buffer_list[eid]
                eid += 1
                key = node['name'] + "_" + str(j) + "__000000" + str(ndbuffer.time_stamp) + ".npy"
                key = key.replace("/", "_")
                file_name = str(self.cli_obj._dump_root + self.cli_obj.dump_folder() + key)
                np.save(file_name, ndbuffer.asnumpy())
                os.rename(file_name, file_name.rpartition('.')[0])

    def _ensure_dir(self, file_path):
        """Create a directory if not exists

        Parameters
        ----------

        file_path: str
            File path to create

        """
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _dump_graph_json(self, ctx, new_graph):
        # save to file
        graph_dump_file_name = '_tvmdbg_graph_dump.json'
        folder_name = "/_tvmdbg_device_,job_localhost,replica_0,task_0,device_"
        folder_name = folder_name + ctx.replace(":", "_") + "/"
        self.cli_obj.dump_folder(folder_name)
        path = self.cli_obj._dump_root + folder_name
        self._ensure_dir(path)
        with open((path + graph_dump_file_name), 'w') as outfile:
            json.dump(new_graph, outfile, indent=2, sort_keys=False)

    def _dump_output_nodes(self, nodes_list, heads_list):
        """Dump the heads to a list

        Parameters
        ----------

        cli_obj: obj
            The CLI object

        heads_list : List
           The list of outputs from the json node

        """
        for output in heads_list:
            self.cli_obj.set_ouputs(nodes_list[output[0]]['name'])

def _make_debug_buffer_list(shapes_list, dltype_list):
    dbg_out_buffer_list = []
    for i in range(len(shapes_list[1])):
        dbg_out_buffer_list.append(nd.empty(shapes_list[1][i], dltype_list[1][i]))
    return dbg_out_buffer_list


def _get_graph_json(nodes_list, dltype_list, shapes_list):
    """Dump the nodes in json format to file

    Parameters
    ----------

    ctx: Str
        context in string

    cli_obj: obj
        CLI object where common information is stored

    nodes_list: List
        List of nodes in the graph

    dltype_list: List
        List of datatypes of each node

    shapes_list: List
        List of shape of each node

    """

    new_graph = {}
    new_graph['nodes'] = []
    nodes_len = len(nodes_list)
    for i in range(nodes_len):
        node = nodes_list[i]
        input_list = []
        for input_node in node['inputs']:
            input_list.append(nodes_list[input_node[0]]['name'])
        #del node['inputs']
        node['inputs'] = input_list
        dltype = str("type: " + dltype_list[1][i])
        if 'attrs' not in node:
            node['attrs'] = {}
            node['op'] = "param"
        else:
            node['op'] = node['attrs']['func_name']
        node['name'] = node['name'].replace("/", "_")
        node['attrs'].update({"T": dltype})
        node['shape'] = shapes_list[1][i]
        new_graph['nodes'].append(node)

    return new_graph


def create(obj, graph, ctx):
    """Create a debug runtime environment and start the CLI

    Parameters
    ----------
    obj: Object
        The object being used to store the graph runtime.

    graph: str
        NNVM graph in json format

    """
    json_obj = json.loads(graph)
    nodes_list = json_obj['nodes']
    dltype_list = json_obj['attrs']['dltype']
    shapes_list = json_obj['attrs']['shape']
    heads_list = json_obj['heads']

    new_graph = _get_graph_json(nodes_list, dltype_list, shapes_list)
    ctx = str(ctx).upper().replace("(", ":").replace(")", "")
    # make the cli object
    cli_obj = tvmdbg.LocalCLIDebugWrapperModule(obj, new_graph, ctx=ctx)
    # prepare the debug out buffer list
    dbg_buff_list = _make_debug_buffer_list(shapes_list, dltype_list)
    m = DebugGraphModule(nodes_list, cli_obj, dbg_buff_list)
    # dump the json information
    m._dump_graph_json(ctx, new_graph)
    m._dump_output_nodes(nodes_list, heads_list)
    return m
