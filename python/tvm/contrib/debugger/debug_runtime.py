"""Graph debug runtime executes TVM debug packed functions."""

import os
import tempfile
import shutil
from datetime import datetime
from tvm._ffi.base import string_types
from tvm.contrib import graph_runtime
from tvm._ffi.function import get_global_func
from . import debug_result

_DUMP_ROOT_PREFIX = "tvmdbg_"
_DUMP_PATH_PREFIX = "_tvmdbg_"

def create(graph_json_str, libmod, ctx, dump_root=None):
    """Create a runtime executor module given a graph and module.

    Parameters
    ----------
    graph_json_str : str or graph class
        The graph to be deployed in json format output by nnvm graph.
        The graph can only contain one operator(tvm_op) that
        points to the name of PackedFunc in the libmod.

    libmod : tvm.Module
        The module of the corresponding function.

    ctx : TVMContext
        The context to deploy the module, can be local or remote.

    dump_root : str
        To select which folder the outputs should be kept.
        None will make a temp folder in /tmp/tvmdbg<rand_string> and does the dumping
    Returns
    -------
    graph_module : GraphModuleDebug
        Debug Runtime graph module that can be used to execute the graph.
    """
    if not isinstance(graph_json_str, string_types):
        try:
            graph_json_str = graph_json_str._tvm_graph_json()
        except AttributeError:
            raise ValueError("Type %s is not supported" % type(graph_json_str))
    try:
        fcreate = get_global_func("tvm.graph_runtime_debug.create")
    except ValueError:
        raise ValueError("Please set '(USE_GRAPH_RUNTIME_DEBUG ON)' in " \
                         "config.cmake and rebuild TVM to enable debug mode")

    ctx, num_rpc_ctx, device_type_id = graph_runtime.get_device_ctx(libmod, ctx)
    if num_rpc_ctx == len(ctx):
        raise NotSupportedError("Remote graph debugging is not supported.")

    func_obj = fcreate(graph_json_str, libmod, *device_type_id)
    return GraphModuleDebug(func_obj, ctx, graph_json_str, dump_root)


class GraphModuleDebug(graph_runtime.GraphModule):
    """Graph debug runtime module.

    This is a debug wrapper over the TVM runtime.
    Runtime interfaces are wrapped with debug functionalities.
    Manage the debug framework to format the debug data and
    trigger the user interfaces.

    Parameters
    ----------
    module : Module
        The interal tvm module that holds the actual graph functions.

    ctx : TVMContext
        The context this module is under.

    graph_json_str : str or graph class
        Content of graph json file in string format

    dump_root : str
        To select which folder the outputs should be kept.
        None will make a temp folder in /tmp/tvmdbg<rand_string> and does the dumping
    """
    def __init__(self, module, ctx, graph_json_str, dump_root):
        self._dump_root = dump_root
        self._dump_path = None
        self._debug_run = module["debug_run"]
        self._get_output_by_layer = module["get_output_by_layer"]
        graph_runtime.GraphModule.__init__(self, module)
        self._create_debug_env(graph_json_str, ctx)

    def _format_context(self, ctx):
        return str(ctx[0]).upper().replace("(", ":").replace(")", "")

    def _ensure_dir(self, directory):
        """Create a directory if not exists

        Parameters
        ----------

        directory : str
            File path to create
        """
        if not os.path.exists(directory):
            os.makedirs(directory, 0o700)

    def _get_dump_path(self, ctx):
        """Make the graph and tensor dump folder and return the path.

        Parameters
        ----------
        ctx : TVMContext
            The context this module is under.

        Returns
        -------
        path : str
            Directory path where the graph and node outputs will be stored.
        """
        # save to file
        folder_name = _DUMP_PATH_PREFIX + "ctx_"
        folder_name = folder_name + ctx.replace(":", "_")
        path = os.path.join(self._dump_root, folder_name)
        self._ensure_dir(path)
        return path

    def _remove_dump_root(self):
        if os.path.isdir(self._dump_root):
            shutil.rmtree(self._dump_root)

    def _create_debug_env(self, graph_json, ctx):
        """Create UI wrapper framework to handle multiple UI frontends for tvmdbg

        Parameters
        ----------
        graph_json : json format
            json formatted NNVM graph contain list of each node's name, shape and type.

        nodes_list : list
            List of all the nodes presented in the graph

        ctx : TVMContext
            The context this module is under.
        """
        # make the dump folder if not given
        if not self._dump_root:
            self._dump_root = tempfile.mktemp(prefix=_DUMP_ROOT_PREFIX)

        # format the context
        ctx = self._format_context(ctx)

        # updates the dumping directories
        self._dump_path = self._get_dump_path(ctx)

        # init the debug dumping environment
        self.debug_datum = debug_result.DebugResult(graph_json, self._dump_path)

    def _run_debug(self):
        """Execute the node spcified with index will be executed.
        Each debug output will be copied to the buffer
        Time consumed for each execuion will be set as debug output.

        """

        for i, node in enumerate(self.debug_datum.get_graph_nodes()):
            start_time = datetime.now().time()
            time_stamp = self._debug_run(i)
            end_time = datetime.now().time()
            self.debug_datum._time_list.append([time_stamp, start_time, end_time])
            num_outputs = self.debug_datum.get_graph_node_output_num(node)
            for j in range(num_outputs):
                out_tensor = self._get_output_by_layer(i, j)
                self.debug_datum._output_tensor_list.append(out_tensor)
    def run(self, **input_dict):
        """Run forward execution of the graph with debug

        Parameters
        ----------
        input_dict : dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)

        # Step 1. Execute the graph
        self._run_debug()
        # Step 2. Dump the output tensors to the dump folder
        self.debug_datum.dump_output_tensor()
        # Step 3. Display the collected information
        self.debug_datum.display_debug_result()

    def exit(self):
        """Exits the dump folder and all its contents"""
        self._remove_dump_root()
