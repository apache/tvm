"""Graph debug runtime executes TVM debug packed functions."""

import os
import tempfile
import shutil
from tvm._ffi.base import string_types
from tvm.contrib import graph_runtime
from tvm._ffi.function import get_global_func
from . import common
from . import debug_result

_DUMP_ROOT_PREFIX = "tvmdbg_"
_DUMP_PATH_PREFIX = "_tvmdbg_"

def create(graph_json_str, libmod, ctx, dbg_ux=None, dump_root=None):
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

    dbg_ux : str
        To select which ux user needs, Example, curses/tensorboard/None.
        None will just do the dumping

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
    device_type = ctx.device_type
    device_id = ctx.device_id
    try:
        fcreate = get_global_func("tvm.graph_runtime_debug.create")
    except ValueError:
        raise ValueError("Please set '(USE_GRAPH_RUNTIME_DEBUG ON)' in " \
                         "config.cmake and rebuild TVM to enable debug mode")
    func_obj = fcreate(graph_json_str, libmod, device_type, device_id)
    return GraphModuleDebug(func_obj, ctx, graph_json_str, dbg_ux, dump_root)


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

    dbg_ux : str
        To select which ui user needs, curses, tensorboard

    dump_root : str
        To select which folder the outputs should be kept.
        None will make a temp folder in /tmp/tvmdbg<rand_string> and does the dumping
    """
    def __init__(self, module, ctx, graph_json_str, dbg_ux, dump_root):
        self._dump_root = dump_root
        self._dump_path = None
        self._debug_run = module["debug_run"]
        self._get_ndarray = module["get_ndarray"]
        graph_runtime.GraphModule.__init__(self, module, ctx)
        self._prepare_data_and_ui(graph_json_str, ctx, dbg_ux)

    def _format_context(self, ctx):
        return str(ctx).upper().replace("(", ":").replace(")", "")

    def _prepare_data_and_ui(self, graph_json, ctx, dbg_ux):
        """Create the framework for debug data dumping and initialize the frontend

        Parameters
        ----------
        graph_json : str or graph class
            Content of graph json file in string format

        ctx : TVMContext
            The context this module is under.

        dbg_ux: str
            'curses'- involve curses based CLI frontend
            'tensorboard'- make data format for tensorbard frontend.
        """

        # create the ux and dump folder
        self._create_debug_ui(graph_json, ctx, dbg_ux)
        # init the debug dumping environment
        self.debug_datum = debug_result.DebugResult(graph_json, self._dump_path)

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
        folder_name = folder_name + ctx.replace(":", "_")# + "/"
        path = os.path.join(self._dump_root, folder_name)
        self._ensure_dir(path)
        return path

    def _remove_dump_root(self):
        if os.path.isdir(self._dump_root):
            shutil.rmtree(self._dump_root)

    def _get_run_command(self):
        """Invoke run from ux"""
        return common.UxAction.DEBUG_RUN

    def _run_end(self, action, retvals):
        """Notify run end to ux

        Parameters
        ----------
        action : common.UxAction
           The previous action

        retvals: int
           The return value of previous execution result, for ux
        """
        action = action
        retvals = retvals
        return common.UxAction.EXIT

    def _create_debug_ui(self, graph_json, ctx, dbg_ux): #pylint: disable=unused-argument
        """Create UI wrapper framework to handle multiple UI frontends for tvmdbg

        Parameters
        ----------
        graph_json : json format
            json formatted NNVM graph contain list of each node's name, shape and type.

        nodes_list : list
            List of all the nodes presented in the graph

        ctx : TVMContext
            The context this module is under.

        dbg_ux : str
            'curses'- involve curses based CLI
            'tensorboard'- make data format for tensorbard.
        """
        #make the dump folder if not given
        if not self._dump_root:
            self._dump_root = tempfile.mktemp(prefix=_DUMP_ROOT_PREFIX)

        #format the context
        ctx = self._format_context(ctx)

        #updates the dumping directories
        self._dump_path = self._get_dump_path(ctx)

    def _debug_run_op_exec(self, index=None):
        """Execute the node spcified with index will be executed.

        Time consumed for each execuion will be set as debug output.

        Parameters
        ----------
        index : int
            Node index to be executed now. Only the op corresponding to this index will be executed
        This will be mainly used for stepping each node and finding the output
        """
        if index:
            time_stamp = self._debug_run(index)
            self.debug_datum.set_time(time_stamp)
            return

        for i, node in enumerate(self.debug_datum.get_graph_nodes()):
            time_stamp = self._debug_run(i)
            self.debug_datum.set_time(time_stamp)
            num_outputs = self.debug_datum.get_graph_node_output_num(node)
            for j in range(num_outputs):
                out_tensor = self._get_ndarray(i, j)
                self.debug_datum.set_output_tensor(out_tensor)

    def _run_debug(self):
        """Invoke cli and when user execute select any operation,
        'get_run_start_resp' return the user input.

        Based on the user input, different type of Run will perform.
        'set_debug_buffer' will set the empty buffer for setting node outputs.
        Once the execution compled, output will be in the dump path and CLI will
        be notified as run ends.
        """

        #The ux may continue to execute multiple times, so after execution, give the control back
        #to ux and it will decide when to stop
        while True:
            action = self._get_run_command()
            if action == common.UxAction.DEBUG_RUN:
                # Step 1. Execute the graph
                retvals = self._debug_run_op_exec()
                # Step 2. Dump the output tensors to the dump folder
                self.debug_datum.dump_output_tensor()
                # Step 3. Display the collected information
                self.debug_datum.display_debug_result()
                # Step 4. Inform ux execution completion.
                action = self._run_end(action, retvals)
            elif action == common.UxAction.NON_DEBUG_RUN:
                retvals = super(GraphModuleDebug, self).run()
                action = self._run_end(action, retvals)
            else:
                break
            #If ux exits
            if action == common.UxAction.EXIT:
                break

    def run(self, **input_dict):
        """Run forward execution of the graph with debug

        Parameters
        ----------
        input_dict : dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)
        self._run_debug()

    def exit(self):
        """Exits the dump folder and all its contents"""
        self._remove_dump_root()
