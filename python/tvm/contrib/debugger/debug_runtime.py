"""Graph debug runtime executes TVM debug packed functions."""

import os
import tempfile
import shutil
from tvm import ndarray as nd
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
        The graph to be deployed in json format output by nnvm graph.
        The graph can only contain one operator(tvm_op) that
        points to the name of PackedFunc in the libmod.

    dbg_ux : str
        To select which ui user needs, curses, tensorboard

    dump_root : str
        To select which folder the outputs should be kept.
        None will make a temp folder in /tmp/tvmdbg<rand_string> and does the dumping
    """
    def __init__(self, module, ctx, graph_json_str, dbg_ux, dump_root):
        self.ui_obj = None
        self._dump_root = dump_root
        self._dump_path = None
        self._debug_buffer = module["set_debug_buffer"]
        self._debug_run = module["debug_run"]
        graph_runtime.GraphModule.__init__(self, module, ctx)
        self._prepare_data_and_ui(graph_json_str, ctx, dbg_ux)

    def _format_context(self, ctx):
        return str(ctx).upper().replace("(", ":").replace(")", "")

    def _prepare_data_and_ui(self, graph_json, ctx, dbg_ux):
        """Create the framework for debug data dumpling and initialize the frontend

        Parameters
        ----------
        graph_json : str or graph class
            The graph to be deployed in json format output by nnvm graph.
            The graph can only contain one operator(tvm_op) that
            points to the name of PackedFunc in the libmod.

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
        # prepare the debug out buffer list
        self._make_debug_buffer_list()


    def _get_debug_buffer_count(self):
        return len(self.dbg_buff_list)

    def _get_debug_buffer(self, eid):
        return self.dbg_buff_list[eid]

    def _set_debug_buffer(self):
        """Set output buffer allocated for each node to copy the node's
        output after Run completed.

        This function will get called before run performs.
        GraphRuntime copy the execution out to the allocated memory for each nodes.
        """
        for eid in range(self._get_debug_buffer_count()):
            self._debug_buffer(self._get_debug_buffer(eid))

    def _ensure_dir(self, directory):
        """Create a directory if not exists

        Parameters
        ----------

        directory : str
            File path to create
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _get_dump_path(self, ctx):
        """Dump json formatted graph.

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

    def _make_debug_buffer_list(self):
        """Allocate output buffer for each node to copy the node's
        output after Run completed.
        """
        debug_datum = self.debug_datum
        shapes_list = debug_datum.get_graph_node_shapes()
        dtype_list = debug_datum.get_graph_node_dtypes()
        dbg_out_buffer_list = []
        for i in range(len(shapes_list[1])):
            dbg_out_buffer_list.append(nd.empty(shapes_list[1][i], dtype_list[1][i]))
        self.dbg_buff_list = dbg_out_buffer_list

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

        nodes_count = len(self.debug_datum.get_graph_nodes())
        for i in range(nodes_count):
            time_stamp = self._debug_run(i)
            self.debug_datum.set_time(time_stamp)

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
                # Step 1. Set the debug buffer to catch the output
                self._set_debug_buffer()
                # Step 2. Execute the graph
                retvals = self._debug_run_op_exec()
                # Step 3. Dump the output tensors to the dump folder
                self.debug_datum.dump_output_tensor(self.dbg_buff_list)
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

    def set_input(self, key=None, value=None, **params):
        """Set inputs to the module via kwargs

        Along with the value setting to runtime, the same will be notified to
        UI frontend as well.

        Parameters
        ----------

        key : int or str
           The input key

        value : the input value.
           The input key

        params : dict of str to NDArray
           Additonal arguments
        """
        super(GraphModuleDebug, self).set_input(key, value, **params)

        if key:
            self.ui_obj.set_input(key, value)

    def exit(self):
        """Exits the dump folder and all its contents"""
        self._remove_dump_root()
