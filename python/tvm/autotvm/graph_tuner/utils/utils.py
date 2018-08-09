# pylint: disable=eval-used,invalid-name,too-many-arguments
"""Utility functions"""
import os
import json

def get_factor(num):
    """Get all factors of a number.

    Parameters
    ----------
    num : int
        Input number.

    Returns
    -------
    out : list of int
        Factors of input number.
    """
    rtv = []
    for i in range(1, num + 1):
        if num % i == 0:
            rtv.append(i)
    return rtv


def log_msg(msg, file_logger, console_logger, log_func_str="info",
            verbose=True):
    """Log message.

    Parameters
    ----------
    msg : str
        Message to be logged.

    file_logger : logging.Logger
        File logger.

    console_logger : logging.Logger
        Console logger.

    log_func_str : str, optional
        Type of log function. Valid options are "debug", "info",
        "warning", "error" and "critical".

    verbose : boolean, optional
        Whether to log to console.
    """
    valid_log_func = ["debug", "info", "warning", "error", "critical"]
    if log_func_str not in valid_log_func:
        raise RuntimeError("Log function must be one of %s, but got %s"
                           % (str(valid_log_func), log_func_str))
    file_logger_func = getattr(file_logger, log_func_str)
    file_logger_func(msg)
    if verbose:
        console_logger_func = getattr(console_logger, log_func_str)
        console_logger_func(msg)
    if log_func_str == "error" or log_func_str == "critical":
        raise RuntimeError(msg)


def str2namedtuple(input_str, namedtuple_obj):
    """Convert string to namedtuple object.

    Parameters
    ----------
    input_str : str
        Input string.

    namedtuple_obj : object
        Name object of namedtuple.

    Returns
    -------
    output : namedtuple
        Output namedtuple.
    """
    tuple_name = namedtuple_obj.__name__
    tuple_fields = input_str[len(tuple_name):]
    if not input_str.startswith(str(tuple_name)):
        raise RuntimeError("%s format error. Expecting %s as tuple name."
                           % (input_str, tuple_name))
    return eval("namedtuple_obj%s" % tuple_fields)


def read_sch_from_json(sch_json_file, workload_obj, schedule_obj_list):
    """Read schedules from a json file.

    The json file should has following format:
        {
          "schedules":
            {
              "Workload(...)": [
                  {
                    "schedule": "AVXConvCommonFwd(...)",
                    "time": 0.15
                  },
                  {
                    "schedule": "AVXConvCommonFwd(...)",
                    "time": 0.135
                  },
                  ...
              ],
              "Workload(...)": [
                  {
                    "schedule": "AVXConv1x1Fwd(...)",
                    "time": 0.25
                  },
                  {
                    "schedule": "AVXConv1x1Fwd(...)",
                    "time": 0.4
                  },
              ],
              ...
            }
        }
    Other entries are allowed in json file.

    Workload and schedule namedtuple can be different. In this case,
    parse_wkl_func and parse_sch_func are required for customized
    workload and schedule formats.

    Parameters
    ==========
    sch_json_file : str
        Json file storing workloads and schedules.

    workload_obj : object
        Namedtuple object for workload.

    schedule_obj_list : list of object
        Namedtuple object for schedules.

    Returns
    -------
    out : dict of namedtuple to list of dict
        Schedule dictionary. Key is workload and value is a list of dictionary,
        which in format {"schedule": sch, "time": execution_time}.
        Time value is in millisecond.
    """
    with open(sch_json_file, "r") as jf:
        schedule_dict_str = json.load(jf)
    schedule_dict = {}
    for key, val in schedule_dict_str["schedules"].items():
        wkl = str2namedtuple(key, workload_obj)
        schedule_dict[wkl] = []
        for item in val:
            sch_str = item["schedule"]
            current_sch_obj = None
            for sch_obj in schedule_obj_list:
                if sch_str.startswith(sch_obj.__name__):
                    current_sch_obj = sch_obj
            if sch_str is None:
                raise RuntimeError("%s pattern not found in %s"
                                   % (sch_str, str(schedule_obj_list)))
            sch = str2namedtuple(sch_str, current_sch_obj)
            schedule_dict[wkl].append({"schedule": sch, "time": item["time"]})
    return schedule_dict


def write_sch_to_json(schedule_dict, sch_json_file):
    """Update schedule json file with the content of a schedule dictionary.
    If specified file doesn't exist, a new file will be created.

    Parameters
    ----------
    schedule_dict : dict of namedtuple to list of dict
        Schedule dictionary. Key is workload and value is a list of dictionary,
        which in format {"schedule": sch, "time": execution_time}.
        Time value is in millisecond.

    sch_json_file : str
        Json file storing workloads and schedules.
    """
    if not os.path.isfile(sch_json_file):
        schedule_dict_str = {}
    else:
        with open(sch_json_file, "r") as jf:
            schedule_dict_str = json.load(jf)
    if "schedules" not in schedule_dict_str:
        schedule_dict_str["schedules"] = {}
    for key, val in schedule_dict.items():
        schedule_dict_str["schedules"][str(key)] = []
        for item in val:
            sch = item["schedule"]
            time = item["time"]
            schedule_dict_str["schedules"][str(key)].append({"schedule": str(sch),
                                                             "time": time})
    with open(sch_json_file, "w") as lf:
        json.dump(schedule_dict_str, lf, indent=2)


def is_input_node(node_list, input_names, node_idx):
    """Whether a node is an input node.

    Parameters
    ----------
    node_list : list of dict of str to object
        List of all nodes in a graph.

    input_names : list of string
        List of input names.

    node_idx : int
        Node index to be checked.

    Returns
    -------
    out : bool
        whether node is a input node.
    """
    return node_list[node_idx]["name"] in input_names


def shape2layout(shape, layout_template):
    """Given a shape and a layout template, return the actual layout.
    For example, given shape (1, 8, 32, 32, 4) and layout template NCWHc,
    the return would be NCHW4c.

    Parameters
    ----------
    shape : tuple of int
        Input shape.

    layout_template : str
        Input layout template.

    Returns
    -------
    out : str
        Output layout.
    """
    if len(shape) != len(layout_template):
        raise RuntimeError("Shape and layout_template format mismatch: "
                           "%s vs %s." % (str(shape), layout_template))
    layout = ""
    for i, c in enumerate(layout_template):
        if not c.isalpha():
            raise RuntimeError("layout_template can only "
                               "contains alphabet character.")
        if c.islower():
            layout += "%d%c" % (shape[i], c)
        else:
            layout += c
    return layout


def get_wkl_map(graph, input_shapes, workload_list, target_op,
                get_wkl_func):
    """Get a dictionary maps node index of a graph to workload
    index in a workload list.

    Parameters
    ----------
    graph : nnvm Graph
        Input graph.

    input_shapes : dict of str to tuple of int
        Input shapes.

    workload_list : list of namedtuple
        Workload list.

    target_op : str
        Target operator name.

    get_wkl_func : function
        Function to convert nodes in a graph to workloads.

    Returns
    -------
    out : dict of int to int
        Dictionary maps node index of a graph to workload index.
    """
    g_dict = json.loads(graph.json())
    node_list = g_dict["nodes"]
    workload_map = {}
    for i, wkl in enumerate(workload_list):
        workload_map[wkl] = i
    graph_wkl_list = get_wkl_func(graph, input_shapes, unique_wkl=False)
    node_map = {}
    graph_wkl_idx = 0
    for idx, node in enumerate(node_list):
        if node["op"] != target_op:
            continue
        node_map[idx] = workload_map[graph_wkl_list[graph_wkl_idx]]
        graph_wkl_idx += 1
    return node_map


def get_real_node(in_node_dict, node_list, idx, target_op):
    """Get the index of first ancestor node with target_op as operator name.

    Parameters
    ----------
    in_node_dict : dict of int to list of int
        Dictionary maps node index to closest input ancestors.
        It can be created with get_in_nodes.

    node_list : list of dict of str to object
        List of all nodes in a graph.

    idx : int
        Input node index.

    target_op : str, optional
        Target operator name.

    Returns
    -------
    out : int
        Output node index.
    """
    if node_list[idx]["op"] == target_op or not in_node_dict[idx]:
        return idx
    anc_node_idx = in_node_dict[idx][0]
    anc_node = node_list[anc_node_idx]
    while anc_node["op"] != target_op:
        anc_node_idx = in_node_dict[anc_node_idx][0]
        anc_node = node_list[anc_node_idx]
    return anc_node_idx


def infer_layout_shape_avx(wkl, current_sch, target_sch, batch_size=1,
                           is_elemlike=False, elemlike_shape=None):
    """Infer actual input and output shapes for layout transformation
    given a workload, input schedule and output schedule.

    This function is for Intel AVX schedule template. Re-implement it
    for different workload and schedule templates.

    Take a CNN as example, a layout transformation can happen
    in two cases:
        1. Between two convolution nodes. Data shape before and after
           layout transformation can be determined purely by workload
           and schedules.
        2. Before element-wise like nodes. Element-wise like nodes
           are defined in _base module. In this case, shape of the
           element-wise like node is required as well.

    Parameters
    ----------
    wkl : namedtuple
        Input workload. If this is an element-wise like node, workload
        should come from the leftmost input node.

    current_sch : namedtuple
        Schedule before the layout transformation.

    target_sch : namedtuple
        Schedule after the layout transformation.

    is_elemlike : bool, optional
        Whether layout transformation happens before an element-wise
        like node.

    elemlike_shape : tuple of int, optional
        Shape of node data if layout transformation happens before
        an element-wise like node.
        Note: this shape should be inferred with original data layout.

    Returns
    -------
    in_shape : tuple of int
        Input shape of layout transformation.

    out_shape : tuple of int
        Output shape of layout transformation.

    is_valid : boolean
        Whether this is a valid layout transformation.
        An invalid transformation usually happens for concatenate operator.
    """
    if is_elemlike:
        if elemlike_shape is None:
            raise RuntimeError("elemlike_shape is required "
                               "if is_elemlike is True.")
        height = elemlike_shape[2]
        width = elemlike_shape[3]
    else:
        height = wkl.height
        width = wkl.width
    oc_bn_c = current_sch.oc_bn
    ic_bn_t = target_sch.ic_bn
    oc_bn_t = target_sch.oc_bn
    is_valid = True
    if is_elemlike:
        if wkl.out_filter % oc_bn_t != 0:
            is_valid = False
        in_shape = (batch_size, wkl.out_filter // oc_bn_c, height, width, oc_bn_c)
        out_shape = (batch_size, wkl.out_filter // oc_bn_t, height, width, oc_bn_t)
    else:
        if wkl.in_filter % oc_bn_c != 0:
            is_valid = False
        in_shape = (batch_size, wkl.in_filter // oc_bn_c, height, width, oc_bn_c)
        out_shape = (batch_size, wkl.in_filter // ic_bn_t, height, width, ic_bn_t)
    return in_shape, out_shape, is_valid
