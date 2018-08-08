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
    valid_log_func = ["debug", "info", "warning", "error", "critical"]
    if log_func_str not in valid_log_func:
        raise RuntimeError("Log function must be one of %s, but got %s"
                           % (str(valid_log_func), log_func_str))
    file_logger_func = getattr(file_logger, log_func_str)
    console_logger_func = getattr(console_logger, log_func_str)
    file_logger_func(msg)
    if verbose:
        console_logger_func(msg)
    if log_func_str == "error" or log_func_str == "critical":
        raise RuntimeError(msg)

def str2namedtuple(input_str, namedtuple_obj):
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
