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
"""
TVMC PassContext Interface
"""

import importlib

import tvm
from tvm.driver.tvmc import TVMCException


def load_function(full_name):
    """Dynamic loading a function by the full name.
    Parameters
    ----------
    full_name: str
        The name of a PackedFunc or a string of the form "path.to.module.func"
        that indicates the module that can be imported.
        You must be aware of the load order here, it first tries to find it via
        TVM global function, if not find, try to import it by "importlib.import_module".
    Returns
    -------
    func: function or PackedFunc
        The loaded fucntion.
    """
    global_func = tvm.get_global_func(full_name, allow_missing=True)
    if global_func is not None:
        return global_func

    # split full name "path.to.module.func" into two parts ["path.to.module", "func"]
    module_name, func_name = full_name.rsplit(".", 1)

    # import module and find the function
    module = importlib.import_module(module_name)
    if hasattr(module, func_name):
        return getattr(module, func_name)

    raise TVMCException(f"No function '{func_name}' found in module '{module_name}'.")


def get_pass_config_value(name, value, config_type):
    """Get a PassContext configuration value, based on its config data type.

    Parameters
    ----------
    name: str
        config identifier name.
    value: str
        value assigned to the config, provided via command line.
    config_type: str
        data type defined to the config, as string.

    Returns
    -------
    parsed_value: bool, int or str
        a representation of the input value, converted to the type
        specified by config_type.
    """

    parsed_value = None

    if config_type == "IntImm":
        # "Bool" configurations in the PassContext are recognized as
        # IntImm, so deal with this case here
        mapping_values = {
            "false": False,
            "true": True,
        }

        if value.isdigit():
            parsed_value = int(value)
        else:
            # if not an int, accept only values on the mapping table, case insensitive
            parsed_value = mapping_values.get(value.lower(), None)

        if parsed_value is None:
            raise TVMCException(f"Invalid value '{value}' for configuration '{name}'.")

    elif config_type == "runtime.String":
        parsed_value = value

    elif config_type == "Array":
        if name == "tir.add_lower_pass":
            pass_list = value.split(",")
            if len(pass_list) % 2 != 0:
                raise TVMCException(
                    f"The configuration of '{name}' must be of the form "
                    "'tir.add_lower_pass=opt_level1,pass1,opt_evel2,pass2'"
                )

            parsed_value = []
            for i in range(0, len(pass_list), 2):
                level, pass_func = pass_list[i].strip(), pass_list[i + 1].strip()
                try:
                    level = int(level)
                except ValueError:
                    raise TVMCException(f"Only integer is allow for configuration '{name}'.")

                # TODO (@leeexyz) We should parse configurations of each tir Pass.
                #     For now, we only use the defaults. Currently, There are four config nodes:
                #     `tir.transform.LoopPartitionConfig`
                #     `tir.transform.UnrollLoopConfig`
                #     `tir.transform.HoistIfThenElseConfig`
                #     `tir.transform.InjectDoubleBufferConfig`
                # loading pass func and calling it to get the Pass
                pass_func = load_function(pass_func)()
                parsed_value.append((level, pass_func))
        else:
            raise TVMCException(f"Unsupported configuration '{name}' for '{config_type}' type.")

    else:
        # not raise here cause we alreay checked before calling this function
        pass

    return parsed_value


def parse_configs(input_configs):
    """Parse configuration values set via command line.

    Parameters
    ----------
    input_configs: list of str
        list of configurations provided via command line.

    Returns
    -------
    pass_context_configs: dict
        a dict containing key-value configs to be used in the PassContext.
    """
    if not input_configs:
        return {}

    all_configs = tvm.ir.transform.PassContext.list_configs()
    supported_config_types = ("IntImm", "runtime.String", "Array")
    supported_configs = [
        name for name in all_configs.keys() if all_configs[name]["type"] in supported_config_types
    ]

    pass_context_configs = {}

    for config in input_configs:
        if not config:
            raise TVMCException(
                f"Invalid format for configuration '{config}', use <config>=<value>"
            )

        # Each config is expected to be provided as "name=value"
        try:
            name, value = config.split("=")
            name = name.strip()
            value = value.strip()
        except ValueError:
            raise TVMCException(
                f"Invalid format for configuration '{config}', use <config>=<value>"
            )

        if name not in all_configs:
            raise TVMCException(
                f"Configuration '{name}' is not defined in TVM. "
                f"These are the existing configurations: {', '.join(all_configs)}"
            )

        if name not in supported_configs:
            raise TVMCException(
                f"Configuration '{name}' uses a data type not supported by TVMC. "
                f"The following configurations are supported: {', '.join(supported_configs)}"
            )

        config_type = all_configs[name]["type"]
        parsed_value = get_pass_config_value(name, value, config_type)

        if config_type == "Array" and name in pass_context_configs:
            # merge configs if the configuration exists
            pass_context_configs[name].extend(parsed_value)
        else:
            pass_context_configs[name] = parsed_value

    return pass_context_configs
