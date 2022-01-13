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

import tvm
from tvm.driver.tvmc import TVMCException


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
            raise TVMCException(f"Invalid value '{value}' for configuration '{name}'. ")

    if config_type == "runtime.String":
        parsed_value = value

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
    supported_config_types = ("IntImm", "runtime.String")
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

        parsed_value = get_pass_config_value(name, value, all_configs[name]["type"])
        pass_context_configs[name] = parsed_value

    return pass_context_configs
