#!/usr/bin/env python

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
manipulate json config file to work with TVMC
"""
import os
import json

from tvm._ffi import libinfo
from tvm.driver.tvmc import TVMCException

CONFIGS_JSON_DIR = None


class ConfigsJsonNotFoundError(TVMCException):
    """Raised when the JSON configs dirtree cannot be found."""


def get_configs_json_dir() -> str:
    """Find the 'configs' directory, containing the JSON files used to configure tvmc
    with persistent argument settings.

    Returns
    -------
    str :
        The path to the 'configs' directory
    """
    global CONFIGS_JSON_DIR
    if CONFIGS_JSON_DIR is None:

        # If a non-default location for the build directory is used, e.g. set via TVM_LIBRARY_PATH
        # we need to provide the user a way to overwrite CONFIGS_JSON_DIR as well.
        if os.environ.get("TVM_CONFIGS_JSON_DIR", None):
            user_config_dir = os.environ["TVM_CONFIGS_JSON_DIR"]
            if os.path.isdir(user_config_dir):
                CONFIGS_JSON_DIR = user_config_dir
                return CONFIGS_JSON_DIR

        candidate_paths = []
        candidate_paths.extend(libinfo.find_lib_path())
        # When running from source, the configs directory will be located one directory above the
        # native libraries, so covering that case.
        candidate_paths.extend(
            [os.path.abspath(os.path.join(lib_path, "..")) for lib_path in libinfo.find_lib_path()]
        )
        candidate_paths.extend(
            [
                os.path.abspath(os.path.join(lib_path, "../.."))
                for lib_path in libinfo.find_lib_path()
            ]
        )
        for path in candidate_paths:
            configs_path = os.path.join(os.path.dirname(path), "configs")
            if os.path.isdir(configs_path):
                CONFIGS_JSON_DIR = configs_path
                break

        else:
            raise ConfigsJsonNotFoundError()

    return CONFIGS_JSON_DIR


def find_json_file(name, path):
    """search for json file given file name a path

    Parameters
    ----------
    name: string
        the file name need to be searched
    path: string
        path to search at

    Returns
    -------
    string
        the full path to that file

    """
    match = ""
    for root, _dirs, files in os.walk(path):
        if name in files:
            match = os.path.join(root, name)
            break

    return match


def read_and_convert_json_into_dict(config_args):
    """Read json configuration file and return a dictionary with all parameters

    Parameters
    ----------
    args: argparse.Namespace
        Arguments from command line parser holding the json file path.

    Returns
    -------
    dictionary
        dictionary with all the json arguments keys and values

    """
    try:
        if ".json" not in config_args.config:
            config_args.config = config_args.config.strip() + ".json"
        if os.path.isfile(config_args.config):
            json_config_file = config_args.config
        else:
            config_dir = get_configs_json_dir()
            json_config_file = find_json_file(config_args.config, config_dir)
        return json.load(open(json_config_file, "rb"))

    except FileNotFoundError:
        raise TVMCException(
            f"File {config_args.config} does not exist at {config_dir} or is wrong format."
        )


def parse_target_from_json(one_target, command_line_list):
    """parse the targets out of the json file struct

    Parameters
    ----------
    one_target: dict
        dictionary with all target's details
    command_line_list: list
        list to update with target parameters
    """
    target_kind, *sub_type = [
        one_target[key] if key == "kind" else (key, one_target[key]) for key in one_target
    ]

    internal_dict = {}
    if sub_type:
        sub_target_type = sub_type[0][0]
        target_value = sub_type[0][1]
        internal_dict[f"target_{target_kind}_{sub_target_type}"] = target_value
        command_line_list.append(internal_dict)

    return target_kind


def convert_config_json_to_cli(json_params):
    """convert all configuration keys & values from dictionary to cli format

    Parameters
    ----------
    args: dictionary
        dictionary with all configuration keys & values.

    Returns
    -------
    int
        list of configuration values in cli format

    """
    command_line_list = []
    for param_key in json_params:
        if param_key == "targets":
            target_list = [
                parse_target_from_json(one_target, command_line_list)
                for one_target in json_params[param_key]
            ]

            internal_dict = {}
            internal_dict["target"] = ", ".join(map(str, target_list))
            command_line_list.append(internal_dict)

        elif param_key in ("executor", "runtime"):
            for key, value in json_params[param_key].items():
                if key == "kind":
                    kind = f"{value}_"
                    new_dict_key = param_key
                else:
                    new_dict_key = f"{param_key}_{kind}{key}"

                internal_dict = {}
                internal_dict[new_dict_key.replace("-", "_")] = value
                command_line_list.append(internal_dict)

        elif isinstance(json_params[param_key], dict):
            internal_dict = {}
            modify_param_key = param_key.replace("-", "_")
            internal_dict[modify_param_key] = []
            for key, value in json_params[param_key].items():
                internal_dict[modify_param_key].append(f"{key}={value}")
            command_line_list.append(internal_dict)

        else:
            internal_dict = {}
            internal_dict[param_key.replace("-", "_")] = json_params[param_key]
            command_line_list.append(internal_dict)

    return command_line_list
