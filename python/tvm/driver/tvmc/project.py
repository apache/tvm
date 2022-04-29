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
TVMC Project Generation Functions
"""

import os
import pathlib
from collections import defaultdict
from typing import Union

from . import TVMCException
from .fmtopt import format_option


def get_project_options(project_info):
    """Get all project options as returned by Project API 'server_info_query'
    and return them in a dict indexed by the API method they belong to.


    Parameters
    ----------
    project_info: dict of list
        a dict of lists as returned by Project API 'server_info_query' among
        which there is a list called 'project_options' containing all the
        project options available for a given project/platform.

    Returns
    -------
    options_by_method: dict of list
        a dict indexed by the API method names (e.g. "generate_project",
        "build", "flash", or "open_transport") of lists containing all the
        options (plus associated metadata and formatted help text) that belong
        to a method.

        The metadata associated to the options include the field 'choices' and
        'required' which are convenient for parsers.

        The formatted help text field 'help_text' is a string that contains the
        name of the option, the choices for the option, and the option's default
        value.
    """
    options = project_info["project_options"]

    options_by_method = defaultdict(list)
    for opt in options:
        # Get list of methods associated with an option based on the
        # existance of a 'required' or 'optional' lists. API specification
        # guarantees at least one of these lists will exist. If a list does
        # not exist it's returned as None by the API.
        metadata = ["required", "optional"]
        option_methods = [(opt[md], bool(md == "required")) for md in metadata if opt[md]]
        for methods, is_opt_required in option_methods:
            for method in methods:
                name = opt["name"]

                # Only for boolean options set 'choices' accordingly to the
                # option type. API returns 'choices' associated to them
                # as None but 'choices' can be deduced from 'type' in this case.
                if opt["type"] == "bool":
                    opt["choices"] = ["true", "false"]

                if opt["choices"]:
                    choices = "{" + ", ".join(opt["choices"]) + "}"
                else:
                    choices = opt["name"].upper()
                option_choices_text = f"{name}={choices}"

                help_text = opt["help"][0].lower() + opt["help"][1:]

                if opt["default"]:
                    default_text = f"Defaults to '{opt['default']}'."
                else:
                    default_text = None

                formatted_help_text = format_option(
                    option_choices_text, help_text, default_text, is_opt_required
                )

                option = {
                    "name": opt["name"],
                    "choices": opt["choices"],
                    "help_text": formatted_help_text,
                    "required": is_opt_required,
                }
                options_by_method[method].append(option)

    return options_by_method


def get_options(options):
    """Get option and option value from the list options returned by the parser.

    Parameters
    ----------
    options: list of str
        list of strings of the form "option=value" as returned by the parser.

    Returns
    -------
    opts: dict
        dict indexed by option names and associated values.
    """

    opts = {}
    for option in options:
        try:
            k, v = option.split("=")
            opts[k] = v
        except ValueError:
            raise TVMCException(f"Invalid option format: {option}. Please use OPTION=VALUE.")

    return opts


def check_options(options, valid_options):
    """Check if an option (required or optional) is valid. i.e. in the list of valid options.

    Parameters
    ----------
    options: dict
        dict indexed by option name of options and options values to be checked.

    valid_options: list of dict
        list of all valid options and choices for a platform.

    Returns
    -------
    None. Raise TVMCException if check fails, i.e. if an option is not in the list of valid options.

    """
    required_options = [opt["name"] for opt in valid_options if opt["required"]]
    for required_option in required_options:
        if required_option not in options:
            raise TVMCException(
                f"Option '{required_option}' is required but was not specified. Use --list-options "
                "to see all required options."
            )

    remaining_options = set(options) - set(required_options)
    optional_options = [opt["name"] for opt in valid_options if not opt["required"]]
    for option in remaining_options:
        if option not in optional_options:
            raise TVMCException(
                f"Option '{option}' is invalid. Use --list-options to see all available options."
            )


def check_options_choices(options, valid_options):
    """Check if an option value is among the option's choices, when choices exist.

    Parameters
    ----------
    options: dict
        dict indexed by option name of options and options values to be checked.

    valid_options: list of dict
        list of all valid options and choices for a platform.

    Returns
    -------
    None. Raise TVMCException if check fails, i.e. if an option value is not valid.

    """
    # Dict of all valid options and associated valid choices.
    # Options with no choices are excluded from the dict.
    valid_options_choices = {
        opt["name"]: opt["choices"] for opt in valid_options if opt["choices"] is not None
    }

    for option in options:
        if option in valid_options_choices:
            if options[option] not in valid_options_choices[option]:
                raise TVMCException(
                    f"Choice '{options[option]}' for option '{option}' is invalid. "
                    "Use --list-options to see all available choices for that option."
                )


def get_and_check_options(passed_options, valid_options):
    """Get options and check if they are valid.  If choices exist for them, check values against it.

    Parameters
    ----------
    passed_options: list of str
        list of strings in the "key=value" form as captured by argparse.

    valid_option: list
        list with all options available for a given API method / project as returned by
        get_project_options().

    Returns
    -------
    opts: dict
        dict indexed by option names and associated values.

    Or None if passed_options is None.

    """

    if passed_options is None:
        # No options to check
        return None

    # From a list of k=v strings, make a dict options[k]=v
    opts = get_options(passed_options)
    # Check if passed options are valid
    check_options(opts, valid_options)
    # Check (when a list of choices exists) if the passed values are valid
    check_options_choices(opts, valid_options)

    return opts


def get_project_dir(project_dir: Union[pathlib.Path, str]) -> str:
    """Get project directory path"""
    if not os.path.isabs(project_dir):
        return os.path.abspath(project_dir)
    return project_dir
