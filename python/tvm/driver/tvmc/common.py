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
Common utility functions shared by TVMC modules.
"""
import re
import json
import logging
import os.path
import argparse

from collections import defaultdict
from urllib.parse import urlparse

import tvm

from tvm.driver import tvmc
from tvm import relay
from tvm import transform
from tvm._ffi import registry
from .fmtopt import format_option

# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


class TVMCException(Exception):
    """TVMC Exception"""


class TVMCSuppressedArgumentParser(argparse.ArgumentParser):
    """
    A silent ArgumentParser class.

    This class is meant to be used as a helper for creating dynamic parsers in
    TVMC. It will create a "supressed" parser based on an existing one (parent)
    which does not include a help message, does not print a usage message (even
    when -h or --help is passed) and does not exit on invalid choice parse
    errors but rather throws a TVMCException so it can be handled and the
    dynamic parser construction is not interrupted prematurely.

    """

    def __init__(self, parent, **kwargs):
        # Don't add '-h' or '--help' options to the newly created parser. Don't print usage message.
        # 'add_help=False' won't supress existing '-h' and '--help' options from the parser (and its
        # subparsers) present in 'parent'. However that class is meant to be used with the main
        # parser, which is created with `add_help=False` - the help is added only later. Hence it
        # the newly created parser won't have help options added in its (main) root parser. The
        # subparsers in the main parser will eventually have help activated, which is enough for its
        # use in TVMC.
        super().__init__(parents=[parent], add_help=False, usage=argparse.SUPPRESS, **kwargs)

    def exit(self, status=0, message=None):
        # Don't exit on error when parsing the command line.
        # This won't catch all the errors generated when parsing tho. For instance, it won't catch
        # errors due to missing required arguments. But this will catch "error: invalid choice",
        # which is what it's necessary for its use in TVMC.
        raise TVMCException()


def convert_graph_layout(mod, desired_layout):
    """Alter the layout of the input graph.

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module to convert.
    desired_layout : str
        The layout to convert to.

    Returns
    -------
    mod : tvm.IRModule
        The converted module.
    """

    # Assume for the time being that graphs only have
    # conv2d as heavily-sensitive operators.
    desired_layouts = {
        "nn.conv2d": [desired_layout, "default"],
        "nn.conv2d_transpose": [desired_layout, "default"],
        "qnn.conv2d": [desired_layout, "default"],
    }

    # Convert the layout of the graph where possible.
    seq = transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
        ]
    )

    with transform.PassContext(opt_level=3):
        try:
            return seq(mod)
        except Exception as err:
            raise TVMCException(
                "Error converting layout to {0}: {1}".format(desired_layout, str(err))
            )


def validate_targets(parse_targets, additional_target_options=None):
    """
    Apply a series of validations in the targets provided via CLI.
    """
    tvm_target_kinds = tvm.target.Target.list_kinds()
    targets = [t["name"] for t in parse_targets]

    if len(targets) > len(set(targets)):
        raise TVMCException("Duplicate target definitions are not allowed")

    if targets[-1] not in tvm_target_kinds:
        tvm_target_names = ", ".join(tvm_target_kinds)
        raise TVMCException(
            f"The last target needs to be a TVM target. Choices: {tvm_target_names}"
        )

    tvm_targets = [t for t in targets if t in tvm_target_kinds]
    if len(tvm_targets) > 2:
        verbose_tvm_targets = ", ".join(tvm_targets)
        raise TVMCException(
            "Only two of the following targets can be used at a time. "
            f"Found: {verbose_tvm_targets}."
        )

    if additional_target_options is not None:
        for target_name in additional_target_options:
            if not any([target for target in parse_targets if target["name"] == target_name]):
                first_option = list(additional_target_options[target_name].keys())[0]
                raise TVMCException(
                    f"Passed --target-{target_name}-{first_option}"
                    f" but did not specify {target_name} target"
                )


def tokenize_target(target):
    """
    Extract a list of tokens from a target specification text.

    It covers some corner-cases that are not covered by the built-in
    module 'shlex', such as the use of "+" as a punctuation character.


    Example
    -------

    For the input `foo -op1=v1 -op2="v ,2", bar -op3=v-4` we
    should obtain:

        ["foo", "-op1=v1", "-op2="v ,2"", ",", "bar", "-op3=v-4"]

    Parameters
    ----------
    target : str
        Target options sent via CLI arguments

    Returns
    -------
    list of str
        a list of parsed tokens extracted from the target string
    """

    # Regex to tokenize the "--target" value. It is split into five parts
    # to match with:
    #  1. target and option names e.g. llvm, -mattr=, -mcpu=
    #  2. option values, all together, without quotes e.g. -mattr=+foo,+opt
    #  3. option values, when single quotes are used e.g. -mattr='+foo, +opt'
    #  4. option values, when double quotes are used e.g. -mattr="+foo ,+opt"
    #  5. commas that separate different targets e.g. "my-target, llvm"
    target_pattern = (
        r"(\-{0,2}[\w\-]+\=?"
        r"(?:[\w\+\-\.]+(?:,[\w\+\-\.])*"
        r"|[\'][\w\+\-,\s\.]+[\']"
        r"|[\"][\w\+\-,\s\.]+[\"])*"
        r"|,)"
    )

    return re.findall(target_pattern, target)


def parse_target(target):
    """
    Parse a plain string of targets provided via a command-line
    argument.

    To send more than one codegen, a comma-separated list
    is expected. Options start with -<option_name>=<value>.

    We use python standard library 'shlex' to parse the argument in
    a POSIX compatible way, so that if options are defined as
    strings with spaces or commas, for example, this is considered
    and parsed accordingly.


    Example
    -------

    For the input `--target="foo -op1=v1 -op2="v ,2", bar -op3=v-4"` we
    should obtain:

      [
        {
            name: "foo",
            opts: {"op1":"v1", "op2":"v ,2"},
            raw: 'foo -op1=v1 -op2="v ,2"'
        },
        {
            name: "bar",
            opts: {"op3":"v-4"},
            raw: 'bar -op3=v-4'
        }
      ]

    Parameters
    ----------
    target : str
        Target options sent via CLI arguments

    Returns
    -------
    codegens : list of dict
        This list preserves the order in which codegens were
        provided via command line. Each Dict contains three keys:
        'name', containing the name of the codegen; 'opts' containing
        a key-value for all options passed via CLI; 'raw',
        containing the plain string for this codegen
    """
    codegen_names = tvmc.composite_target.get_codegen_names()
    codegens = []

    tvm_target_kinds = tvm.target.Target.list_kinds()
    parsed_tokens = tokenize_target(target)

    split_codegens = []
    current_codegen = []
    split_codegens.append(current_codegen)
    for token in parsed_tokens:
        # every time there is a comma separating
        # two codegen definitions, prepare for
        # a new codegen
        if token == ",":
            current_codegen = []
            split_codegens.append(current_codegen)
        else:
            # collect a new token for the current
            # codegen being parsed
            current_codegen.append(token)

    # at this point we have a list of lists,
    # each item on the first list is a codegen definition
    # in the comma-separated values
    for codegen_def in split_codegens:
        # the first is expected to be the name
        name = codegen_def[0]
        is_tvm_target = name in tvm_target_kinds and name not in codegen_names
        raw_target = " ".join(codegen_def)
        all_opts = codegen_def[1:] if len(codegen_def) > 1 else []
        opts = {}
        for opt in all_opts:
            try:
                # deal with -- prefixed flags
                if opt.startswith("--"):
                    opt_name = opt[2:]
                    opt_value = True
                else:
                    opt = opt[1:] if opt.startswith("-") else opt
                    opt_name, opt_value = opt.split("=", maxsplit=1)

                    # remove quotes from the value: quotes are only parsed if they match,
                    # so it is safe to assume that if the string starts with quote, it ends
                    # with quote.
                    opt_value = opt_value[1:-1] if opt_value[0] in ('"', "'") else opt_value
            except ValueError:
                raise ValueError(f"Error when parsing '{opt}'")

            opts[opt_name] = opt_value

        codegens.append(
            {"name": name, "opts": opts, "raw": raw_target, "is_tvm_target": is_tvm_target}
        )

    return codegens


def is_inline_json(target):
    try:
        json.loads(target)
        return True
    except json.decoder.JSONDecodeError:
        return False


def _combine_target_options(target, additional_target_options=None):
    if additional_target_options is None:
        return target
    if target["name"] in additional_target_options:
        target["opts"].update(additional_target_options[target["name"]])
    return target


def _recombobulate_target(target):
    name = target["name"]
    opts = " ".join([f"-{key}={value}" for key, value in target["opts"].items()])
    return f"{name} {opts}"


def target_from_cli(target, additional_target_options=None):
    """
    Create a tvm.target.Target instance from a
    command line interface (CLI) string.

    Parameters
    ----------
    target : str
        compilation target as plain string,
        inline JSON or path to a JSON file

    additional_target_options: Optional[Dict[str, Dict[str,str]]]
        dictionary of additional target options to be
        combined with parsed targets

    Returns
    -------
    tvm.target.Target
        an instance of target device information
    extra_targets : list of dict
        This list preserves the order in which extra targets were
        provided via command line. Each Dict contains three keys:
        'name', containing the name of the codegen; 'opts' containing
        a key-value for all options passed via CLI; 'raw',
        containing the plain string for this codegen
    """
    extra_targets = []

    if os.path.isfile(target):
        with open(target) as target_file:
            logger.debug("target input is a path: %s", target)
            target = "".join(target_file.readlines())
    elif is_inline_json(target):
        logger.debug("target input is inline JSON: %s", target)
    else:
        logger.debug("target input is plain text: %s", target)
        try:
            parsed_targets = parse_target(target)
        except ValueError as ex:
            raise TVMCException(f"Error parsing target string '{target}'.\nThe error was: {ex}")

        validate_targets(parsed_targets, additional_target_options)
        tvm_targets = [
            _combine_target_options(t, additional_target_options)
            for t in parsed_targets
            if t["is_tvm_target"]
        ]

        # Validated target strings have 1 or 2 tvm targets, otherwise
        # `validate_targets` above will fail.
        if len(tvm_targets) == 1:
            target = _recombobulate_target(tvm_targets[0])
            target_host = None
        else:
            assert len(tvm_targets) == 2
            target = _recombobulate_target(tvm_targets[0])
            target_host = _recombobulate_target(tvm_targets[1])

        extra_targets = [t for t in parsed_targets if not t["is_tvm_target"]]

    return tvm.target.Target(target, host=target_host), extra_targets


def tracker_host_port_from_cli(rpc_tracker_str):
    """Extract hostname and (optional) port from strings
    like "1.2.3.4:9090" or "4.3.2.1".

    Used as a helper function to cover --rpc-tracker
    command line argument, in different subcommands.

    Parameters
    ----------
    rpc_tracker_str : str
        hostname (or IP address) and port of the RPC tracker,
        in the format 'hostname[:port]'.

    Returns
    -------
    rpc_hostname : str or None
        hostname or IP address, extracted from input.
    rpc_port : int or None
        port number extracted from input (9090 default).
    """

    rpc_hostname = rpc_port = None

    if rpc_tracker_str:
        parsed_url = urlparse("//%s" % rpc_tracker_str)
        rpc_hostname = parsed_url.hostname
        rpc_port = parsed_url.port or 9090
        logger.info("RPC tracker hostname: %s", rpc_hostname)
        logger.info("RPC tracker port: %s", rpc_port)

    return rpc_hostname, rpc_port


def parse_pass_list_str(input_string):
    """Parse an input string for existing passes

    Parameters
    ----------
    input_string: str
        Possibly comma-separated string with the names of passes

    Returns
    -------
    list: a list of existing passes.
    """
    _prefix = "relay._transform."
    pass_list = input_string.split(",")
    missing_list = [
        p.strip()
        for p in pass_list
        if len(p.strip()) > 0 and tvm.get_global_func(_prefix + p.strip(), True) is None
    ]
    if len(missing_list) > 0:
        available_list = [
            n[len(_prefix) :] for n in registry.list_global_func_names() if n.startswith(_prefix)
        ]
        raise argparse.ArgumentTypeError(
            "Following passes are not registered within tvm: {}. Available: {}.".format(
                ", ".join(missing_list), ", ".join(sorted(available_list))
            )
        )
    return pass_list


def parse_shape_string(inputs_string):
    """Parse an input shape dictionary string to a usable dictionary.

    Parameters
    ----------
    inputs_string: str
        A string of the form "input_name:[dim1,dim2,...,dimn] input_name2:[dim1,dim2]" that
        indicates the desired shape for specific model inputs. Colons, forward slashes and dots
        within input_names are supported. Spaces are supported inside of dimension arrays.

    Returns
    -------
    shape_dict: dict
        A dictionary mapping input names to their shape for use in relay frontend converters.
    """

    # Create a regex pattern that extracts each separate input mapping.
    # We want to be able to handle:
    # * Spaces inside arrays
    # * forward slashes inside names (but not at the beginning or end)
    # * colons inside names (but not at the beginning or end)
    # * dots inside names
    pattern = r"(?:\w+\/)?[:\w.]+\:\s*\[\-?\d+(?:\,\s*\-?\d+)*\]"
    input_mappings = re.findall(pattern, inputs_string)
    if not input_mappings:
        raise argparse.ArgumentTypeError(
            "--input-shapes argument must be of the form "
            '"input_name:[dim1,dim2,...,dimn] input_name2:[dim1,dim2]"'
        )
    shape_dict = {}
    for mapping in input_mappings:
        # Remove whitespace.
        mapping = mapping.replace(" ", "")
        # Split mapping into name and shape.
        name, shape_string = mapping.rsplit(":", 1)
        # Convert shape string into a list of integers or Anys if negative.
        shape = [int(x) if int(x) > 0 else relay.Any() for x in shape_string.strip("][").split(",")]
        # Add parsed mapping to shape dictionary.
        shape_dict[name] = shape

    return shape_dict


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
        om = [(opt[md], bool(md == "required")) for md in metadata if opt[md]]
        for methods, is_opt_required in om:
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
