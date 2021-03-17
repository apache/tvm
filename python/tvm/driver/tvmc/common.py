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

from urllib.parse import urlparse

import tvm

from tvm import relay
from tvm import transform


# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


class TVMCException(Exception):
    """TVMC Exception"""


def convert_graph_layout(mod, desired_layout):
    """Alter the layout of the input graph.

    Parameters
    ----------
    mod : tvm.relay.Module
        The relay module to convert.
    desired_layout : str
        The layout to convert to.

    Returns
    -------
    mod : tvm.relay.Module
        The converted module.
    """

    # Assume for the time being that graphs only have
    # conv2d as heavily-sensitive operators.
    desired_layouts = {
        "nn.conv2d": [desired_layout, "default"],
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


def validate_targets(parse_targets):
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
    if len(tvm_targets) > 1:
        verbose_tvm_targets = ", ".join(tvm_targets)
        raise TVMCException(
            f"Only one of the following targets can be used at a time. "
            "Found: {verbose_tvm_targets}."
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
    codegens = []

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

        codegens.append({"name": name, "opts": opts, "raw": raw_target})

    return codegens


def is_inline_json(target):
    try:
        json.loads(target)
        return True
    except json.decoder.JSONDecodeError:
        return False


def target_from_cli(target):
    """
    Create a tvm.target.Target instance from a
    command line interface (CLI) string.

    Parameters
    ----------
    target : str
        compilation target as plain string,
        inline JSON or path to a JSON file

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

        validate_targets(parsed_targets)
        target = parsed_targets[-1]["raw"]
        extra_targets = parsed_targets[:-1] if len(parsed_targets) > 1 else []

    return tvm.target.Target(target), extra_targets


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


def parse_shape_string(inputs_string):
    """Parse an input shape dictionary string to a usable dictionary.

    Parameters
    ----------
    inputs_string: str
        A string of the form "input_name:[dim1,dim2,...,dimn] input_name2:[dim1,dim2]" that
        indicates the desired shape for specific model inputs.

    Returns
    -------
    shape_dict: dict
        A dictionary mapping input names to their shape for use in relay frontend converters.
    """

    # Create a regex pattern that extracts each separate input mapping.
    pattern = r"\w+\:\s*\[\-?\d+(?:\,\s*\-?\d+)*\]"
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
        name, shape_string = mapping.split(":")
        # Convert shape string into a list of integers or Anys if negative.
        shape = [int(x) if int(x) > 0 else relay.Any() for x in shape_string.strip("][").split(",")]
        # Add parsed mapping to shape dictionary.
        shape_dict[name] = shape

    return shape_dict
