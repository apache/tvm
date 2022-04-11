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
This file contains functions for processing target inputs for the TVMC CLI
"""

import os
import logging
import json
import re

import tvm
from tvm.driver import tvmc
from tvm.driver.tvmc import TVMCException
from tvm.driver.tvmc.composite_target import get_codegen_by_target, get_codegen_names
from tvm.ir.attrs import make_node, _ffi_api as attrs_api
from tvm.ir.transform import PassContext
from tvm.target import Target, TargetKind

# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")

# We can't tell the type inside an Array but all current options are strings so
# it can default to that. Bool is used alongside Integer but aren't distinguished
# between as both are represented by IntImm
INTERNAL_TO_NATIVE_TYPE = {"runtime.String": str, "IntImm": int, "Array": str}
INTERNAL_TO_HELP = {"runtime.String": " string", "IntImm": "", "Array": " options"}


def _valid_target_kinds():
    codegen_names = tvmc.composite_target.get_codegen_names()
    return filter(lambda target: target not in codegen_names, Target.list_kinds())


def _generate_target_kind_args(parser, kind_name):
    target_group = parser.add_argument_group(f"target {kind_name}")
    for target_option, target_type in TargetKind.options_from_name(kind_name).items():
        if target_type in INTERNAL_TO_NATIVE_TYPE:
            target_group.add_argument(
                f"--target-{kind_name}-{target_option}",
                type=INTERNAL_TO_NATIVE_TYPE[target_type],
                help=f"target {kind_name} {target_option}{INTERNAL_TO_HELP[target_type]}",
            )


def _generate_codegen_args(parser, codegen_name):
    codegen = get_codegen_by_target(codegen_name)
    pass_configs = PassContext.list_configs()

    if codegen["config_key"] is not None and codegen["config_key"] in pass_configs:
        target_group = parser.add_argument_group(f"target {codegen_name}")
        attrs = make_node(pass_configs[codegen["config_key"]]["type"])
        fields = attrs_api.AttrsListFieldInfo(attrs)
        for field in fields:
            for tvm_type, python_type in INTERNAL_TO_NATIVE_TYPE.items():
                if field.type_info.startswith(tvm_type):
                    target_option = field.name
                    target_group.add_argument(
                        f"--target-{codegen_name}-{target_option}",
                        type=python_type,
                        help=f"target {codegen_name} {target_option}{python_type}",
                    )


def generate_target_args(parser):
    """Walks through the TargetKind registry and generates arguments for each Target's options"""
    parser.add_argument(
        "--target",
        help="compilation target as plain string, inline JSON or path to a JSON file",
        required=False,
    )
    for target_kind in _valid_target_kinds():
        _generate_target_kind_args(parser, target_kind)
    for codegen_name in get_codegen_names():
        _generate_codegen_args(parser, codegen_name)


def _reconstruct_target_kind_args(args, kind_name):
    kind_options = {}
    for target_option, target_type in TargetKind.options_from_name(kind_name).items():
        if target_type in INTERNAL_TO_NATIVE_TYPE:
            var_name = f"target_{kind_name.replace('-', '_')}_{target_option.replace('-', '_')}"
            option_value = getattr(args, var_name)
            if option_value is not None:
                kind_options[target_option] = getattr(args, var_name)
    return kind_options


def _reconstruct_codegen_args(args, codegen_name):
    codegen = get_codegen_by_target(codegen_name)
    pass_configs = PassContext.list_configs()
    codegen_options = {}

    if codegen["config_key"] is not None and codegen["config_key"] in pass_configs:
        attrs = make_node(pass_configs[codegen["config_key"]]["type"])
        fields = attrs_api.AttrsListFieldInfo(attrs)
        for field in fields:
            for tvm_type in INTERNAL_TO_NATIVE_TYPE:
                if field.type_info.startswith(tvm_type):
                    target_option = field.name
                    var_name = (
                        f"target_{codegen_name.replace('-', '_')}_{target_option.replace('-', '_')}"
                    )
                    option_value = getattr(args, var_name)
                    if option_value is not None:
                        codegen_options[target_option] = option_value
    return codegen_options


def reconstruct_target_args(args):
    """Reconstructs the target options from the arguments"""
    reconstructed = {}
    for target_kind in _valid_target_kinds():
        kind_options = _reconstruct_target_kind_args(args, target_kind)
        if kind_options:
            reconstructed[target_kind] = kind_options

    for codegen_name in get_codegen_names():
        codegen_options = _reconstruct_codegen_args(args, codegen_name)
        if codegen_options:
            reconstructed[codegen_name] = codegen_options

    return reconstructed


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

    tvm_targets = [t for t in targets if t in _valid_target_kinds()]
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
        except ValueError as error:
            raise TVMCException(f"Error parsing target string '{target}'.\nThe error was: {error}")

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

        extra_targets = [
            _combine_target_options(t, additional_target_options)
            for t in parsed_targets
            if not t["is_tvm_target"]
        ]

    return tvm.target.Target(target, host=target_host), extra_targets
