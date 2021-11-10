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

from tvm.driver import tvmc
from tvm.target import Target

# We can't tell the type inside an Array but all current options are strings so
# it can default to that. Bool is used alongside Integer but aren't distinguished
# between as both are represented by IntImm
INTERNAL_TO_NATIVE_TYPE = {"runtime.String": str, "IntImm": int, "Array": str}
INTERNAL_TO_HELP = {"runtime.String": " string", "IntImm": "", "Array": " options"}


def _valid_target_kinds():
    codegen_names = tvmc.composite_target.get_codegen_names()
    return filter(lambda target: target not in codegen_names, Target.list_kinds())


def _generate_target_kind_args(parser, kind):
    target_group = parser.add_argument_group(f"target {kind.name}")
    for target_option, target_type in kind.options.items():
        if target_type in INTERNAL_TO_NATIVE_TYPE:
            target_group.add_argument(
                f"--target-{kind.name}-{target_option}",
                type=INTERNAL_TO_NATIVE_TYPE[target_type],
                help=f"target {kind.name} {target_option}{INTERNAL_TO_HELP[target_type]}",
            )


def generate_target_args(parser):
    """Walks through the TargetKind registry and generates arguments for each Target's options"""
    parser.add_argument(
        "--target",
        help="compilation target as plain string, inline JSON or path to a JSON file",
        required=True,
    )
    for target_kind in _valid_target_kinds():
        target = Target(target_kind)
        _generate_target_kind_args(parser, target.kind)


def _reconstruct_target_kind_args(args, kind):
    kind_options = {}
    for target_option, target_type in kind.options.items():
        if target_type in INTERNAL_TO_NATIVE_TYPE:
            var_name = f"target_{kind.name.replace('-', '_')}_{target_option.replace('-', '_')}"
            option_value = getattr(args, var_name)
            if option_value is not None:
                kind_options[target_option] = getattr(args, var_name)
    return kind_options


def reconstruct_target_args(args):
    """Reconstructs the target options from the arguments"""
    reconstructed = {}
    for target_kind in _valid_target_kinds():
        target = Target(target_kind)
        kind_options = _reconstruct_target_kind_args(args, target.kind)
        if kind_options:
            reconstructed[target.kind.name] = kind_options
    return reconstructed
