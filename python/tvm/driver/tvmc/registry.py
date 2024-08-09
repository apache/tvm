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
This file contains functions for processing registry based inputs for the TVMC CLI
"""

from tvm.driver.tvmc import TVMCException

# We can't tell the type inside an Array but all current options are
# strings so it can default to that. runtime.BoxBool is used to
# distinguish from runtime.BoxInt.
INTERNAL_TO_NATIVE_TYPE = {
    "runtime.String": str,
    "runtime.BoxBool": bool,
    "runtime.BoxFloat": float,
    "runtime.BoxInt": int,
    "Array": str,
}
INTERNAL_TO_HELP = {
    "runtime.String": " string",
    "runtime.BoxBool": " bool",
    "runtime.BoxInt": " int",
    "runtime.BoxFloat": " float",
    "Array": " options",
}


def _generate_registry_option_args(parser, registry, name):
    target_group = parser.add_argument_group(f"{registry.flag_registry_name} {name}")
    for option_name, option_type in registry.list_registered_options(name).items():
        if option_type in INTERNAL_TO_NATIVE_TYPE:
            target_group.add_argument(
                f"--{registry.flag_registry_name}-{name}-{option_name}",
                type=INTERNAL_TO_NATIVE_TYPE[option_type],
                help=(
                    f"{registry.flag_registry_name.title()} "
                    + "{name} {option_name}{INTERNAL_TO_HELP[option_type]}"
                ),
            )


def generate_registry_args(parser, registry, default=None):
    """Walks through the given registry and generates arguments for each of the available options"""
    parser.add_argument(
        f"--{registry.flag_registry_name}",
        help=f"{registry.flag_registry_name.title()} to compile the model with",
        required=False,
        default=default,
    )
    names = registry.list_registered()

    for name in names:
        _generate_registry_option_args(parser, registry, name)


def _reconstruct_registry_options(args, registry, name):
    options = {}
    for option, option_type in registry.list_registered_options(name).items():
        if option_type in INTERNAL_TO_NATIVE_TYPE:
            var_name = f"{registry.flag_registry_name}_{name}_{option.replace('-', '_')}"
            option_value = getattr(args, var_name)
            if option_value is not None:
                options[option] = option_value
    return options


def reconstruct_registry_entity(args, registry):
    """Reconstructs an entity from arguments generated from a registry"""
    possible_names = registry.list_registered()
    name = getattr(args, registry.flag_registry_name)
    if name is None:
        return None

    if name not in possible_names:
        raise TVMCException(f'{registry.flag_registry_name.title()} "{name}" is not defined')

    reconstructed = {
        possible_name: _reconstruct_registry_options(args, registry, possible_name)
        for possible_name in possible_names
    }

    for possible_name in possible_names:
        if possible_name != name and reconstructed[possible_name]:
            first_option = list(reconstructed[possible_name])[0]
            raise TVMCException(
                f"Passed --{registry.flag_registry_name}-{possible_name}-{first_option} "
                f"but did not specify {possible_name} executor"
            )

    return registry(name, reconstructed[name])
