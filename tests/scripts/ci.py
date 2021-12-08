#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import sys
import multiprocessing
import os
import getpass
import inspect
import argparse
import grp
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NPROC = multiprocessing.cpu_count()


class col:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_color(color: str, msg: str, **kwargs: Any) -> None:
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        print(col.BOLD + color + msg + col.RESET, **kwargs)
    else:
        print(msg, **kwargs)


def clean_exit(msg: str) -> None:
    print_color(col.RED, msg, file=sys.stderr)
    exit(1)


def cmd(commands: List[Any], **kwargs: Any):
    commands = [str(s) for s in commands]
    command_str = " ".join(commands)
    print_color(col.BLUE, command_str)
    proc = subprocess.run(commands, **kwargs)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: '{command_str}'")


def docker(name: str, image: str, scripts: List[str], env: Dict[str, str]):
    """
    Invoke a set of bash scripts through docker/bash.sh
    """
    if sys.platform == "linux":
        # Check that the user is in the docker group before running
        try:
            group = grp.getgrnam("docker")
            if getpass.getuser() not in group.gr_mem:
                print_color(
                    col.YELLOW, f"Note: User '{getpass.getuser()}' is not in the 'docker' group"
                )
        except KeyError:
            print_color(col.YELLOW, f"Note: 'docker' group does not exist")

    docker_bash = REPO_ROOT / "docker" / "bash.sh"
    command = [docker_bash, "--name", name]
    for key, value in env.items():
        command.append("--env")
        command.append(f"{key}={value}")
    command += [image, "bash", "-c", " && ".join(scripts)]

    try:
        cmd(command)
    except RuntimeError as e:
        clean_exit(f"Error invoking Docker: {e}")
    except KeyboardInterrupt:
        cmd(["docker", "stop", "--time", "1", name])


def docs(
    tutorial_pattern: Optional[str] = None,
    full: bool = False,
    precheck: bool = False,
    cpu: bool = False,
) -> None:
    """
    Build the documentation from gallery/ and docs/. By default this builds only
    the Python docs. If you are on a CPU machine, you can skip the tutorials
    and build the docs with the '--precheck --cpu' options.

    arguments:
    full -- Build all language docs, not just Python
    precheck -- Run Sphinx precheck script
    tutorial-pattern -- Regex for which tutorials to execute when building docs (can also be set via TVM_TUTORIAL_EXEC_PATTERN)
    cpu -- Use CMake defaults for building TVM (useful for building docs on a CPU machine.)
    """
    config = "./tests/scripts/task_config_build_gpu.sh"
    if cpu and full:
        clean_exit("--full cannot be used with --cpu")

    if cpu:
        # The docs import tvm.micro, so it has to be enabled in the build
        config = "cd build && cp ../cmake/config.cmake . && echo set\(USE_MICRO ON\) >> config.cmake && cd .."

    scripts = [
        config,
        f"./tests/scripts/task_build.sh build -j{NPROC}",
        "./tests/scripts/task_ci_setup.sh",
        "./tests/scripts/task_sphinx_precheck.sh"
        if precheck
        else "./tests/scripts/task_python_docs.sh",
    ]

    if tutorial_pattern is None:
        tutorial_pattern = os.getenv("TVM_TUTORIAL_EXEC_PATTERN", ".py" if full else "none")

    env = {
        "TVM_TUTORIAL_EXEC_PATTERN": tutorial_pattern,
        "PYTHON_DOCS_ONLY": "0" if full else "1",
        "IS_LOCAL": "1",
    }
    docker(name="ci-docs", image="ci_gpu", scripts=scripts, env=env)


def serve_docs(directory: str = "_docs") -> None:
    """
    Serve the docs using Python's http server

    arguments:
    directory -- Directory to serve from
    """
    directory = Path(directory)
    if not directory.exists():
        clean_exit("Docs have not been build, run 'ci.py docs' first")
    cmd([sys.executable, "-m", "http.server"], cwd=directory)


def lint() -> None:
    """
    Run CI's Sanity Check step
    """
    docker(
        name="ci-lint",
        image="ci_lint",
        scripts=["./tests/scripts/task_lint.sh"],
        env={},
    )


def cli_name(s: str) -> str:
    return s.replace("_", "-")


def add_subparser(func, subparsers) -> Any:
    """
    Utility function to make it so subparser commands can be defined locally
    as a function rather than directly via argparse and manually dispatched
    out.
    """

    # Each function is intended follow the example for arguments in PEP257, so
    # split apart the function documentation from the arguments
    split = [s.strip() for s in func.__doc__.split("arguments:\n")]
    if len(split) == 1:
        args_help = None
        command_help = split[0]
    else:
        command_help, args_help = split

    # Parse out the help text for each argument if present
    arg_help_texts = {}
    if args_help is not None:
        for line in args_help.split("\n"):
            line = line.strip()
            name, help_text = [t.strip() for t in line.split(" -- ")]
            arg_help_texts[name] = help_text

    subparser = subparsers.add_parser(cli_name(func.__name__), help=command_help)

    # Add each parameter to the subparser
    signature = inspect.signature(func)
    for name, value in signature.parameters.items():
        kwargs = {"help": arg_help_texts[cli_name(name)]}

        # Grab the default value if present
        if value.default is not value.empty:
            kwargs["default"] = value.default

        # Check if it should be a flag
        if value.annotation is bool:
            kwargs["action"] = "store_true"
        subparser.add_argument(f"--{cli_name(name)}", **kwargs)

    return subparser


def main():
    parser = argparse.ArgumentParser(description="Run CI scripts locally via Docker")
    subparsers = parser.add_subparsers(dest="command")

    subparser_functions = {cli_name(func.__name__): func for func in [docs, serve_docs, lint]}
    for func in subparser_functions.values():
        add_subparser(func, subparsers)

    args = parser.parse_args()
    func = subparser_functions[args.command]

    # Extract out the parsed args and invoke the relevant function
    kwargs = {k: getattr(args, k) for k in dir(args) if not k.startswith("_") and k != "command"}
    func(**kwargs)


if __name__ == "__main__":
    main()
