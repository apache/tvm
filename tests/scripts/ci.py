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
import click
import subprocess
from pathlib import Path
from typing import List, Dict, Any

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
    docker_bash = REPO_ROOT / "docker" / "bash.sh"
    command = ["sudo", docker_bash, "--name", name]
    for key, value in env.items():
        command.append("--env")
        command.append(f"{key}={value}")
    command += [image, "bash", "-c", " && ".join(scripts)]

    try:
        cmd(command)
    except RuntimeError as e:
        clean_exit(f"Error invoking Docker: {e}")
    except KeyboardInterrupt:
        cmd(["sudo", "docker", "stop", name])


@click.group()
def cli() -> None:
    """
    Run CI scripts locally via Docker
    """
    pass


@cli.command()
@click.option(
    "--full", default=False, is_flag=True, help="Build all language docs, not just Python"
)
@click.option("--precheck", default=False, is_flag=True, help="Run Sphinx precheck script")
def docs(full: bool, precheck: bool) -> None:
    """
    Build the documentation from gallery/ and docs/
    """
    scripts = []
    if precheck:
        scripts += [
            "./tests/scripts/task_sphinx_precheck.sh",
        ]
    scripts += [
        "./tests/scripts/task_config_build_gpu.sh",
        f"./tests/scripts/task_build.sh build -j{NPROC}",
        "./tests/scripts/task_ci_setup.sh",
        "./tests/scripts/task_python_docs.sh",
    ]
    env = {
        "TVM_TUTORIAL_EXEC_PATTERN": os.getenv(
            "TVM_TUTORIAL_EXEC_PATTERN", ".py" if full else "None"
        ),
        "PYTHON_DOCS_ONLY": "0" if full else "1",
        "IS_LOCAL": "1",
    }
    docker(name="ci-docs", image="tlcpack/ci-gpu:v0.78", scripts=scripts, env=env)


@cli.command()
@click.option(
    "--directory", default="_docs"
)
def serve_docs(directory) -> None:
    """
    Serve the docs using Python's http server
    """
    directory = Path(directory)
    if not directory.exists():
        clean_exit("Docs have not been build, run 'ci.py docs' first")
    cmd([sys.executable, "-m", "http.server"], cwd=directory)


if __name__ == "__main__":
    cli()
