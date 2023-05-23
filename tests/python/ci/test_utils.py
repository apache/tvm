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
Constants used in various CI tests
"""
import subprocess
import pathlib
from typing import List, Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
GITHUB_SCRIPT_ROOT = REPO_ROOT / "ci" / "scripts" / "github"
JENKINS_SCRIPT_ROOT = REPO_ROOT / "ci" / "scripts" / "jenkins"


class TempGit:
    """
    A wrapper to run commands in a directory (specifically for use in CI tests)
    """

    def __init__(self, cwd):
        self.cwd = cwd
        # Jenkins git is too old and doesn't have 'git init --initial-branch',
        # so init and checkout need to be separate steps
        self.run("init", stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        self.run("checkout", "-b", "main", stderr=subprocess.PIPE)
        self.run("remote", "add", "origin", "https://github.com/apache/tvm.git")

    def run(self, *args, **kwargs):
        """
        Run a git command based on *args
        """
        proc = subprocess.run(
            ["git"] + list(args), encoding="utf-8", cwd=self.cwd, check=False, **kwargs
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git command failed: '{args}'")

        return proc


def run_script(command: List[Any], check: bool = True, **kwargs):
    """
    Wrapper to run a script and print its output if there was an error
    """
    command = [str(c) for c in command]
    kwargs_to_send = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "encoding": "utf-8",
    }
    kwargs_to_send.update(kwargs)
    proc = subprocess.run(
        command,
        check=False,
        **kwargs_to_send,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"Process failed:\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}")

    return proc
