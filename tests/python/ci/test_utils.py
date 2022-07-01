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

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent


class TempGit:
    """
    A wrapper to run commands in a directory
    """

    def __init__(self, cwd):
        self.cwd = cwd

    def run(self, *args, **kwargs):
        proc = subprocess.run(
            ["git"] + list(args), encoding="utf-8", cwd=self.cwd, check=False, **kwargs
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git command failed: '{args}'")

        return proc
