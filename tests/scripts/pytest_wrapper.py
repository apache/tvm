#!/usr/bin/env python3
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
import argparse
import subprocess
import textwrap
import junitparser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def lstrip(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        s = s[len(prefix) :]
    return s


def classname_to_file(classname: str) -> str:
    classname = lstrip(classname, "cython.")
    classname = lstrip(classname, "ctypes.")
    return classname.replace(".", "/") + ".py"


def repro_command():
    """
    Parse available JUnit XML files and output a command that users can run to
    reproduce CI failures locally
    """
    build_type = "<unknown>"
    junit_dir = REPO_ROOT / "build" / "pytest-results"
    failed_node_ids = []
    for junit in junit_dir.glob("*.xml"):
        xml = junitparser.JUnitXml.fromfile(str(junit))
        for suite in xml:
            build_type = lstrip(suite.hostname, "ci-")
            # handle suites
            for case in suite:
                if len(case.result) > 0:
                    node_id = classname_to_file(case.classname) + "::" + case.name
                    failed_node_ids.append(node_id)

    return f"python3 tests/scripts/ci.py {build_type} --tests {' '.join(failed_node_ids)}"


def show_failure_help():
    print("================== Detected Pytest Failures ==================")
    print("You can reproduce these specific failures locally with this command:\n")
    print(textwrap.indent(repro_command(), prefix="    "))
    print("")
    print(
        "If you believe these test failures are spurious or are not due to this change, please file an issue: https://github.com/apache/tvm/issues/new?assignees=&labels=test%3A+flaky&template=flaky-test.md&title=%5BFlaky+Test%5D+"
    )


def run(script, remaining_args):
    proc = subprocess.run([script] + remaining_args)
    if proc.returncode != 0:
        show_failure_help()
        exit(proc.returncode)
    else:
        exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wrap TVM CI scripts and output helpful information on failures"
    )
    parser.add_argument("script", help="script to run")
    args, other = parser.parse_known_args()
    print(args, other)
