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
import textwrap
import junitparser
from pathlib import Path
from typing import List, Optional
import os
import urllib.parse
import logging

from cmd_utils import init_log


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def lstrip(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        s = s[len(prefix) :]
    return s


def classname_to_file(classname: str) -> str:
    classname = lstrip(classname, "cython.")
    classname = lstrip(classname, "ctypes.")
    return classname.replace(".", "/") + ".py"


def failed_test_ids() -> List[str]:
    FAILURE_TYPES = (junitparser.Failure, junitparser.Error)
    junit_dir = REPO_ROOT / "build" / "pytest-results"
    failed_node_ids = []
    for junit in junit_dir.glob("*.xml"):
        xml = junitparser.JUnitXml.fromfile(str(junit))
        for suite in xml:
            # handle suites
            for case in suite:
                if case.result is None:
                    logging.warn(f"Incorrectly formatted JUnit found, result was None on {case}")
                    continue

                if len(case.result) > 0 and isinstance(case.result[0], FAILURE_TYPES):
                    node_id = classname_to_file(case.classname) + "::" + case.name
                    failed_node_ids.append(node_id)

    return list(set(failed_node_ids))


def repro_command(build_type: str, failed_node_ids: List[str]) -> Optional[str]:
    """
    Parse available JUnit XML files and output a command that users can run to
    reproduce CI failures locally
    """
    test_args = [f"--tests {node_id}" for node_id in failed_node_ids]
    test_args_str = " ".join(test_args)
    return f"python3 tests/scripts/ci.py {build_type} {test_args_str}"


def make_issue_url(failed_node_ids: List[str]) -> str:
    names = [f"`{node_id}`" for node_id in failed_node_ids]
    run_url = os.getenv("RUN_DISPLAY_URL", "<insert run URL>")
    test_bullets = [f"  - `{node_id}`" for node_id in failed_node_ids]
    params = {
        "labels": "test: flaky",
        "title": "[Flaky Test] " + ", ".join(names),
        "body": textwrap.dedent(
            f"""
            These tests were found to be flaky (intermittently failing on `main` or failed in a PR with unrelated changes). See [the docs](https://github.com/apache/tvm/blob/main/docs/contribute/ci.rst#handling-flaky-failures) for details.

            ### Tests(s)\n
            """
        )
        + "\n".join(test_bullets)
        + f"\n\n### Jenkins Links\n\n  - {run_url}",
    }
    return "https://github.com/apache/tvm/issues/new?" + urllib.parse.urlencode(params)


def show_failure_help(failed_suites: List[str]) -> None:
    failed_node_ids = failed_test_ids()

    if len(failed_node_ids) == 0:
        return

    build_type = os.getenv("PLATFORM")

    if build_type is None:
        raise RuntimeError("build type was None, cannot show command")

    repro = repro_command(build_type=build_type, failed_node_ids=failed_node_ids)
    if repro is None:
        print("No test failures detected")
        return

    print(f"Report flaky test shortcut: {make_issue_url(failed_node_ids)}")
    print("=============================== PYTEST FAILURES ================================")
    print(
        "These pytest suites failed to execute. The results can be found in the "
        "Jenkins 'Tests' tab or by scrolling up through the raw logs here. "
        "If there is no test listed below, the failure likely came from a segmentation "
        "fault which you can find in the logs above.\n"
    )
    if failed_suites is not None and len(failed_suites) > 0:
        print("\n".join([f"    - {suite}" for suite in failed_suites]))
        print("")

    print("You can reproduce these specific failures locally with this command:\n")
    print(textwrap.indent(repro, prefix="    "))
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print information about a failed pytest run")
    args, other = parser.parse_known_args()
    init_log()

    try:
        show_failure_help(failed_suites=other)
    except Exception as e:
        # This script shouldn't ever introduce failures since it's just there to
        # add extra information, so ignore any errors
        logging.exception(e)
