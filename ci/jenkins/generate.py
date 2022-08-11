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
import jinja2
import argparse
import difflib
import fnmatch
import datetime
import re
import textwrap
import subprocess
from typing import List

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
JENKINSFILE_TEMPLATE = REPO_ROOT / "ci" / "jenkins" / "Jenkinsfile.j2"
JENKINSFILE = REPO_ROOT / "Jenkinsfile"


data = {
    "images": [
        {
            "name": "ci_arm",
            "platform": "ARM",
        },
        {
            "name": "ci_cortexm",
            "platform": "CPU",
        },
        {
            "name": "ci_cpu",
            "platform": "CPU",
        },
        {
            "name": "ci_gpu",
            "platform": "CPU",
        },
        {
            "name": "ci_hexagon",
            "platform": "CPU",
        },
        {
            "name": "ci_i386",
            "platform": "CPU",
        },
        {
            "name": "ci_lint",
            "platform": "CPU",
        },
        {
            "name": "ci_minimal",
            "platform": "CPU",
        },
        {
            "name": "ci_riscv",
            "platform": "CPU",
        },
        {
            "name": "ci_wasm",
            "platform": "CPU",
        },
    ]
}


def lines_without_generated_tag(content: str) -> List[str]:
    return [
        line for line in content.splitlines(keepends=True) if not line.startswith("// Generated at")
    ]


def changed_files(base_ref: str) -> List[str]:
    proc = subprocess.run(
        ["git", "diff", base_ref, "HEAD", "--name-only"],
        check=True,
        stdout=subprocess.PIPE,
        encoding="utf-8",
    )
    files = [f.strip() for f in proc.stdout.strip().split("\n")]
    files = [f for f in files if f != ""]
    return files


def check_timestamp(base_ref: str) -> None:
    """
    Assert that the git diff against main contains a timestamp
    """
    files = changed_files(base_ref=base_ref)

    # Check the Jenkinsfile timestamp if the templates were edited
    if any(fnmatch.fnmatch(f, "ci/jenkins/*.j2") for f in files):
        proc = subprocess.run(
            ["git", "diff", base_ref, "HEAD", "Jenkinsfile"],
            check=True,
            stdout=subprocess.PIPE,
            encoding="utf-8",
        )
        diff = proc.stdout
        if re.search(r"^\+// Generated at", diff, flags=re.MULTILINE) is None:
            print(
                "Newly generated Jenkinsfile was missing an updated timestamp, "
                "please ensure that the timestamp is updated by running "
                "'python3 ci/jenkins/generate.py' and commit your changes "
                "to avoid merge conflicts"
            )
            exit(1)


if __name__ == "__main__":
    help = "Regenerate Jenkinsfile from template"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument(
        "--base-ref", help="git ref to diff against for checking", default="origin/main"
    )
    parser.add_argument("--check", action="store_true", help="just verify the output didn't change")
    args = parser.parse_args()

    with open(JENKINSFILE) as f:
        content = f.read()

    data["generated_time"] = datetime.datetime.now().isoformat()

    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(REPO_ROOT),
        undefined=jinja2.StrictUndefined,
        lstrip_blocks=True,
        trim_blocks=True,
        keep_trailing_newline=True,
    )
    template = environment.get_template(str(JENKINSFILE_TEMPLATE.relative_to(REPO_ROOT)))
    new_content = template.render(**data)

    diff = "".join(
        difflib.unified_diff(
            lines_without_generated_tag(content), lines_without_generated_tag(new_content)
        )
    )
    if args.check:
        check_timestamp(base_ref=args.base_ref)
        if not diff:
            print(
                f"Success, the newly generated Jenkinsfile matched the one on disk at {JENKINSFILE}"
            )
            exit(0)
        else:
            print(
                textwrap.dedent(
                    """
                Newly generated Jenkinsfile did not match the one on disk! If you have made
                edits to the Jenkinsfile, move them to 'jenkins/Jenkinsfile.j2' and
                regenerate the Jenkinsfile from the template with

                    python3 -m pip install -r jenkins/requirements.txt
                    python3 jenkins/generate.py

                Diffed changes:
            """
                ).strip()
            )
            print(diff)
            exit(1)
    else:
        with open(JENKINSFILE, "w") as f:
            f.write(new_content)
        if not diff:
            print(f"Wrote output to {JENKINSFILE.relative_to(REPO_ROOT)}, no changes made")
        else:
            print(f"Wrote output to {JENKINSFILE.relative_to(REPO_ROOT)}, changes:")
            print(diff)
