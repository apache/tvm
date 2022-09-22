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
import datetime
import re
import textwrap

from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
JENKINSFILE_TEMPLATE = REPO_ROOT / "ci" / "jenkins" / "Jenkinsfile.j2"
JENKINSFILE = REPO_ROOT / "Jenkinsfile"


class Change:
    IMAGES_ONLY = object()
    NONE = object()
    FULL = object()


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


def lines_without_generated_tag(content):
    return [
        line for line in content.splitlines(keepends=True) if not line.startswith("// Generated at")
    ]


def change_type(lines: List[str]) -> Change:
    """
    Return True if 'line' only edits an image tag or if 'line' is not a changed
    line in a diff
    """
    added_images = []
    removed_images = []
    diff_lines = []

    for line in lines[2:]:
        if not line.startswith("-") and not line.startswith("+"):
            # not a diff line, ignore it
            continue

        diff_lines.append(line)

    if len(diff_lines) == 0:
        # no changes made
        return Change.NONE

    for line in diff_lines:
        is_add = line.startswith("+")
        line = line.strip().lstrip("+").lstrip("-")
        match = re.search(
            r"^(ci_[a-zA-Z0-9]+) = \'.*\'$",
            line.strip().lstrip("+").lstrip("-"),
            flags=re.MULTILINE,
        )
        if match is None:
            # matched a non-image line, quit early
            return Change.FULL

        if is_add:
            added_images.append(match.groups()[0])
        else:
            removed_images.append(match.groups()[0])

    # make sure that the added image lines match the removed image lines
    if len(added_images) > 0 and added_images == removed_images:
        return Change.IMAGES_ONLY
    else:
        return Change.FULL


if __name__ == "__main__":
    help = "Regenerate Jenkinsfile from template"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--force", action="store_true", help="always overwrite timestamp")
    parser.add_argument("--check", action="store_true", help="just verify the output didn't change")
    args = parser.parse_args()

    with open(JENKINSFILE) as f:
        content = f.read()

    data["generated_time"] = datetime.datetime.now().isoformat()
    timestamp_match = re.search(r"^// Generated at (.*)$", content, flags=re.MULTILINE)
    if not timestamp_match:
        raise RuntimeError("Could not find timestamp in Jenkinsfile")
    original_timestamp = timestamp_match.groups()[0]

    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(REPO_ROOT),
        undefined=jinja2.StrictUndefined,
        lstrip_blocks=True,
        trim_blocks=True,
        keep_trailing_newline=True,
    )
    template = environment.get_template(str(JENKINSFILE_TEMPLATE.relative_to(REPO_ROOT)))
    new_content = template.render(**data)

    diff = [
        line
        for line in difflib.unified_diff(
            lines_without_generated_tag(content), lines_without_generated_tag(new_content)
        )
    ]
    change = change_type(diff)
    if not args.force and change == Change.IMAGES_ONLY or change == Change.NONE:
        if change != Change.NONE:
            print("Detected only Docker-image name changes, skipping timestamp update")
        new_content = new_content.replace(data["generated_time"], original_timestamp)

    diff = "".join(diff)

    if args.check:
        if not diff:
            print("Success, the newly generated Jenkinsfile matched the one on disk")
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
