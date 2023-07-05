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
from typing import List, Optional
from dataclasses import dataclass

from data import data


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
JENKINS_DIR = REPO_ROOT / "ci" / "jenkins"
TEMPLATES_DIR = JENKINS_DIR / "templates"
GENERATED_DIR = JENKINS_DIR / "generated"


class Change:
    IMAGES_ONLY = object()
    NONE = object()
    FULL = object()


@dataclass
class ChangeData:
    diff: Optional[str]
    content: str
    destination: Path
    source: Path


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


def update_jenkinsfile(source: Path) -> ChangeData:
    destination = GENERATED_DIR / source.stem

    data["generated_time"] = datetime.datetime.now().isoformat()
    if destination.exists():
        with open(destination) as f:
            old_generated_content = f.read()

        timestamp_match = re.search(
            r"^// Generated at (.*)$", old_generated_content, flags=re.MULTILINE
        )
        if not timestamp_match:
            raise RuntimeError(
                f"Could not find timestamp in Jenkinsfile: {destination.relative_to(TEMPLATES_DIR)}"
            )
        original_timestamp = timestamp_match.groups()[0]

    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        undefined=jinja2.StrictUndefined,
        lstrip_blocks=True,
        trim_blocks=True,
        keep_trailing_newline=True,
    )
    template = environment.get_template(str(source.relative_to(TEMPLATES_DIR)))
    new_content = template.render(**data)

    if not destination.exists():
        # New file, create it from scratch
        return ChangeData(
            diff=new_content, content=new_content, source=source, destination=destination
        )

    diff = [
        line
        for line in difflib.unified_diff(
            lines_without_generated_tag(old_generated_content),
            lines_without_generated_tag(new_content),
        )
    ]
    change = change_type(diff)
    if not args.force and change == Change.IMAGES_ONLY or change == Change.NONE:
        if change != Change.NONE:
            print("Detected only Docker-image name changes, skipping timestamp update")
        new_content = new_content.replace(data["generated_time"], original_timestamp)

    diff = "".join(diff)

    return ChangeData(diff=diff, content=new_content, source=source, destination=destination)


if __name__ == "__main__":
    help = "Regenerate Jenkinsfile from template"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--force", action="store_true", help="always overwrite timestamp")
    parser.add_argument("--check", action="store_true", help="just verify the output didn't change")
    args = parser.parse_args()

    sources = TEMPLATES_DIR.glob("*_jenkinsfile.groovy.j2")
    changes = [update_jenkinsfile(source) for source in sources if source.name != "base.groovy.j2"]

    if args.check:
        if all(not data.diff for data in changes):
            print("Success, the newly generated Jenkinsfiles matched the ones on disk")
            exit(0)
        else:
            print(
                textwrap.dedent(
                    """
                Newly generated Jenkinsfiles did not match the ones on disk! If you have made
                edits to the Jenkinsfiles in generated/, move them to the corresponding source and
                regenerate the Jenkinsfiles from the templates with

                    python3 -m pip install -r jenkins/requirements.txt
                    python3 jenkins/generate.py

                Diffed changes:
            """
                ).strip()
            )
            for data in changes:
                if data.diff:
                    source = data.source.relative_to(REPO_ROOT)
                    print(source)
                    print(data.diff)

            exit(1)
    else:
        for data in changes:
            with open(data.destination, "w") as f:
                f.write(data.content)

            if not data.diff:
                print(f"Wrote output to {data.destination.relative_to(REPO_ROOT)}, no changes made")
            else:
                print(f"Wrote output to {data.destination.relative_to(REPO_ROOT)}, changes:")
                print(data.diff)
