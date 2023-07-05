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
import sys

LINK_BASE = "https://github.com/apache/tvm-rfcs/blob/main/"
COMMIT_BASE = "https://github.com/apache/tvm-rfcs/commit/"


def sprint(*args):
    print(*args, file=sys.stderr)


if __name__ == "__main__":
    help = "List out RFCs since a commit"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--since-commit", required=True, help="last commit to include")
    parser.add_argument("--rfcs-repo", required=True, help="path to checkout of apache/tvm-rfcs")
    args = parser.parse_args()
    user = "apache"
    repo = "tvm"
    rfc_repo = args.rfcs_repo
    subprocess.run("git fetch origin main", cwd=rfc_repo, shell=True)
    subprocess.run("git checkout main", cwd=rfc_repo, shell=True)
    subprocess.run("git reset --hard origin/main", cwd=rfc_repo, shell=True)
    r = subprocess.run(
        f"git log {args.since_commit}..HEAD --format='%H %s'",
        cwd=rfc_repo,
        shell=True,
        stdout=subprocess.PIPE,
        encoding="utf-8",
    )
    commits = r.stdout.strip().split("\n")
    for commit in commits:
        parts = commit.split()
        commit = parts[0]
        subject = " ".join(parts[1:])

        r2 = subprocess.run(
            f"git diff-tree --no-commit-id --name-only -r {commit}",
            cwd=rfc_repo,
            shell=True,
            stdout=subprocess.PIPE,
            encoding="utf-8",
        )
        files = r2.stdout.strip().split("\n")
        rfc_file = None
        for file in files:
            if file.startswith("rfcs/") and file.endswith(".md"):
                if rfc_file is not None:
                    sprint(f"error on {commit} {subject}")
                rfc_file = file

        if rfc_file is None:
            sprint(f"error on {commit} {subject}")
            continue

        print(f" * [{subject}]({LINK_BASE + rfc_file}) ([`{commit[:7]}`]({COMMIT_BASE + commit}))")
