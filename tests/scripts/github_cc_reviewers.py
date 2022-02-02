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

import os
import json
import argparse
import re
from urllib import error
from typing import Dict, Any, List


from git_utils import git, GitHubRepo, parse_remote


def find_reviewers(body: str) -> List[str]:
    print(f"Parsing body:\n{body}")
    matches = re.findall(r"(cc( @[-A-Za-z0-9]+)+)", body, flags=re.MULTILINE)
    matches = [full for full, last in matches]

    print("Found matches:", matches)
    reviewers = []
    for match in matches:
        if match.startswith("cc "):
            match = match.replace("cc ", "")
        users = [x.strip() for x in match.split("@")]
        reviewers += users

    reviewers = set(x for x in reviewers if x != "")
    return sorted(list(reviewers))


if __name__ == "__main__":
    help = "Add @cc'ed people in a PR body as reviewers"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--remote", default="origin", help="ssh remote to parse")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="run but don't send any request to GitHub",
    )
    args = parser.parse_args()

    remote = git(["config", "--get", f"remote.{args.remote}.url"])
    user, repo = parse_remote(remote)

    pr = json.loads(os.environ["PR"])

    number = pr["number"]
    body = pr["body"]
    if body is None:
        body = ""

    to_add = find_reviewers(body)
    print("Adding reviewers:", to_add)

    if not args.dry_run:
        github = GitHubRepo(token=os.environ["GITHUB_TOKEN"], user=user, repo=repo)

        # Add reviewers 1 by 1 since GitHub will error out if any of the
        # requested reviewers aren't members / contributors
        for reviewer in to_add:
            try:
                github.post(f"pulls/{number}/requested_reviewers", {"reviewers": [reviewer]})
            except error.HTTPError as e:
                print(f"Failed to add reviewer {reviewer}: {e}")
