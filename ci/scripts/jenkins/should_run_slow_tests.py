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
import argparse
import textwrap
from typing import Tuple, List, Optional


from git_utils import GitHubRepo, parse_remote, git


SLOW_TEST_TRIGGERS = [
    "@tvm-bot run slow tests",
    "@tvm-bot run slow test",
    "@tvm-bot run slow",
    "@tvm-bot slow tests",
    "@tvm-bot slow test",
    "@tvm-bot slow",
]


def check_match(s: str, searches: List[str]) -> Tuple[bool, Optional[str]]:
    for search in searches:
        if search in s:
            return True, search

    return False, None


def display(long_str: str) -> str:
    return textwrap.indent(long_str, "    ")


if __name__ == "__main__":
    help = "Exits with 1 if CI should run slow tests, 0 otherwise"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--pr", required=True)
    parser.add_argument("--remote", default="origin", help="ssh remote to parse")
    parser.add_argument(
        "--pr-body", help="(testing) PR body to use instead of fetching from GitHub"
    )
    args = parser.parse_args()

    branch = git(["rev-parse", "--abbrev-ref", "HEAD"])

    # Don't skip slow tests on main or ci-docker-staging
    skip_branches = {"main", "ci-docker-staging"}
    if branch in skip_branches:
        print(f"Branch {branch} is in {skip_branches}, running slow tests")
        exit(1)
    print(f"Branch {branch} is not in {skip_branches}, checking last commit...")

    if args.pr_body:
        body = args.pr_body
    else:
        remote = git(["config", "--get", f"remote.{args.remote}.url"])
        user, repo = parse_remote(remote)

        github = GitHubRepo(token=os.environ["GITHUB_TOKEN"], user=user, repo=repo)
        pr = github.get(f"pulls/{args.pr}")
        body = pr["body"]

    body_match, reason = check_match(body, SLOW_TEST_TRIGGERS)

    if body_match:
        print(f"Matched {reason} in PR body:\n{display(body)}, running slow tests")
        exit(1)

    print(
        f"PR Body:\n{display(body)}\ndid not have any of {SLOW_TEST_TRIGGERS}, skipping slow tests"
    )
    exit(0)
