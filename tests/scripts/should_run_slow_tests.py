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
import subprocess
import re
import textwrap
from urllib import request
from typing import Dict, Tuple, Any, List, Optional

SLOW_TEST_TRIGGERS = [
    "@ci run slow tests",
    "@ci run slow test",
    "@ci run slow",
    "@ci slow tests",
    "@ci slow test",
    "@ci slow",
]


class GitHubRepo:
    def __init__(self, user, repo, token):
        self.token = token
        self.user = user
        self.repo = repo
        self.base = f"https://api.github.com/repos/{user}/{repo}/"

    def headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
        }

    def get(self, url: str) -> Dict[str, Any]:
        url = self.base + url
        print("Requesting", url)
        req = request.Request(url, headers=self.headers())
        with request.urlopen(req) as response:
            response = json.loads(response.read())
        return response


def parse_remote(remote: str) -> Tuple[str, str]:
    """
    Get a GitHub (user, repo) pair out of a git remote
    """
    if remote.startswith("https://"):
        # Parse HTTP remote
        parts = remote.split("/")
        if len(parts) < 2:
            raise RuntimeError(f"Unable to parse remote '{remote}'")
        return parts[-2], parts[-1].replace(".git", "")
    else:
        # Parse SSH remote
        m = re.search(r":(.*)/(.*)\.git", remote)
        if m is None or len(m.groups()) != 2:
            raise RuntimeError(f"Unable to parse remote '{remote}'")
        return m.groups()


def check_match(s: str, searches: List[str]) -> Tuple[bool, Optional[str]]:
    for search in searches:
        if search in s:
            return True, search

    return False, None


def git(command):
    proc = subprocess.run(["git"] + command, stdout=subprocess.PIPE, check=True)
    return proc.stdout.decode().strip()


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

    log = git(["log", "--format=%B", "-1"])

    # Check if anything in the last commit's body message matches
    log_match, reason = check_match(log, SLOW_TEST_TRIGGERS)
    if log_match:
        print(f"Matched {reason} in commit message:\n{display(log)}, running slow tests")
        exit(1)

    print(
        f"Last commit:\n{display(log)}\ndid not have any of {SLOW_TEST_TRIGGERS}, checking PR body..."
    )

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
