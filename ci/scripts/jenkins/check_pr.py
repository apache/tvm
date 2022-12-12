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
import re
import os
import json
import textwrap
from dataclasses import dataclass
from typing import Any, List, Callable


from git_utils import GitHubRepo, parse_remote, git
from cmd_utils import init_log, tags_from_title


GITHUB_USERNAME_REGEX = re.compile(r"(@[a-zA-Z0-9-]+)", flags=re.MULTILINE)
OK = object()
FAIL = object()


@dataclass
class Check:
    # check to run, returning OK means it passed, anything else means it failed
    check: Callable[[str], Any]

    # function to call to generate the error message
    error_fn: Callable[[Any], str]


def non_empty(s: str):
    if len(s) == 0:
        return FAIL
    return OK


def usernames(s: str):
    m = GITHUB_USERNAME_REGEX.findall(s)
    return m if m else OK


def tags(s: str):
    items = tags_from_title(s)
    if len(items) == 0:
        return FAIL
    return OK


def trailing_period(s: str):
    if s.endswith("."):
        return FAIL
    return OK


title_checks = [
    Check(check=non_empty, error_fn=lambda d: "PR must have a title but title was empty"),
    Check(check=trailing_period, error_fn=lambda d: "PR must not end in a tailing '.'"),
    # TODO(driazati): enable this check once https://github.com/apache/tvm/issues/12637 is done
    # Check(
    #     check=usernames,
    #     error_fn=lambda d: f"PR title must not tag anyone but found these usernames: {d}",
    # ),
]
body_checks = [
    Check(check=non_empty, error_fn=lambda d: "PR must have a body but body was empty"),
    # TODO(driazati): enable this check once https://github.com/apache/tvm/issues/12637 is done
    # Check(
    #     check=usernames,
    #     error_fn=lambda d: f"PR body must not tag anyone but found these usernames: {d}",
    # ),
]


def run_checks(checks: List[Check], s: str, name: str) -> bool:
    print(f"Running checks for {name}")
    print(textwrap.indent(s, prefix="    "))
    passed = True
    print("    Checks:")
    for i, check in enumerate(checks):
        result = check.check(s)
        if result == OK:
            print(f"        [{i+1}] {check.check.__name__}: PASSED")
        else:
            passed = False
            msg = check.error_fn(result)
            print(f"        [{i+1}] {check.check.__name__}: FAILED: {msg}")

    return passed


if __name__ == "__main__":
    init_log()
    help = "Check a PR's title and body for conformance to guidelines"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--pr", required=True)
    parser.add_argument("--remote", default="origin", help="ssh remote to parse")
    parser.add_argument(
        "--pr-data", help="(testing) PR data to use instead of fetching from GitHub"
    )
    args = parser.parse_args()

    try:
        pr = int(args.pr)
    except ValueError:
        print(f"PR was not a number: {args.pr}")
        exit(0)

    if args.pr_data:
        pr = json.loads(args.pr_data)
    else:
        remote = git(["config", "--get", f"remote.{args.remote}.url"])
        user, repo = parse_remote(remote)

        github = GitHubRepo(token=os.environ["GITHUB_TOKEN"], user=user, repo=repo)
        pr = github.get(f"pulls/{args.pr}")

    body = "" if pr["body"] is None else pr["body"].strip()
    title = "" if pr["title"] is None else pr["title"].strip()

    title_passed = run_checks(checks=title_checks, s=title, name="PR title")
    print("")
    body_passed = run_checks(checks=body_checks, s=body, name="PR body")

    if title_passed and body_passed:
        print("All checks passed!")
        exit(0)
    else:
        print(
            "Some checks failed, please review the logs above and edit your PR on GitHub accordingly"
        )
        exit(1)
