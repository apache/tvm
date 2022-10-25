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

from check_pr import get_pr_title_body
from cmd_utils import init_log

HEXAGON_TEST_AUTHORIZED_USERS = ["mehrdadh"]


def check_pr_title(pr_number: int) -> bool:
    pr_title, _ = get_pr_title_body(pr_number)
    if "[hexagon]" in pr_title.lower():
        return True
    return False


def trigger_hexagon_ci_if_required(pr_number: int, author: str):
    """Trigger Hexagon hardware CI for certain PRs."""
    if not check_pr_title(pr_number):
        print("Skip Hexagon CI: Hexagon not found in the PR title.")
        return

    if author not in HEXAGON_TEST_AUTHORIZED_USERS:
        print("Skip Hexagon CI: User is not authorized.")
        return

    print("triggering...")


if __name__ == "__main__":
    init_log()
    help = "Check if a PR is required testing on Hexagon devices."
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--pr", required=True)
    parser.add_argument("--author", required=True, help="Pull request author github username.")
    args = parser.parse_args()

    try:
        pr_number = int(args.pr)
    except ValueError:
        print(f"PR was not a number: {args.pr}")
        exit(0)

    trigger_hexagon_ci_if_required(pr_number, args.author)
