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
from urllib import error
from typing import Any

from git_utils import git, GitHubRepo, parse_remote


_pr_query_fields = """
    number
    reviewDecision
    commits(last:1) {
        nodes {
            commit {
                statusCheckRollup {
                    contexts(last:100) {
                        nodes {
                            ... on CheckRun {
                                conclusion
                                status
                                name
                                checkSuite {
                                    workflowRun {
                                        workflow {
                                            name
                                        }
                                    }
                                }
                            }
                            ... on StatusContext {
                                context
                                state
                            }
                        }
                    }
                }
            }
        }
    }
"""


def prs_query(user: str, repo: str, cursor: str = None):
    after = ""
    if cursor is not None:
        after = f', before:"{cursor}"'
    return f"""
        {{
    repository(name: "{repo}", owner: "{user}") {{
        pullRequests(states: [OPEN], last: 10{after}) {{
        edges {{
            cursor
        }}
        nodes {{
            {_pr_query_fields}
        }}
        }}
    }}
    }}
    """


def is_pr_ready(data: Any) -> bool:
    """
    Returns true if a PR is approved and all of its statuses are SUCCESS
    """
    approved = data["reviewDecision"] == "APPROVED"
    print("Is approved?", approved)

    commit = data["commits"]["nodes"][0]["commit"]

    if commit["statusCheckRollup"] is None:
        # No statuses, not mergeable
        return False

    statuses = commit["statusCheckRollup"]["contexts"]["nodes"]
    unified_statuses = []
    for status in statuses:
        if "context" in status:
            # Parse non-GHA status
            unified_statuses.append((status["context"], status["state"] == "SUCCESS"))
        else:
            # Parse GitHub Actions item
            workflow = status["checkSuite"]["workflowRun"]["workflow"]["name"]
            name = f"{workflow} / {status['name']}"
            unified_statuses.append((name, status["conclusion"] == "SUCCESS"))

    print("Got statuses:", json.dumps(unified_statuses, indent=2))
    passed_ci = all(status for name, status in unified_statuses)
    return approved and passed_ci


if __name__ == "__main__":
    help = "Adds label to PRs that have passed CI and are approved"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--remote", default="origin", help="ssh remote to parse")
    parser.add_argument("--dry-run", action="store_true", help="don't submit to GitHub")
    parser.add_argument("--label", default="ready-for-merge", help="label to add")
    parser.add_argument(
        "--pr-json", help="(testing) PR data to use instead of fetching from GitHub"
    )
    args = parser.parse_args()

    remote = git(["config", "--get", f"remote.{args.remote}.url"])
    user, repo = parse_remote(remote)

    if args.pr_json:
        pr = json.loads(args.pr_json)

    if args.pr_json:
        r = json.loads(args.pr_json)
    else:
        github = GitHubRepo(token=os.environ["GITHUB_TOKEN"], user=user, repo=repo)
        q = prs_query(user, repo)
        r = github.graphql(q)

    # Loop until all PRs have been checked
    while True:
        prs = r["data"]["repository"]["pullRequests"]["nodes"]

        for pr in prs:
            print(f"Checking PR {pr['number']}")
            if is_pr_ready(pr):
                print(f"PR {pr['number']} passed CI and is approved, labelling...")
                if not args.dry_run:
                    github.post(f"issues/{pr['number']}/labels", {"labels": [args.label]})
            else:
                print(f"PR {pr['number']} is not ready for merge")
                if not args.dry_run:
                    try:
                        github.delete(f"issues/{pr['number']}/labels/{args.label}")
                    except error.HTTPError as e:
                        print(e)
                        print("Failed to remove label (it may not have been there at all)")

        edges = r["data"]["repository"]["pullRequests"]["edges"]
        if len(edges) == 0:
            # No more results to check
            break

        cursor = edges[0]["cursor"]
        r = github.graphql(prs_query(user, repo, cursor))
