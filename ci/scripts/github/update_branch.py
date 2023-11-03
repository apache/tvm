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
import sys
from pathlib import Path
from typing import Any, Dict

# Hackery to enable importing of utils from ci/scripts/jenkins
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(REPO_ROOT / "ci" / "scripts" / "jenkins"))

from git_utils import git, GitHubRepo, parse_remote


_commit_query_fields = """
    messageHeadline
    oid
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
"""


def commits_query(user: str, repo: str, cursor: str = None):
    """
    Create a GraphQL query to find the last N commits along with their statuses
    and some metadata (paginated after 'cursor')
    """
    after = ""
    if cursor is not None:
        after = f', after:"{cursor}"'

    return f"""
    {{
    repository(name: "{repo}", owner: "{user}") {{
        defaultBranchRef {{
        target {{
            ... on Commit {{
            history(first: 15{after}) {{
                edges {{ cursor }}
                nodes {{
                    {_commit_query_fields}
                }}
            }}
            }}
        }}
        }}
    }}
    }}
    """


EXPECTED_CI_JOBS = [
    "cross-isa-minimal/branch",
    "gpu/branch",
    "hexagon/branch",
    "arm/branch",
    "cortexm/branch",
    "cpu/branch",
    "docker/branch",
    "i386/branch",
    "lint/branch",
    "minimal/branch",
    "riscv/branch",
    "wasm/branch",
]


def commit_passed_ci(commit: Dict[str, Any]) -> bool:
    """
    Returns true if all of a commit's statuses are SUCCESS
    """
    statuses = commit["statusCheckRollup"]["contexts"]["nodes"]

    # GitHub Actions statuses are different from external GitHub statuses, so
    # unify them into 1 representation
    # https://docs.github.com/en/developers/webhooks-and-events/webhooks/webhook-events-and-payloads
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

    print(f"Statuses on {commit['oid']}:", json.dumps(unified_statuses, indent=2))

    # Assert that specific jobs are present in the commit statuses (i.e. don't
    # approve if CI was broken and didn't schedule a job)
    job_names = {name for name, status in unified_statuses}
    for job in EXPECTED_CI_JOBS:
        if job not in job_names:
            # Did not find expected job name
            return False

    passed_ci = all(status for name, status in unified_statuses)
    return passed_ci


def update_branch(user: str, repo: str, sha: str, branch_name: str) -> None:
    git(["fetch", "origin", sha])
    git(["reset", "--hard", "FETCH_HEAD"])
    try:
        git(["branch", "-D", branch_name])
    except RuntimeError:
        # Ignore failures (i.e. the branch did not exist in the first place)
        pass
    git(["checkout", "-b", branch_name])

    # Create and push the branch
    git(["push", "origin", "--force", branch_name])
    print(f"Pushed branch {branch_name} with commit {sha}")


if __name__ == "__main__":
    help = "Push the a branch to the last commit that passed all CI runs"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--remote", default="origin", help="ssh remote to parse")
    parser.add_argument("--dry-run", action="store_true", help="don't submit to GitHub")
    parser.add_argument("--branch", default="last-successful", help="branch name")
    parser.add_argument(
        "--testonly-json", help="(testing) data to use instead of fetching from GitHub"
    )
    args = parser.parse_args()

    remote = git(["config", "--get", f"remote.{args.remote}.url"])
    user, repo = parse_remote(remote)
    # TODO: Remove this before landing
    user, repo = ("apache", "tvm")

    if args.testonly_json:
        r = json.loads(args.testonly_json)
    else:
        github = GitHubRepo(token=os.environ["GITHUB_TOKEN"], user=user, repo=repo)
        q = commits_query(user, repo)
        r = github.graphql(q)

    commits = r["data"]["repository"]["defaultBranchRef"]["target"]["history"]["nodes"]

    # Limit GraphQL pagination
    MAX_COMMITS_TO_CHECK = 50
    i = 0

    while i < MAX_COMMITS_TO_CHECK:
        # Check each commit
        for commit in commits:
            if commit_passed_ci(commit):
                print(f"Found last good commit: {commit['oid']}: {commit['messageHeadline']}")
                if not args.dry_run:
                    update_branch(
                        user=user,
                        repo=repo,
                        sha=commit["oid"],
                        branch_name=args.branch,
                    )
                # Nothing to do after updating the branch, exit early
                exit(0)

        # No good commit found, proceed to next page of results
        edges = r["data"]["repository"]["defaultBranchRef"]["target"]["history"]["edges"]
        if len(edges) == 0:
            break
        else:
            q = commits_query(user, repo, cursor=edges[-1]["cursor"])
            r = github.graphql(q)
            commits = r["data"]["repository"]["defaultBranchRef"]["target"]["history"]["nodes"]

        # Backstop to prevent looking through all the past commits
        i += len(commits)

    print(f"No good commits found in the last {len(commits)} commits")
    exit(1)
