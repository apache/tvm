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
import datetime
import json
import sys
import textwrap
from pathlib import Path
from typing import List

# Hackery to enable importing of utils from ci/scripts/jenkins
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(REPO_ROOT / "ci" / "scripts" / "jenkins"))

from git_utils import git, parse_remote

GIT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def prs_query(user: str, repo: str, cursor: str = None):
    after = ""
    if cursor is not None:
        after = f', before:"{cursor}"'
    time_keys = "createdAt updatedAt lastEditedAt publishedAt"
    return f"""
        {{
    repository(name: "{repo}", owner: "{user}") {{
        pullRequests(states: [OPEN], last: 10{after}) {{
        edges {{
            cursor
        }}
        nodes {{
            number
            url
            body
            {time_keys}
            isDraft
            author {{
                login
            }}
            reviews(last:100) {{
                nodes {{
                    {time_keys}
                    bodyText
                    author {{ login }}
                    comments(last:100) {{
                        nodes {{
                            {time_keys}
                            bodyText
                        }}
                    }}
                }}
            }}
            comments(last:100) {{
                nodes {{
                    authorAssociation
                    bodyText
                    {time_keys}
                    author {{
                        login
                    }}
                }}
            }}
        }}
        }}
    }}
    }}
    """


def find_reviewers(body: str) -> List[str]:
    matches = re.findall(r"(cc( @[-A-Za-z0-9]+)+)", body, flags=re.MULTILINE)
    matches = [full for full, last in matches]

    reviewers = []
    for match in matches:
        if match.startswith("cc "):
            match = match.replace("cc ", "")
        users = [x.strip() for x in match.split("@")]
        reviewers += users

    reviewers = set(x for x in reviewers if x != "")
    return list(reviewers)


def check_pr(pr, wait_time, now):
    last_action = None

    author = pr["author"]["login"]

    def update_last(new_time, description):
        if isinstance(new_time, str):
            new_time = datetime.datetime.strptime(new_time, GIT_DATE_FORMAT)
        if new_time is None:
            print(f"  time not found: {description}")
            return
        nonlocal last_action
        if last_action is None or new_time > last_action[0]:
            last_action = (new_time, description)

    def check_obj(obj, name):
        update_last(obj["publishedAt"], f"{name} publishedAt: {obj}")
        update_last(obj["updatedAt"], f"{name} updatedAt: {obj}")
        update_last(obj["lastEditedAt"], f"{name} lastEditedAt: {obj}")
        update_last(obj["createdAt"], f"{name} lastEditedAt: {obj}")

    check_obj(pr, "pr")

    # GitHub counts comments left as part of a review separately than standalone
    # comments
    reviews = pr["reviews"]["nodes"]
    review_comments = []
    for review in reviews:
        review_comments += review["comments"]["nodes"]
        check_obj(review, "review")

    # Collate all comments
    comments = pr["comments"]["nodes"] + review_comments

    # Find the last date of any comment
    for comment in comments:
        check_obj(comment, "comment")

    time_since_last_action = now - last_action[0]

    # Find reviewers in the PR's body
    pr_body_reviewers = find_reviewers(pr["body"])

    # Pull out reviewers from any cc @... text in a comment
    cc_reviewers = [find_reviewers(c["bodyText"]) for c in comments]
    cc_reviewers = [r for revs in cc_reviewers for r in revs]

    # Anyone that has left a review as a reviewer (this may include the PR
    # author since their responses count as reviews)
    review_reviewers = list(set(r["author"]["login"] for r in reviews))

    reviewers = cc_reviewers + review_reviewers + pr_body_reviewers
    reviewers = list(set(reviewers))
    reviewers = [r for r in reviewers if r != author]

    if time_since_last_action > wait_time:
        print(
            "  Pinging reviewers",
            reviewers,
            "on",
            pr["url"],
            "since it has been",
            time_since_last_action,
            f"since anything happened on that PR (last action: {last_action[1]})",
        )
        return reviewers
    else:
        print(
            f"  Not pinging PR {pr['number']} since it has been only {time_since_last_action} since the last action: {last_action[1]}"
        )

    return None


def make_ping_message(pr, reviewers):
    reviewers = [f"@{r}" for r in reviewers]
    author = f'@{pr["author"]["login"]}'
    text = (
        "It has been a while since this PR was updated, "
        + " ".join(reviewers)
        + " please leave a review or address the outstanding comments. "
        + f"{author} if this PR is still a work in progress, please [convert it to a draft](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request#converting-a-pull-request-to-a-draft)"
        " until it is ready for review."
    )
    return text


if __name__ == "__main__":
    help = "Comment on languishing issues and PRs"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--remote", default="origin", help="ssh remote to parse")
    parser.add_argument("--wait-time-minutes", required=True, type=int, help="ssh remote to parse")
    parser.add_argument("--cutoff-pr-number", default=0, type=int, help="ssh remote to parse")
    parser.add_argument("--dry-run", action="store_true", help="don't update GitHub")
    parser.add_argument("--pr-json", help="(testing) data for testing to use instead of GitHub")
    parser.add_argument("--now", help="(testing) custom string for current time")
    args = parser.parse_args()

    remote = git(["config", "--get", f"remote.{args.remote}.url"])
    user, repo = parse_remote(remote)

    wait_time = datetime.timedelta(minutes=int(args.wait_time_minutes))
    cutoff_pr_number = int(args.cutoff_pr_number)
    print(
        "Running with:\n"
        f"  time cutoff: {wait_time}\n"
        f"  number cutoff: {cutoff_pr_number}\n"
        f"  dry run: {args.dry_run}\n"
        f"  user/repo: {user}/{repo}\n",
        end="",
    )

    if args.pr_json:
        r = json.loads(args.pr_json)
    else:
        q = prs_query(user, repo)
        r = github.graphql(q)

    now = datetime.datetime.utcnow()
    if args.now:
        now = datetime.datetime.strptime(args.now, GIT_DATE_FORMAT)

    # Loop until all PRs have been checked
    while True:
        prs = r["data"]["repository"]["pullRequests"]["nodes"]

        # Don't look at draft PRs at all
        prs_to_check = []
        for pr in prs:
            if pr["isDraft"]:
                print(f"Skipping #{pr['number']} since it's a draft")
            elif pr["number"] <= cutoff_pr_number:
                print(
                    f"Skipping #{pr['number']} since it's too old ({pr['number']} <= {cutoff_pr_number})"
                )
            else:
                print(f"Checking #{pr['number']}")
                prs_to_check.append(pr)

        print(f"Summary: Checking {len(prs_to_check)} of {len(prs)} fetched")

        # Ping reviewers on each PR in the response if necessary
        for pr in prs_to_check:
            print("Checking", pr["url"])
            reviewers = check_pr(pr, wait_time, now)
            if reviewers is not None:
                message = make_ping_message(pr, reviewers)
                if args.dry_run:
                    print(
                        f"Would have commented on #{pr['number']}:\n{textwrap.indent(message, prefix='  ')}"
                    )
                else:
                    r = github.post(f"issues/{pr['number']}/comments", {"body": message})
                    print(r)

        edges = r["data"]["repository"]["pullRequests"]["edges"]
        if len(edges) == 0:
            # No more results to check
            break

        cursor = edges[0]["cursor"]
        r = github.graphql(prs_query(user, repo, cursor))
