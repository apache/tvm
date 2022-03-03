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
from typing import Dict, Any, List, Tuple


from git_utils import git, GitHubRepo, parse_remote, find_ccs


def parse_line(line: str) -> Tuple[str, List[str]]:
    line = line.lstrip(" -")
    line = line.split()

    # Parse out the name as anything up until the first tagged person
    tag_items = []
    tag_end = 0
    for i, piece in enumerate(line):
        if piece.startswith("@"):
            tag_end = i
            break

        tag_items.append(piece)

    tag = " ".join(tag_items).rstrip(":")

    # From the last word that was part of the tag name, start looking for users
    # tagged with a '@'
    users = []
    for piece in line[tag_end:]:
        if piece.startswith("@"):
            users.append(piece.lstrip("@"))

    return (tag, list(sorted(users)))


def fetch_issue(github: GitHubRepo, issue_number: int):
    query = """query($owner: String!, $name: String!, $number: Int!){
    repository(owner: $owner, name: $name) {
        issue(number: $number) {
        body
        comments(first:100) {
            nodes {
            body
            }
        }
        }
    }
    }"""
    r = github.graphql(
        query,
        variables={
            "owner": github.user,
            "name": github.repo,
            "number": issue_number,
        },
    )
    return r


def find_rollout_users(r: Dict[str, Any]):
    issue = r["data"]["repository"]["issue"]
    body = issue["body"]
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("[temporary] opt-in: "):
            line = line[len("[temporary] opt-in: ") :]
            return find_ccs("cc " + line)

    return []


def parse_teams(r: Dict[str, Any], issue_number: int) -> Dict[str, str]:
    """
    Fetch an issue and parse out series of tagged people from the issue body
    and comments
    """
    issue = r["data"]["repository"]["issue"]

    if issue is None or issue.get("body") is None:
        raise RuntimeError(f"Could not find issue #{issue_number}\n\n{json.dumps(r, indent=2)}")

    result = {}

    def add_tag(tag, users):
        if tag in result:
            result[tag] += users
        else:
            result[tag] = users

    # Parse the issue body (only bullets are looked at)
    for line in issue["body"].split("\n"):
        line = line.strip()
        if not line.startswith("- "):
            continue
        if "@" not in line:
            continue

        tag, users = parse_line(line)
        add_tag(tag, users)

    # Parse comment bodies
    for comment in issue["comments"]["nodes"]:
        for line in comment["body"].split("\n"):
            if "@" not in line:
                continue

            tag, users = parse_line(line)
            add_tag(tag, users)

    # De-duplicate users listed twice for the same tag
    for tag in result:
        result[tag] = list(set(result[tag]))

    return {k.lower(): v for k, v in result.items()}


def tags_from_title(title: str) -> List[str]:
    tags = re.findall(r"\[(.*?)\]", title)
    tags = [t.strip() for t in tags]
    return tags


def tags_from_labels(labels: List[Dict[str, Any]]) -> List[str]:
    return [label["name"] for label in labels]


def add_ccs_to_body(body: str, to_cc: List[str]) -> str:
    lines = body.split("\n")

    cc_line_idx = None
    for i, line in enumerate(reversed(lines)):
        if line.strip() == "":
            continue
        if line.startswith("cc @"):
            cc_line_idx = len(lines) - i - 1
        else:
            break

    def gen_cc_line(users):
        users = sorted(users)
        return "cc " + " ".join([f"@{user}" for user in users])

    if cc_line_idx is None:
        print("Did not find existing cc line")
        lines.append("")
        lines.append(gen_cc_line(to_cc))
    else:
        # Edit cc line in place
        line = lines[cc_line_idx]
        print(f"Found existing cc line at {cc_line_idx}: {line}")
        existing_ccs = find_ccs(line)
        print(f"Found cc's: {existing_ccs}")

        if set(to_cc).issubset(set(existing_ccs)):
            # Don't do anything if there is no update needed
            return None

        line = gen_cc_line(set(existing_ccs + to_cc))

        lines[cc_line_idx] = line

    return "\n".join(lines)


if __name__ == "__main__":
    help = "Automatically tag people based on PR / issue labels"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--remote", default="origin", help="ssh remote to parse")
    parser.add_argument("--team-issue", default="10317", help="issue number to look at for ccs")
    parser.add_argument(
        "--team-issue-json", help="(testing only) issue JSON to parse rather than fetch from GitHub"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="run but don't send any request to GitHub",
    )
    args = parser.parse_args()

    remote = git(["config", "--get", f"remote.{args.remote}.url"])
    user, repo = parse_remote(remote)

    if args.team_issue_json:
        issue_data = json.loads(args.team_issue_json)
    else:
        github = GitHubRepo(token=os.environ["GITHUB_TOKEN"], user=user, repo=repo)
        issue_data = fetch_issue(github, issue_number=int(args.team_issue))

    # Fetch the list of teams
    teams = parse_teams(issue_data, issue_number=int(args.team_issue))
    # When rolling out this tool it is limited to certain users, so find that list
    rollout_users = find_rollout_users(issue_data)
    print(f"[slow rollout] Limiting to opted-in users: {rollout_users}")

    print(f"Found these teams in issue #{args.team_issue}\n{json.dumps(teams, indent=2)}")

    # Extract the payload from GitHub Actions
    issue = json.loads(os.getenv("ISSUE", "null"))
    pr = json.loads(os.getenv("PR", "null"))
    if (issue is None and pr is None) or (issue is not None and pr is not None):
        raise RuntimeError("Exactly one of $PR or $ISSUE must be set in the environment")

    if pr is not None:
        if pr["draft"]:
            print(f"Terminating since {pr['number']} is a draft")
            exit(0)

    # PRs/issues have the same structure for the fields needed here
    item = issue if issue is not None else pr
    title = item["title"]
    body = item["body"]
    author = item["user"]["login"]
    tags = tags_from_title(item["title"]) + tags_from_labels(item["labels"])

    tags = [t.lower() for t in tags]
    print(f"Found tags: {tags}")

    if author not in rollout_users:
        print(f"Author {author} is not opted in, quitting")
        exit(0)

    # Update the PR or issue based on tags in the title and GitHub tags
    to_cc = [teams.get(t, []) for t in tags]
    to_cc = list(set(item for sublist in to_cc for item in sublist))
    to_cc = [user for user in to_cc if user != author]
    print("Users to cc based on labels", to_cc)

    # Create the new PR/issue body
    if len(to_cc) == 0:
        print("No one to cc, exiting")
        exit(0)

    new_body = add_ccs_to_body(body, to_cc)
    if new_body is None:
        print(f"Everyone to cc is already cc'ed, no update needed")
        exit(0)

    print(f"Changing body from:\n----\n{body}\n----\nto:\n----\n{new_body}\n----")

    # Set the PR/issue body on GitHub
    data = {"body": new_body}
    if issue is not None:
        issue_number = issue["number"]
        url = f"issues/{issue_number}"
    elif pr is not None:
        pr_number = pr["number"]
        url = f"pulls/{pr_number}"
    else:
        raise RuntimeError("Unreachable, please report a bug with a link to the failed workflow")

    if not args.dry_run:
        github.post(url, data=data)
    else:
        print(f"Dry run, would have updated {url} with {data}")
