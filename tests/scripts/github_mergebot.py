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
import warnings
import logging
import traceback
from typing import Dict, Any, List, Optional
from pathlib import Path

from git_utils import git, GitHubRepo, parse_remote
from cmd_utils import init_log


Review = Dict[str, Any]
CIJob = Dict[str, Any]


def js(obj: Any) -> str:
    return json.dumps(obj, indent=2)


PR_QUERY = """
    query ($owner: String!, $name: String!, $number: Int!) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $number) {
          title
          body
          state
          comments(last: 100) {
            pageInfo {
              hasPreviousPage
            }
            nodes {
              author {
                login
              }
              updatedAt
              body
            }
          }
          authorCommits:commits(last:100) {
            nodes {
              commit {
                authors(first:100) {
                  nodes {
                    name
                    email
                  }
                }
              }
            }
          }
          commits(last: 1) {
            nodes {
              commit {
                oid
                statusCheckRollup {
                  contexts(first: 100) {
                    pageInfo {
                      hasNextPage
                    }
                    nodes {
                      ... on CheckRun {
                        name
                        checkSuite {
                          workflowRun {
                            workflow {
                              name
                            }
                          }
                        }
                        status
                        conclusion
                        url
                      }
                      ... on StatusContext {
                        state
                        context
                        targetUrl
                      }
                    }
                  }
                }
              }
            }
          }
          reviewDecision
          reviews(last: 100) {
            pageInfo {
              hasPreviousPage
            }
            nodes {
              body
              updatedAt
              authorCanPushToRepository
              commit {
                oid
              }
              state
            }
          }
        }
      }
    }
    """


def walk(obj, visitor, parent_key=None):
    """
    Recursively call 'visitor' on all the children of a dictionary
    """
    visitor(obj, parent_key)
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, visitor, parent_key=k)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, visitor)


class PR:
    def __init__(
        self,
        number: int,
        owner: str,
        repo: str,
        dry_run: bool = False,
        raw_data: Dict[str, Any] = None,
    ):
        self.owner = owner
        self.number = number
        self.repo_name = repo
        self.dry_run = dry_run

        self.github = GitHubRepo(user=owner, repo=repo, token=os.environ["GITHUB_TOKEN"])

        if dry_run and raw_data:
            # In test mode there is no need to fetch anything
            self.raw = raw_data
        else:
            if os.getenv("DEBUG", "0") == "1":
                # For local runs fill in the requested data but cache it for
                # later use
                cached_path = Path("pr.json")
                if not cached_path.exists():
                    self.raw = self.fetch_data()
                    with open(cached_path, "w") as f:
                        json.dump(self.raw, f, indent=2)
                else:
                    with open(cached_path) as f:
                        self.raw = json.load(f)
            else:
                # Usual path, fetch the PR's data based on the number from
                # GitHub
                self.raw = self.fetch_data()

        def checker(obj, parent_key):
            """
            Verify that any paged results don't have extra data (if so the bot
            may still work since most relevant comments will be more recent)
            """
            if parent_key == "pageInfo":
                if obj.get("hasPreviousPage", False):
                    warnings.warn(f"Found {obj} with a previous page, bot may be missing data")
                if obj.get("hasNextPage", False):
                    warnings.warn(f"Found {obj} with a next page, bot may be missing data")

        walk(self.raw, checker)

        logging.info(f"Verified data, running with PR {js(self.raw)}")

    def __repr__(self):
        return json.dumps(self.raw, indent=2)

    def head_commit(self):
        return self.raw["commits"]["nodes"][0]["commit"]

    def co_authors(self) -> List[str]:
        authors = []
        for commit in self.raw["authorCommits"]["nodes"]:
            # Co-authors always come after the main author according to the
            # GitHub docs, so ignore the first item
            for author in commit["commit"]["authors"]["nodes"][1:]:
                name = author["name"]
                email = author["email"]
                authors.append(f"{name} <{email}>")

        return list(set(authors))

    def head_oid(self):
        return self.head_commit()["oid"]

    def ci_jobs(self) -> List[CIJob]:
        """
        Get a list of all CI jobs (GitHub Actions and other) in a unified format
        """
        jobs = []
        for item in self.head_commit()["statusCheckRollup"]["contexts"]["nodes"]:
            if "checkSuite" in item:
                # GitHub Actions job, parse separately
                status = item["conclusion"]
                if status is None:
                    # If the 'conclusion' isn't filled out the job hasn't
                    # finished yet
                    status = "PENDING"
                jobs.append(
                    {
                        "name": item["checkSuite"]["workflowRun"]["workflow"]["name"]
                        + " / "
                        + item["name"],
                        "url": item["url"],
                        "status": status.upper(),
                    }
                )
            else:
                # GitHub Status (e.g. from Jenkins)
                jobs.append(
                    {
                        "name": item["context"],
                        "url": item["targetUrl"],
                        "status": item["state"].upper(),
                    }
                )

        logging.info(f"Found CI jobs for {self.head_commit()['oid']} {js(jobs)}")
        return jobs

    def reviews(self) -> List[Review]:
        return self.raw["reviews"]["nodes"]

    def head_commit_reviews(self) -> List[Review]:
        """
        Find reviews associated with the head commit
        """
        commits_to_review_status: Dict[str, List[Review]] = {}

        for review in self.reviews():
            if not review["authorCanPushToRepository"]:
                # ignore reviews from non-committers
                continue

            oid = review["commit"]["oid"]
            if oid in commits_to_review_status:
                commits_to_review_status[oid].append(review)
            else:
                commits_to_review_status[oid] = [review]

        # Only use the data for the head commit of the PR
        head_reviews = commits_to_review_status.get(self.head_oid(), [])
        return head_reviews

    def fetch_data(self):
        """
        Fetch the data for this PR from GitHub
        """
        return self.github.graphql(
            query=PR_QUERY,
            variables={
                "owner": self.owner,
                "name": self.repo_name,
                "number": self.number,
            },
        )["data"]["repository"]["pullRequest"]

    def comment(self, text: str) -> None:
        """
        Leave the comment 'text' on this PR
        """
        logging.info(f"Commenting:\n{text}")
        # TODO: Update latest comment in-place if there has been no activity
        data = {"body": text}
        url = f"issues/{self.number}/comments"
        if self.dry_run:
            logging.info(
                f"Dry run, would have commented on url={url} commenting with data={js(data)}"
            )
            return

        self.github.post(url, data=data)

    def state(self) -> str:
        """
        PR state (OPEN, CLOSED, MERGED, etc)
        """
        return self.raw["state"]

    def lint_commit_message(self, subject: str, body: str) -> bool:
        # TODO: NYI (Add rules as decided in https://discuss.tvm.apache.org/t/commit-message-guideline/12334)
        return True

    def processed_body(self) -> str:
        body = self.raw["body"].strip().replace("\r", "")
        body = body.replace(
            "Thanks for contributing to TVM!   Please refer to guideline https://tvm.apache.org/docs/contribute/ for useful information and tips. After the pull request is submitted, please request code reviews from [Reviewers](https://github.com/apache/incubator-tvm/blob/master/CONTRIBUTORS.md#reviewers) by @ them in the pull request thread.",
            "",
        )
        return body

    def body_with_co_authors(self) -> str:
        """
        Add 'Co-authored-by' strings to the PR body based on the prior commits
        in the PR
        """
        body = self.processed_body()
        author_lines = self.co_authors()
        logging.info(f"Found co-authors: author_lines={author_lines}")
        for author_line in author_lines:
            if author_line not in body:
                # If the line isn't already in the PR body (it could have been
                # added manually), put it in
                body = f"{body}\n\nCo-authored-by: {author_line}"

        return body

    def merge(self) -> None:
        """
        Request a merge of this PR via the GitHub API
        """
        url = f"pulls/{self.number}/merge"

        title = self.raw["title"]
        body = self.body_with_co_authors()
        self.lint_commit_message(title, body)
        logging.info(f"Full commit:\n{title}\n\n{body}")

        data = {
            "commit_title": title,
            "commit_message": body,
            # The SHA is necessary in case there was an update right when this
            # script ran, GitHub will sort out who won
            "sha": self.head_oid(),
            "merge_method": "squash",
        }
        if self.dry_run:
            logging.info(f"Dry run, would have merged with url={url} and data={js(data)}")
            return

        self.github.put(url, data=data)

    def merge_requested(self) -> bool:
        """
        Check if this PR has had a merge requested
        """
        merge_commands = [
            "merge",
            "merge this",
            "merge this pr",
        ]
        cancel_commands = [
            "cancel",
            "cancel merge",
            "cancel the merge",
            "stop",
            "stop merge",
            "stop the merge",
        ]

        def parse_action(s: str) -> Optional[str]:
            if any(f"@tvm-bot {c}" in s for c in merge_commands):
                return "merge"

            if any(f"@tvm-bot {c}" in s for c in cancel_commands):
                return "cancel"

            return None

        # Check regular comments
        all_comments = []
        for comment in self.raw["comments"]["nodes"]:
            all_comments.append((comment["updatedAt"], comment["body"]))

        # Check top-level review comments
        for review in self.reviews():
            all_comments.append((review["updatedAt"], review["body"]))

        all_comments = sorted(all_comments, key=lambda x: x[0])
        actions = [parse_action(body) for _, body in all_comments]
        logging.info(f"Found these tvm-bot actions: {actions}")
        actions = [a for a in actions if a is not None]

        if len(actions) == 0:
            return False

        return actions[-1] == "merge"

    def merge_if_passed_checks(self):
        # NEUTRAL is GitHub Action's way of saying cancelled
        failed_ci_jobs = [
            job
            for job in self.ci_jobs()
            if job["status"] not in {"SUCCESS", "SUCCESSFUL", "NEUTRAL", "SKIPPED"}
        ]
        all_ci_passed = False
        has_one_approval = False
        if len(failed_ci_jobs) > 0:
            failed_jobs_msg = "\n".join(
                [f" * [{job['name']} (`{job['status']}`)]({job['url']})" for job in failed_ci_jobs]
            )
            self.comment(
                f"Cannot merge, these CI jobs are not successful on {self.head_oid()}:\n{failed_jobs_msg}"
            )
            return
        else:
            all_ci_passed = True

        head_commit_reviews = self.head_commit_reviews()
        for review in head_commit_reviews:
            if review["state"] == "CHANGES_REQUESTED":
                self.comment(
                    f"Cannot merge, found [this review]({review['url']}) on {self.head_oid()} with changes requested"
                )
                return

            if review["state"] == "APPROVED":
                has_one_approval = True
                logging.info(f"Found approving review: {js(review)}")

        if has_one_approval and all_ci_passed:
            self.merge()
        elif not has_one_approval:
            self.comment(
                f"Cannot merge, did not find any approving reviews from users with write access on {self.head_oid()}"
            )
            return
        elif not all_ci_passed:
            self.comment(f"Cannot merge, CI did not pass on on {self.head_oid()}")
            return


if __name__ == "__main__":
    help = "Automatically tag people based on PR / issue labels"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--remote", default="origin", help="ssh remote to parse")
    parser.add_argument("--pr", required=True, help="pr number to check")
    parser.add_argument("--run-url", required=True, help="workflow run URL")
    parser.add_argument("--testing-pr-json", help="(testing only) manual data for testing")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="run but don't send any request to GitHub",
    )
    args = parser.parse_args()
    init_log()

    remote = git(["config", "--get", f"remote.{args.remote}.url"])
    logging.info(f"Using remote remote={remote}")
    owner, repo = parse_remote(remote)

    if args.pr.strip() == "":
        logging.info("No PR number passed")
        exit(0)

    logging.info(f"Checking owner={owner} repo={repo}")
    if args.testing_pr_json:
        pr = PR(
            number=int(args.pr),
            owner=owner,
            repo=repo,
            dry_run=args.dry_run,
            raw_data=json.loads(args.testing_pr_json),
        )
    else:
        pr = PR(number=int(args.pr), owner=owner, repo=repo, dry_run=args.dry_run)

    state = pr.state()

    if state != "OPEN":
        logging.info(f"Ignoring event on PR, state was not OPEN, instead was state={state}")
        exit(0)

    if pr.merge_requested():
        try:
            pr.merge_if_passed_checks()
        except Exception as e:
            if not args.dry_run:
                msg = traceback.format_exc()
                pr.comment(
                    f"Failed to process merge request in {args.run_url}\n\n<details>\n\n```\n{msg}\n```\n\n</details>"
                )
            raise e
    else:
        logging.info("No merge requested, exiting")
