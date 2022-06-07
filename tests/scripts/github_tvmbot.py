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
import re
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

from git_utils import git, GitHubRepo, parse_remote, post
from cmd_utils import init_log


Review = Dict[str, Any]
CIJob = Dict[str, Any]
Comment = Dict[str, Any]
CommentChecker = Callable[[Comment], bool]

EXPECTED_JOBS = ["tvm-ci/pr-head"]
TVM_BOT_JENKINS_TOKEN = os.environ["TVM_BOT_JENKINS_TOKEN"]
JENKINS_URL = "https://ci.tlcpack.ai/"
THANKS_MESSAGE = r"(\s*)Thanks for contributing to TVM!   Please refer to guideline https://tvm.apache.org/docs/contribute/ for useful information and tips. After the pull request is submitted, please request code reviews from \[Reviewers\]\(https://github.com/apache/incubator-tvm/blob/master/CONTRIBUTORS.md#reviewers\) by  them in the pull request thread.(\s*)"


def to_json_str(obj: Any) -> str:
    return json.dumps(obj, indent=2)


COLLABORATORS_QUERY = """
query ($owner: String!, $name: String!, $user: String!) {
  repository(owner: $owner, name: $name) {
    collaborators(query: $user, first: 1) {
      nodes {
        login
      }
    }
  }
}
"""


PR_QUERY = """
    query ($owner: String!, $name: String!, $number: Int!) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $number) {
          title
          body
          state
          author {
            login
          }
          comments(last: 100) {
            pageInfo {
              hasPreviousPage
            }
            nodes {
              authorAssociation
              author {
                login
              }
              id
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
              url
              id
              authorCanPushToRepository
              commit {
                oid
              }
              author {
                login
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

        if dry_run and raw_data:
            # In test mode there is no need to fetch anything
            self.raw = raw_data
            self.github = None
        else:
            self.github = GitHubRepo(user=owner, repo=repo, token=os.environ["GITHUB_TOKEN"])
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

        logging.info(f"Verified data, running with PR {to_json_str(self.raw)}")

    def __repr__(self):
        return json.dumps(self.raw, indent=2)

    def plus_one(self, comment: Dict[str, Any]):
        """
        React with a thumbs up to a comment
        """
        url = f"issues/comments/{comment['id']}/reactions"
        data = {"content": "+1"}
        if self.dry_run:
            logging.info(f"Dry run, would have +1'ed to {url} with {data}")
        else:
            self.github.post(url, data=data)

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

        logging.info(f"Found CI jobs for {self.head_commit()['oid']} {to_json_str(jobs)}")
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

    def search_collaborator(self, user: str) -> List[Dict[str, Any]]:
        """
        Query GitHub for collaborators matching 'user'
        """
        return self.github.graphql(
            query=COLLABORATORS_QUERY,
            variables={
                "owner": self.owner,
                "name": self.repo_name,
                "user": user,
            },
        )["data"]["repository"]["collaborators"]["nodes"]

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
                f"Dry run, would have commented on url={url} commenting with data={to_json_str(data)}"
            )
            return

        self.github.post(url, data=data)

    def state(self) -> str:
        """
        PR state (OPEN, CLOSED, MERGED, etc)
        """
        return self.raw["state"]

    def processed_body(self) -> str:
        body = self.raw["body"].strip().replace("\r", "")
        # Remove any @-mentions of people
        body = re.sub(r"(\s)@", "\g<1>", body)

        # Remove the auto-inserted text since it's not useful to have in the commit log
        body = re.sub(THANKS_MESSAGE, "\n\n", body)
        return body.strip()

    def body_with_co_authors(self) -> str:
        """
        Add 'Co-authored-by' strings to the PR body based on the prior commits
        in the PR
        """
        body = self.processed_body()
        author_lines = self.co_authors()
        logging.info(f"Found co-authors: author_lines={author_lines}")
        full_author_lines = [f"Co-authored-by: {author_line}" for author_line in author_lines]

        authors_to_add = []
        for author_line in author_lines:
            if author_line not in body:
                authors_to_add.append(f"Co-authored-by: {author_line}")

        if len(authors_to_add) > 0:
            # If the line isn't already in the PR body (it could have been
            # added manually), put it in
            full_author_text = "\n".join(authors_to_add)
            body = f"{body}\n\n{full_author_text}"

        return body

    def merge(self) -> None:
        """
        Request a merge of this PR via the GitHub API
        """
        url = f"pulls/{self.number}/merge"

        title = self.raw["title"] + f" (#{self.number})"
        body = self.body_with_co_authors()
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
            logging.info(f"Dry run, would have merged with url={url} and data={to_json_str(data)}")
            return

        self.github.put(url, data=data)

    def author(self) -> str:
        return self.raw["author"]["login"]

    def find_failed_ci_jobs(self) -> List[CIJob]:
        # NEUTRAL is GitHub Action's way of saying cancelled
        return [
            job
            for job in self.ci_jobs()
            if job["status"] not in {"SUCCESS", "SUCCESSFUL", "SKIPPED"}
        ]

    def find_missing_expected_jobs(self) -> List[str]:
        # Map of job name: has seen in completed jobs
        seen_expected_jobs = {name: False for name in EXPECTED_JOBS}
        logging.info(f"Expected to see jobs: {seen_expected_jobs}")

        missing_expected_jobs = []
        for job in self.ci_jobs():
            seen_expected_jobs[job["name"]] = True

        for name, seen in seen_expected_jobs.items():
            if not seen:
                missing_expected_jobs.append(name)

        return missing_expected_jobs

    def merge_if_passed_checks(self) -> None:
        failed_ci_jobs = self.find_failed_ci_jobs()
        all_ci_passed = len(failed_ci_jobs) == 0
        has_one_approval = False

        if not all_ci_passed:
            failed_jobs_msg = "\n".join(
                [f" * [{job['name']} (`{job['status']}`)]({job['url']})" for job in failed_ci_jobs]
            )
            self.comment(
                f"Cannot merge, these CI jobs are not successful on {self.head_oid()}:\n{failed_jobs_msg}"
            )
            return

        missing_expected_jobs = self.find_missing_expected_jobs()

        if len(missing_expected_jobs) > 0:
            missing_jobs_msg = "\n".join([f" * `{name}`" for name in missing_expected_jobs])
            self.comment(f"Cannot merge, missing expected jobs:\n{missing_jobs_msg}")
            return

        head_commit_reviews = self.head_commit_reviews()
        for review in head_commit_reviews:
            if review["state"] == "CHANGES_REQUESTED":
                self.comment(
                    f"Cannot merge, found [this review]({review['url']}) on {self.head_oid()} with changes requested"
                )
                return

            if review["state"] == "APPROVED":
                has_one_approval = True
                logging.info(f"Found approving review: {to_json_str(review)}")

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

    def rerun_jenkins_ci(self) -> None:
        url = JENKINS_URL + f"job/tvm/job/PR-{self.number}/buildWithParameters"
        logging.info(f"Rerunning ci with URL={url}")
        if self.dry_run:
            logging.info("Dry run, not sending POST")
        else:
            post(url, auth=("tvm-bot", TVM_BOT_JENKINS_TOKEN))


class Merge:
    triggers = [
        "merge",
        "merge this",
        "merge this pr",
    ]

    @staticmethod
    def run(pr: PR):
        try:
            pr.merge_if_passed_checks()
        except Exception as e:
            if not args.dry_run:
                msg = traceback.format_exc()
                pr.comment(
                    f"Failed to process merge request in {args.run_url}\n\n<details>\n\n```\n{msg}\n```\n\n</details>"
                )
            raise e


class Rerun:
    triggers = [
        "rerun",
        "rerun ci",
        "re-run",
        "re-run ci",
        "run",
        "run ci",
    ]

    @staticmethod
    def run(pr: PR):
        pr.rerun_jenkins_ci()


if __name__ == "__main__":
    help = "Check if a PR has comments trying to merge it, and do so based on reviews/CI status"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--remote", default="origin", help="ssh remote to parse")
    parser.add_argument("--pr", required=True, help="pr number to check")
    parser.add_argument("--run-url", required=True, help="workflow run URL")
    parser.add_argument(
        "--trigger-comment-json", required=True, help="json of the comment that triggered this run"
    )
    parser.add_argument("--testing-pr-json", help="(testing only) manual data for testing")
    parser.add_argument(
        "--testing-collaborators-json", help="(testing only) manual data for testing"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="run but don't send any request to GitHub",
    )
    args = parser.parse_args()
    init_log()
    comment = json.loads(args.trigger_comment_json)
    body = comment["body"].strip()

    # Check that the comment was addressed to tvm-bot
    if not body.startswith("@tvm-bot "):
        logging.info(f"Not a bot comment, '{body}' does not start with '@tvm-bot'")
        exit(0)

    # Find the code to run for the command from the user
    user_command = body.lstrip("@tvm-bot").strip()
    command_to_run = None
    for command in [Merge, Rerun]:
        if user_command in command.triggers:
            command_to_run = command
            break

    if command_to_run is None:
        logging.info(f"Command '{user_command}' did not match anything")
        exit(0)

    # Find the remote for querying more data about the PR
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

    # Acknowledge the comment with a react
    pr.plus_one(comment)

    # Check the comment author
    comment_author = comment["user"]["login"]
    if pr.author() == comment_author:
        logging.info("Comment user is PR author, continuing")
    else:
        logging.info("Comment is not from PR author, checking collaborators")
        # Get the list of collaborators for the repo filtered by the comment
        # author
        if args.testing_collaborators_json:
            collaborators = json.loads(args.testing_collaborators_json)
        else:
            collaborators = pr.search_collaborator(comment_author)
        logging.info(f"Found collaborators: {collaborators}")

        if len(collaborators) > 0:
            logging.info("Comment is from collaborator")
        else:
            logging.info("Comment is not from from PR author or collaborator, quitting")
            exit(0)

    state = pr.state()

    if state != "OPEN":
        logging.info(f"Ignoring event on PR, state was not OPEN, instead was state={state}")
        exit(0)

    # Run the command
    command_to_run.run(pr)
