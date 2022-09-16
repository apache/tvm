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
import os
import json
import logging

from git_utils import git, GitHubRepo, parse_remote, DRY_RUN
from cmd_utils import init_log
from github_commenter import BotCommentBuilder
from github_skipped_tests_comment import get_skipped_tests_comment
from github_tag_teams import get_tags
from github_docs_comment import get_doc_url

PR_QUERY = """
    query ($owner: String!, $name: String!, $number: Int!) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $number) {
          title
          body
          state
          number
          isDraft
          author {
            login
          }
          labels(first:100) {
            nodes {
              name
            }
          }
          comments(last: 100) {
            pageInfo {
              hasPreviousPage
            }
            nodes {
              author {
                login
              }
              databaseId
              body
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
        }
      }
    }
"""


if __name__ == "__main__":
    help = "Comment a welcome message on PRs"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--remote", default="origin", help="ssh remote to parse")
    parser.add_argument("--pr", required=True)
    parser.add_argument("--run-url", required=True, help="URL of the current GitHub Actions run")
    parser.add_argument("--test-data", help="(testing) mock GitHub API data")
    parser.add_argument("--test-comments", help="(testing) testing comments")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="run but don't send any request to GitHub",
    )
    args = parser.parse_args()
    init_log()

    remote = git(["config", "--get", f"remote.{args.remote}.url"])
    user, repo = parse_remote(remote)

    test_data = None
    if args.test_data is not None:
        test_data = json.loads(args.test_data)

    github = GitHubRepo(
        user=user,
        repo=repo,
        token=DRY_RUN if args.dry_run else os.environ["GITHUB_TOKEN"],
        test_data=test_data,
    )

    pr_data = github.graphql(
        PR_QUERY,
        {
            "owner": user,
            "name": repo,
            "number": int(args.pr),
        },
    )

    pr_data = pr_data["data"]["repository"]["pullRequest"]
    commenter = BotCommentBuilder(github=github, data=pr_data, run_url=args.run_url)

    commenters = {
        "ccs": (
            get_tags,
            {
                "pr_data": pr_data,
                "github": github,
                "team_issue": 10317,
            },
        ),
        "skipped-tests": (
            get_skipped_tests_comment,
            {
                "pr": pr_data,
                "github": github,
            },
        ),
        "docs": (
            get_doc_url,
            {
                "pr": pr_data,
            },
        ),
    }
    items = {}
    logging.info(f"Dispatching to {commenters.keys()}")

    test_comments = None
    if args.test_comments is not None:
        test_comments = json.loads(args.test_comments)

    for tag, (callable, args) in commenters.items():
        if test_comments is not None:
            items[tag] = test_comments[tag]
        else:
            try:
                items[tag] = callable(**args)
            except Exception as e:
                logging.exception(e)
                items[tag] = None

    commenter.post_items(items=items.items())
