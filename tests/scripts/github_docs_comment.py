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
import logging
import argparse
import sys
from urllib import error

from git_utils import git, GitHubRepo, parse_remote
from cmd_utils import init_log


def build_docs_url(base_url_docs, pr_number, build_number):
    return f"{base_url_docs}/PR-{str(pr_number)}/{str(build_number)}/docs/index.html"


def get_pr_and_build_numbers(target_url):
    target_url = target_url[target_url.find("PR-") : len(target_url)]
    split = target_url.split("/")
    pr_number = split[0].strip("PR-")
    build_number = split[1]
    return {"pr_number": pr_number, "build_number": build_number}


if __name__ == "__main__":
    help = "Add comment with link to docs"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--remote", default="origin", help="ssh remote to parse")
    parser.add_argument("--base-url-docs", default="https://pr-docs.tlcpack.ai")
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

    target_url = os.environ["TARGET_URL"]
    pr_and_build = get_pr_and_build_numbers(target_url)

    commit_sha = os.environ["COMMIT_SHA"]

    docs_url = build_docs_url(
        args.base_url_docs, pr_and_build["pr_number"], pr_and_build["build_number"]
    )

    url = f'issues/{pr_and_build["pr_number"]}/comments'
    body = f"Built docs for commit {commit_sha} can be found [here]({docs_url})."
    if not args.dry_run:
        github = GitHubRepo(token=os.environ["GITHUB_TOKEN"], user=user, repo=repo)

        # For now, only comment for PRs open by driazati, gigiblender and areusch.
        get_pr_url = f'pulls/{pr_and_build["pr_number"]}'
        pull_request_body = github.get(get_pr_url)
        author = pull_request_body["user"]["login"]
        if author not in ["driazati", "gigiblender", "areusch"]:
            logging.info(f"Skipping this action for user {author}")
            sys.exit(0)

        try:
            github.post(url, {"body": body})
        except error.HTTPError as e:
            logging.exception(f"Failed to add docs comment {docs_url}: {e}")
    else:
        logging.info(f"Dry run, would have posted {url} with data {body}.")
