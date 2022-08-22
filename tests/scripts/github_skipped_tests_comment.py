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
import subprocess
import sys
from urllib import error
from xml.etree import ElementTree

import requests

from git_utils import git, GitHubRepo, parse_remote
from cmd_utils import init_log

SKIPPED_TESTS_COMMENT_MARKER = "<!---skipped-tests-comment-->\n\n"
GITHUB_ACTIONS_BOT_LOGIN = "github-actions[bot]"

PR_TEST_REPORT_DIR = "pr-reports"
MAIN_TEST_REPORT_DIR = "main-reports"


def retrieve_test_report(s3_url, target_dir):
    command = f"aws s3 cp {s3_url} {target_dir} --recursive"
    logging.info(f"Running command {command}")
    proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed {command}:\nstdout:\n{proc.stdout}")


def retrieve_test_reports(pr_number, build_number, s3_prefix, jenkins_prefix):
    cur_build_s3_link = (
        f"s3://{s3_prefix}/tvm/PR-{str(pr_number)}/{str(build_number)}/pytest-results"
    )
    retrieve_test_report(cur_build_s3_link, PR_TEST_REPORT_DIR)

    latest_main_build = requests.get(
        f"https://{jenkins_prefix}/job/tvm/job/main/lastSuccessfulBuild/buildNumber"
    ).text
    latest_build_s3_link = f"s3://{s3_prefix}/tvm/main/{latest_main_build}/pytest-results"
    retrieve_test_report(latest_build_s3_link, MAIN_TEST_REPORT_DIR)


def get_pr_and_build_numbers(target_url):
    target_url = target_url[target_url.find("PR-") : len(target_url)]
    split = target_url.split("/")
    pr_number = split[0].strip("PR-")
    build_number = split[1]
    return {"pr_number": pr_number, "build_number": build_number}


def build_test_set(directory):
    subdir_to_skipped = {}
    subdirs = [
        item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))
    ]
    for subdir in subdirs:
        subdir_to_skipped[subdir] = set()
        for root, _, files in os.walk(directory + "/" + subdir):
            for file in files:
                test_report = ElementTree.parse(root + "/" + file)
                for testcase in test_report.iter("testcase"):
                    skipped = testcase.find("skipped")
                    if skipped is not None:
                        key = testcase.attrib["classname"] + "#" + testcase.attrib["name"]
                        subdir_to_skipped[subdir].add(key)
    return subdir_to_skipped


def to_node_name(dir_name: str):
    return dir_name.replace("_", ": ", 1)


def build_comment(skipped_list, pr_number, build_number, commit_sha, jenkins_prefix):
    if len(skipped_list) == 0:
        return f"{SKIPPED_TESTS_COMMENT_MARKER}No additional skipped tests found in this branch for commit {commit_sha}."

    text = (
        f"{SKIPPED_TESTS_COMMENT_MARKER}The list below shows some tests that ran in main but were skipped in the "
        f"CI build of {commit_sha}:\n"
        f"```\n"
    )
    for skip in skipped_list:
        text += skip + "\n"
    text += (
        f"```\nA detailed report of ran tests is [here](https://{jenkins_prefix}/job/tvm/job/PR-{str(pr_number)}"
        f"/{str(build_number)}/testReport/)."
    )
    return text


def get_pr_comments(github, url):
    try:
        return github.get(url)
    except error.HTTPError as e:
        logging.exception(f"Failed to retrieve PR comments: {url}: {e}")
        return []


def search_for_docs_comment(comments):
    for comment in comments:
        if (
            comment["user"]["login"] == GITHUB_ACTIONS_BOT_LOGIN
            and SKIPPED_TESTS_COMMENT_MARKER in comment["body"]
        ):
            return comment
    return None


if __name__ == "__main__":
    help = (
        "Compares the skipped tests of this PR against the last successful build on main. Also comments on the PR "
        "issue when tests are skipped in this PR and not on main."
    )
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--remote", default="origin", help="ssh remote to parse")
    parser.add_argument("--s3-prefix", default="tvm-jenkins-artifacts-prod")
    parser.add_argument("--jenkins-prefix", default="ci.tlcpack.ai")
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

    if not args.dry_run:
        retrieve_test_reports(
            pr_number=pr_and_build["pr_number"],
            build_number=pr_and_build["build_number"],
            s3_prefix=args.s3_prefix,
            jenkins_prefix=args.jenkins_prefix,
        )

    main_tests = build_test_set(MAIN_TEST_REPORT_DIR)
    build_tests = build_test_set(PR_TEST_REPORT_DIR)

    skipped_list = []
    for subdir, skipped_set in build_tests.items():
        skipped_main = main_tests[subdir]
        if skipped_main is None:
            logging.warning(f"Could not find directory {subdir} in main.")
            continue

        diff_set = skipped_set - skipped_main
        if len(diff_set) != 0:
            for test in diff_set:
                skipped_list.append(f"{to_node_name(subdir)} -> {test}")

    if len(skipped_list) == 0:
        logging.info("No skipped tests found.")

    body = build_comment(
        skipped_list,
        pr_and_build["pr_number"],
        pr_and_build["build_number"],
        commit_sha,
        args.jenkins_prefix,
    )
    url = f'issues/{pr_and_build["pr_number"]}/comments'
    if not args.dry_run:
        github = GitHubRepo(token=os.environ["GITHUB_TOKEN"], user=user, repo=repo)

        # For now, only comment for PRs open by driazati, gigiblender and areusch.
        get_pr_url = f'pulls/{pr_and_build["pr_number"]}'
        pull_request_body = github.get(get_pr_url)
        author = pull_request_body["user"]["login"]
        if author not in ["driazati", "gigiblender", "areusch"]:
            logging.info(f"Skipping this action for user {author}")
            sys.exit(0)

        pr_comments = get_pr_comments(github, url)
        comment = search_for_docs_comment(pr_comments)

        if comment is not None:
            comment_url = comment["url"]
            comment_id = comment_url[comment_url.find("comments/"): len(comment_url)].strip("comments/")
            github.patch(f'issues/comments/{comment_id}', {"body": body})
        else:
            github.post(url, {"body": body})
    else:
        logging.info(f"Dry run, would have posted {url} with data {body}.")
