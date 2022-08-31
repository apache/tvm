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
import json
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


def run_subprocess(command):
    logging.info(f"Running command {command}")
    proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed {command}:\nstdout:\n{proc.stdout}")
    return proc


def retrieve_test_report(s3_url, target_dir):
    command = f"aws s3 cp {s3_url} {target_dir} --recursive"
    run_subprocess(command)


def get_common_commit_sha():
    command = "git merge-base origin/main HEAD"
    proc = run_subprocess(command)
    return proc.stdout.strip()


def get_main_jenkins_build_number(github, common_commit):
    json = github.get(f"commits/{common_commit}/status")
    for status in reversed(json["statuses"]):
        if status["context"] != "tvm-ci/branch":
            continue
        state = status["state"]
        target_url = str(status["target_url"])
        build_number = (
            target_url[target_url.find("job/main") : len(target_url)]
            .strip("job/main/")
            .strip("/display/redirect")
        )
        assert build_number.isdigit()
        return {"build_number": build_number, "state": state}
    raise RuntimeError(f"Failed to find main build number for commit {common_commit}")


def retrieve_test_reports(common_main_build, pr_number, build_number, s3_prefix):
    cur_build_s3_link = (
        f"s3://{s3_prefix}/tvm/PR-{str(pr_number)}/{str(build_number)}/pytest-results"
    )
    retrieve_test_report(cur_build_s3_link, PR_TEST_REPORT_DIR)

    common_build_s3_link = f"s3://{s3_prefix}/tvm/main/{common_main_build}/pytest-results"
    retrieve_test_report(common_build_s3_link, MAIN_TEST_REPORT_DIR)


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


def build_comment(
    common_commit_sha,
    common_main_build,
    skipped_list,
    pr_number,
    build_number,
    commit_sha,
    jenkins_prefix,
):
    if common_main_build["state"] != "success":
        return f"{SKIPPED_TESTS_COMMENT_MARKER}Unable to run tests bot because main failed to pass CI at {common_commit_sha}."

    if len(skipped_list) == 0:
        return f"{SKIPPED_TESTS_COMMENT_MARKER}No additional skipped tests found in this branch for commit {commit_sha}."

    text = (
        f"{SKIPPED_TESTS_COMMENT_MARKER}The list below shows some tests that ran in main {common_commit_sha} but were "
        f"skipped in the CI build of {commit_sha}:\n"
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
    parser.add_argument("--common-main-build")
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
        github = GitHubRepo(token=os.environ["GITHUB_TOKEN"], user=user, repo=repo)
        common_commit_sha = get_common_commit_sha()
        common_main_build = get_main_jenkins_build_number(github, common_commit_sha)
        retrieve_test_reports(
            common_main_build=common_main_build["build_number"],
            pr_number=pr_and_build["pr_number"],
            build_number=pr_and_build["build_number"],
            s3_prefix=args.s3_prefix,
        )
    else:
        assert args.common_main_build is not None
        common_main_build = json.loads(args.common_main_build)
        common_commit_sha = os.environ["COMMIT_SHA"]

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

    # Sort the list to maintain an order in the output. Helps when validating the output in tests.
    skipped_list.sort()

    if len(skipped_list) == 0:
        logging.info("No skipped tests found.")

    body = build_comment(
        common_commit_sha,
        common_main_build,
        skipped_list,
        pr_and_build["pr_number"],
        pr_and_build["build_number"],
        commit_sha,
        args.jenkins_prefix,
    )
    url = f'issues/{pr_and_build["pr_number"]}/comments'
    if not args.dry_run:
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
            comment_id = comment_url[comment_url.find("comments/") : len(comment_url)].strip(
                "comments/"
            )
            github.patch(f"issues/comments/{comment_id}", {"body": body})
        else:
            github.post(url, {"body": body})
    else:
        logging.info(f"Dry run, would have posted {url} with data {body}.")
