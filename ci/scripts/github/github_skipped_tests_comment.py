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
import inspect
import json
import os
import logging
import subprocess
from xml.etree import ElementTree
from pathlib import Path
from typing import Dict, Any, Optional


def run_subprocess(command):
    logging.info(f"Running command {command}")
    proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed {command}:\nstdout:\n{proc.stdout}")
    return proc


def retrieve_test_report(s3_url, target_dir):
    command = f"aws --region us-west-2 s3 cp {s3_url} {target_dir} --recursive --no-sign-request"
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


def retrieve_test_reports(
    common_main_build, pr_number, build_number, s3_prefix, pr_test_report_dir, main_test_report_dir
):
    cur_build_s3_link = (
        f"s3://{s3_prefix}/tvm/PR-{str(pr_number)}/{str(build_number)}/pytest-results"
    )
    retrieve_test_report(cur_build_s3_link, pr_test_report_dir)

    common_build_s3_link = f"s3://{s3_prefix}/tvm/main/{common_main_build}/pytest-results"
    retrieve_test_report(common_build_s3_link, main_test_report_dir)


def get_pr_and_build_numbers(target_url):
    target_url = target_url[target_url.find("PR-") : len(target_url)]
    split = target_url.split("/")
    pr_number = split[0].strip("PR-")
    build_number = split[1]
    return {"pr_number": pr_number, "build_number": build_number}


def build_test_set(directory):
    directory = Path(directory)
    subdir_to_skipped = {}
    subdirs = [
        item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))
    ]
    for subdir in subdirs:
        subdir_to_skipped[subdir] = set()
        for root, _, files in os.walk(directory / subdir):
            for file in files:
                test_report = ElementTree.parse(Path(root) / file)
                for testcase in test_report.iter("testcase"):
                    skipped = testcase.find("skipped")
                    if skipped is not None:
                        key = testcase.attrib["classname"] + "#" + testcase.attrib["name"]
                        subdir_to_skipped[subdir].add(key)
    return subdir_to_skipped


def to_node_name(dir_name: str):
    return dir_name.replace("_", ": ", 1)


def build_diff_comment_with_main(
    common_commit_sha,
    skipped_list,
    commit_sha,
):
    if len(skipped_list) == 0:
        return f"No diff in skipped tests with main found in this branch for commit {commit_sha}.\n"

    text = (
        f"The list below shows tests that ran in main {common_commit_sha} but were "
        f"skipped in the CI build of {commit_sha}:\n"
        f"```\n"
    )
    for skip in skipped_list:
        text += skip + "\n"
    text += f"```\n"
    return text


def build_comment(
    common_commit_sha,
    common_main_build,
    skipped_list,
    additional_skipped_list,
    pr_number,
    build_number,
    commit_sha,
    jenkins_prefix,
):
    if common_main_build["state"] != "success":
        return f"Unable to run tests bot because main failed to pass CI at {common_commit_sha}."

    text = build_diff_comment_with_main(common_commit_sha, skipped_list, commit_sha)

    if len(additional_skipped_list) != 0:
        text += "\n"
        text += (
            f"Additional tests that were skipped in the CI build and present in the [`required_tests_to_run`]"
            f"(https://github.com/apache/tvm/blob/main/ci/scripts/github/required_tests_to_run.json) file:"
            f"\n```\n"
        )
        for skip in additional_skipped_list:
            text += skip + "\n"
        text += f"```\n"

    text += (
        f"A detailed report of ran tests is [here](https://{jenkins_prefix}/job/tvm/job/PR-{str(pr_number)}"
        f"/{str(build_number)}/testReport/)."
    )
    return text


def find_target_url(pr_head: Dict[str, Any]):
    for status in pr_head["statusCheckRollup"]["contexts"]["nodes"]:
        if status.get("context", "") == "tvm-ci/pr-head":
            return status["targetUrl"]

    raise RuntimeError(f"Unable to find tvm-ci/pr-head status in {pr_head}")


def get_skipped_tests_comment(
    pr: Dict[str, Any],
    github,
    s3_prefix: str = "tvm-jenkins-artifacts-prod",
    jenkins_prefix: str = "ci.tlcpack.ai",
    pr_test_report_dir: str = "pr-reports",
    main_test_report_dir: str = "main-reports",
    common_commit_sha: Optional[str] = None,
    common_main_build: Optional[Dict[str, Any]] = None,
    additional_tests_to_check_file: str = "required_tests_to_run.json",
) -> str:
    pr_head = pr["commits"]["nodes"][0]["commit"]
    target_url = find_target_url(pr_head)
    pr_and_build = get_pr_and_build_numbers(target_url)
    logging.info(f"Getting comment for {pr_head} with target {target_url}")

    commit_sha = pr_head["oid"]

    is_dry_run = common_commit_sha is not None

    if not is_dry_run:
        logging.info("Fetching common commit sha and build info")
        common_commit_sha = get_common_commit_sha()
        common_main_build = get_main_jenkins_build_number(github, common_commit_sha)

        retrieve_test_reports(
            common_main_build=common_main_build["build_number"],
            pr_number=pr_and_build["pr_number"],
            build_number=pr_and_build["build_number"],
            s3_prefix=s3_prefix,
            main_test_report_dir=main_test_report_dir,
            pr_test_report_dir=pr_test_report_dir,
        )
    else:
        logging.info("Dry run, expecting PR and main reports on disk")

    main_tests = build_test_set(main_test_report_dir)
    build_tests = build_test_set(pr_test_report_dir)

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

    if not is_dry_run:
        current_file = Path(__file__).resolve()
        additional_tests_to_check_file = Path(current_file).parent / "required_tests_to_run.json"

    logging.info(
        f"Checking additional tests in file {additional_tests_to_check_file} are not skipped."
    )
    try:
        with open(additional_tests_to_check_file, "r") as f:
            additional_tests_to_check = json.load(f)
    except IOError:
        logging.info(
            f"Failed to read additional tests from file: {additional_tests_to_check_file}."
        )
        additional_tests_to_check = {}

    # Assert that tests present in "required_tests_to_run.json" are not skipped.
    additional_skipped_tests = []
    for subdir, test_set in additional_tests_to_check.items():
        if subdir not in build_tests.keys():
            logging.warning(f"Could not find directory {subdir} in the build test set.")
            continue

        for test in test_set:
            if test in build_tests[subdir]:
                additional_skipped_tests.append(f"{to_node_name(subdir)} -> {test}")

    if len(additional_skipped_tests) == 0:
        logging.info("No skipped tests found in the additional list.")

    body = build_comment(
        common_commit_sha,
        common_main_build,
        skipped_list,
        additional_skipped_tests,
        pr_and_build["pr_number"],
        pr_and_build["build_number"],
        commit_sha,
        jenkins_prefix,
    )

    return body
