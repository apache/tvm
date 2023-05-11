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
import logging
import datetime
import os
import json
import re
import shlex
from urllib import error
from typing import List, Dict, Any, Optional, Callable
from git_utils import git, parse_remote, GitHubRepo
from cmd_utils import REPO_ROOT, init_log
from should_rebuild_docker import docker_api

JENKINS_DIR = REPO_ROOT / "ci" / "jenkins"
IMAGES_FILE = JENKINS_DIR / "data.py"
GENERATE_SCRIPT = JENKINS_DIR / "generate.py"
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
BRANCH = "nightly-docker-update"


def _testing_docker_api(data: Dict[str, Any]) -> Callable[[str], Dict[str, Any]]:
    """Returns a function that can be used in place of docker_api"""

    def mock(url: str) -> Dict[str, Any]:
        if url in data:
            return data[url]
        else:
            raise error.HTTPError(url, 404, f"Not found: {url}", {}, None)

    return mock


def parse_docker_date(d: str) -> datetime.datetime:
    """Turn a date string from the Docker API into a datetime object"""
    return datetime.datetime.strptime(d, "%Y-%m-%dT%H:%M:%S.%fZ")


def check_tag(tag: Dict[str, Any]) -> bool:
    return re.match(r"^[0-9]+-[0-9]+-[a-z0-9]+$", tag["name"]) is not None


def latest_tag(user: str, repo: str) -> List[Dict[str, Any]]:
    """
    Queries Docker Hub and finds the most recent tag for the specified image/repo pair
    """
    r = docker_api(f"repositories/{user}/{repo}/tags")
    results = r["results"]

    for result in results:
        result["last_updated"] = parse_docker_date(result["last_updated"])

    results = list(sorted(results, key=lambda d: d["last_updated"]))
    results = [tag for tag in results if check_tag(tag)]
    return results[-1]


def latest_tlcpackstaging_image(source: str) -> Optional[str]:
    """
    Finds the latest full tag to use in the Jenkinsfile or returns None if no
    update is needed
    """
    name, current_tag = source.split(":")
    user, repo = name.split("/")
    logging.info(
        f"Running with name: {name}, current_tag: {current_tag}, user: {user}, repo: {repo}"
    )

    staging_repo = repo.replace("-", "_")
    latest_tlcpackstaging_tag = latest_tag(user="tlcpackstaging", repo=staging_repo)
    logging.info(f"Found latest tlcpackstaging tag:\n{latest_tlcpackstaging_tag}")

    if latest_tlcpackstaging_tag["name"] == current_tag:
        logging.info(f"tlcpackstaging tag is the same as the one in the Jenkinsfile")

    latest_tlcpack_tag = latest_tag(user="tlcpack", repo=repo)
    logging.info(f"Found latest tlcpack tag:\n{latest_tlcpack_tag}")

    if latest_tlcpack_tag["name"] == latest_tlcpackstaging_tag["name"]:
        logging.info("Tag names were the same, no update needed")
        return None

    if latest_tlcpack_tag["last_updated"] > latest_tlcpackstaging_tag["last_updated"]:
        new_spec = f"tlcpack/{repo}:{latest_tlcpack_tag['name']}"
    else:
        # Even if the image doesn't exist in tlcpack, it will fall back to tlcpackstaging
        # so hardcode the username here
        new_spec = f"tlcpack/{repo}:{latest_tlcpackstaging_tag['name']}"
        logging.info("Using tlcpackstaging tag on tlcpack")

    logging.info(f"Found newer image, using: {new_spec}")
    return new_spec


if __name__ == "__main__":
    init_log()
    help = "Open a PR to update the Docker images to use the latest available in tlcpackstaging"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--remote", default="origin", help="ssh remote to parse")
    parser.add_argument("--dry-run", action="store_true", help="don't send PR to GitHub")
    parser.add_argument("--testing-docker-data", help="JSON data to mock Docker Hub API response")
    args = parser.parse_args()

    # Install test mock if necessary
    if args.testing_docker_data is not None:
        docker_api = _testing_docker_api(data=json.loads(args.testing_docker_data))

    remote = git(["config", "--get", f"remote.{args.remote}.url"])
    user, repo = parse_remote(remote)

    # Read the existing images from the Jenkinsfile
    logging.info(f"Reading {IMAGES_FILE}")
    with open(IMAGES_FILE) as f:
        content = f.readlines()

    # Build a new Jenkinsfile with the latest images from tlcpack or tlcpackstaging
    replacements = {}

    for line in content:
        m = re.match(r'"tag": "(.*)",', line.strip())
        if m is not None:
            image_spec = m.groups()[0]
            logging.info(f"Found match on line {line.strip()}")
            new_image = latest_tlcpackstaging_image(image_spec)
            if new_image is None:
                logging.info(f"No new image found")
            else:
                logging.info(f"Using new image {new_image}")
                new_line = f'        "tag": "{new_image}",'
                replacements[line] = new_line

    # Re-generate the Jenkinsfiles
    command = f"python3 {shlex.quote(str(GENERATE_SCRIPT))}"

    content = "\n".join(content)
    for old_line, new_line in replacements.items():
        content = content.replace(old_line, new_line)

    print(f"Updated to:\n{content}")

    if args.dry_run:
        print(f"Would have run:\n{command}")
    else:
        with open(IMAGES_FILE, "w") as f:
            f.write(content)
        Sh().run(command)

    # Publish the PR
    title = "[ci][docker] Nightly Docker image update"
    body = "This bumps the Docker images to the latest versions from Docker Hub."
    message = f"{title}\n\n\n{body}"

    if args.dry_run:
        logging.info("Dry run, would have committed Jenkinsfiles")
    else:
        logging.info(f"Creating git commit")
        git(["checkout", "-B", BRANCH])
        git(["add", str(JENKINS_DIR.relative_to(REPO_ROOT))])
        git(["config", "user.name", "tvm-bot"])
        git(["config", "user.email", "95660001+tvm-bot@users.noreply.github.com"])
        git(["commit", "-m", message])
        git(["push", "--set-upstream", args.remote, BRANCH, "--force"])

    logging.info(f"Sending PR to GitHub")
    github = GitHubRepo(user=user, repo=repo, token=GITHUB_TOKEN)
    data = {
        "title": title,
        "body": body,
        "head": BRANCH,
        "base": "main",
        "maintainer_can_modify": True,
    }
    url = "pulls"
    if args.dry_run:
        logging.info(f"Dry run, would have sent {data} to {url}")
    else:
        try:
            github.post(url, data=data)
        except error.HTTPError as e:
            # Ignore the exception if the PR already exists (which gives a 422). The
            # existing PR will have been updated in place
            if e.code == 422:
                logging.info("PR already exists, ignoring error")
                logging.exception(e)
            else:
                raise e
