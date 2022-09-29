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
import datetime
import json
import logging
import urllib.error
from pathlib import Path

from typing import Dict, Any


from http_utils import get
from cmd_utils import init_log

DOCKER_API_BASE = "https://hub.docker.com/v2/"
PAGE_SIZE = 25
TEST_DATA = None


def docker_api(url: str, use_pagination: bool = False) -> Dict[str, Any]:
    """
    Run a paginated fetch from the public Docker Hub API
    """
    if TEST_DATA is not None:
        if url not in TEST_DATA:
            raise urllib.error.HTTPError(url, 404, "Not found", {}, None)
        return TEST_DATA[url]
    pagination = ""
    if use_pagination:
        pagination = f"?page_size={PAGE_SIZE}&page=1"
    url = DOCKER_API_BASE + url + pagination
    r, headers = get(url)
    reset = headers.get("x-ratelimit-reset")
    if reset is not None:
        reset = datetime.datetime.fromtimestamp(int(reset))
        reset = reset.isoformat()
    logging.info(
        f"Docker API Rate Limit: {headers.get('x-ratelimit-remaining')} / {headers.get('x-ratelimit-limit')} (reset at {reset})"
    )
    return r


def image_exists(spec: str) -> bool:
    name, tag = spec.split(":")
    try:
        r = docker_api(f"repositories/{name}/tags/{tag}")
        logging.info(f"Image exists, got response: {json.dumps(r, indent=2)}")
        return True
    except urllib.error.HTTPError as e:
        # Image was not found
        logging.exception(e)
        return False


if __name__ == "__main__":
    init_log()
    parser = argparse.ArgumentParser(
        description="Writes out Docker images names to be used to .docker-image-names/"
    )
    parser.add_argument(
        "--testing-docker-data",
        help="(testing only) JSON data to mock response from Docker Hub API",
    )
    parser.add_argument(
        "--base-dir",
        default=".docker-image-names",
        help="(testing only) Folder to write image names to",
    )
    args, other = parser.parse_known_args()
    name_dir = Path(args.base_dir)

    images = {}
    for item in other:
        name, tag = item.split("=")
        images[name] = tag

    if args.testing_docker_data is not None:
        TEST_DATA = json.loads(args.testing_docker_data)

    logging.info(f"Checking if these images exist in tlcpack: {images}")

    name_dir.mkdir(exist_ok=True)
    images_to_use = {}
    for filename, spec in images.items():
        if image_exists(spec):
            logging.info(f"{spec} found in tlcpack")
            images_to_use[filename] = spec
        else:
            logging.info(f"{spec} not found in tlcpack, using tlcpackstaging")
            part, tag = spec.split(":")
            user, repo = part.split("/")
            tlcpackstaging_tag = f"tlcpackstaging/{repo.replace('-', '_')}:{tag}"
            images_to_use[filename] = tlcpackstaging_tag

    for filename, image in images_to_use.items():
        logging.info(f"Writing image {image} to {name_dir / filename}")
        with open(name_dir / filename, "w") as f:
            f.write(image)
