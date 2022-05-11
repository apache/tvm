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
import subprocess
import json
import logging
from typing import Optional

from git_utils import git
from cmd_utils import Sh, init_log


if __name__ == "__main__":
    init_log()
    parser = argparse.ArgumentParser(description="Remove all non-matching Docker images")
    parser.add_argument("--names", required=True, help="comma separated list of Docker image specs to keep")
    parser.add_argument("--dry-run", action='store_true', help="don't remove anything")
    args = parser.parse_args()
    to_keep = [x.strip() for x in args.names.split(",") if x.strip() != ""]
    logging.info(f"Keeping images: {to_keep}")

    sh = Sh()
    r = sh.run("docker image ls --all --format '{{json .}}'", stdout=subprocess.PIPE, encoding="utf-8")
    images = [json.loads(line) for line in r.stdout.splitlines()]
    to_remove = []
    kept_images = []
    for image in images:
        spec = f"{image['Repository']}:{image['Tag']}"
        if any([x in spec for x in to_keep]):
            kept_images.append(image)
        else:
            to_remove.append(image)
        
    
    logging.info(f"Removing {len(to_remove)} images of {len(images)} total")
    kept_names = [f"{image['Repository']}:{image['Tag']}" for image in kept_images]
    logging.info(f"Kept {len(kept_images)} images of {len(images)} total ({', '.join(kept_names)})")

    for image in to_remove:
        cmd = f"docker rmi {image['ID']}"
        if args.dry_run:
            logging.info(f"Would have run: {cmd}")
        else:
            sh.run(cmd)
