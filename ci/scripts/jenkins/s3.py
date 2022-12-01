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
import re
from pathlib import Path
from typing import List
from enum import Enum

from cmd_utils import Sh, REPO_ROOT, init_log

RETRY_SCRIPT = REPO_ROOT / "ci" / "scripts" / "jenkins" / "retry.sh"
S3_DOWNLOAD_REGEX = re.compile(r"download: s3://.* to (.*)")
SH = Sh()


class Action(Enum):
    UPLOAD = 1
    DOWNLOAD = 2


def show_md5(item: str) -> None:
    if not Path(item).is_dir():
        sh.run(f"md5sum {item}")


def parse_output_files(stdout: str) -> List[str]:
    """
    Grab the list of downloaded files from the output of 'aws s3 cp'. Lines look
    like:

        download: s3://some/prefix/a_file.txt to a_file.txt
    """
    files = []
    for line in stdout.split("\n"):
        line = line.strip()
        if line == "":
            continue
        m = S3_DOWNLOAD_REGEX.match(line)
        if m:
            files.append(m.groups()[0])

    return files


def chmod(files: List[str]) -> None:
    """
    S3 has no concept of file permissions so add them back in here to every file
    """
    # Add execute bit for downloads
    to_chmod = [str(f) for f in files]
    logging.info(f"Adding execute bit for files: {to_chmod}")
    if len(to_chmod) > 0:
        SH.run(f"chmod +x {' '.join(to_chmod)}")


def s3(source: str, destination: str, recursive: bool) -> List[str]:
    """
    Send or download the source to the destination in S3
    """
    cmd = f". {RETRY_SCRIPT.relative_to(REPO_ROOT)} && retry 3 aws s3 cp --no-progress"

    if recursive:
        cmd += " --recursive"

    cmd += f" {source} {destination}"
    _, stdout = SH.tee(cmd)
    return stdout


if __name__ == "__main__":
    init_log()
    help = "Uploads or downloads files from S3"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--action", help="either 'upload' or 'download'", required=True)
    parser.add_argument("--bucket", help="s3 bucket", required=True)
    parser.add_argument(
        "--prefix", help="s3 bucket + tag (e.g. s3://tvm-ci-prod/PR-1234/cpu", required=True
    )
    parser.add_argument("--items", help="files and folders to upload", nargs="+")

    args = parser.parse_args()
    logging.info(args)

    sh = Sh()

    if Path.cwd() != REPO_ROOT:
        logging.error(f"s3.py can only be executed from the repo root, instead was in {Path.cwd()}")
        exit(1)

    prefix = args.prefix.strip("/")
    s3_path = f"s3://{args.bucket}/{prefix}"
    logging.info(f"Using s3 path: {s3_path}")

    if args.action == "upload":
        action = Action.UPLOAD
    elif args.action == "download":
        action = Action.DOWNLOAD
    else:
        logging.error(f"Unsupported action: {args.action}")
        exit(1)

    if args.items is None:
        if args.action == "upload":
            logging.error(f"Cannot upload without --items")
            exit(1)
        else:
            # Download the whole prefix
            items = ["."]

    else:
        items = args.items

    for item in items:
        if action == Action.DOWNLOAD:
            source = s3_path
            recursive = True
            if item != ".":
                source = s3_path + "/" + item
                recursive = False
            stdout = s3(source=source, destination=item, recursive=recursive)
            files = parse_output_files(stdout)
            chmod(files)
            for file in files:
                # Show md5 after downloading
                show_md5(file)
        elif action == Action.UPLOAD:
            show_md5(item)
            s3(item, s3_path + "/" + item, recursive=Path(item).is_dir())
