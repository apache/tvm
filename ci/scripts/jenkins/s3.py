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
from pathlib import Path

from cmd_utils import Sh, REPO_ROOT, init_log

RETRY_SCRIPT = REPO_ROOT / "ci" / "scripts" / "jenkins" / "retry.sh"


def show_md5(item: str) -> None:
    if not Path(item).is_dir():
        sh.run(f"md5sum {item}")


def is_file(bucket: str, prefix: str) -> bool:
    cmd = f"aws s3api head-object --bucket {bucket} --key {prefix}"
    proc = Sh().run(cmd, check=False)
    return proc.returncode == 0


if __name__ == "__main__":
    init_log()
    help = "Uploads or downloads files from S3"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--action", help="either 'upload' or 'download'", required=True)
    parser.add_argument("--bucket", help="s3 bucket", required=True)
    parser.add_argument(
        "--prefix", help="s3 bucket + tag (e.g. s3://tvm-ci-prod/PR-1234/cpu", required=True
    )
    parser.add_argument("--items", help="files and folders to upload", nargs="+", required=True)

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
        upload = True
    elif args.action == "download":
        upload = False
    else:
        logging.error(f"Unsupported action: {args.action}")
        exit(1)

    for item in args.items:
        destination = s3_path + "/" + item
        recursive_arg = ""

        if upload:
            # Show md5 before uploading
            show_md5(item)

        # Download/upload the item
        cmd = f". {RETRY_SCRIPT.relative_to(REPO_ROOT)} && retry 3 aws s3 cp --no-progress"
        if upload and Path(item).is_dir():
            cmd += " --recursive"
        if not upload and not is_file(args.bucket, prefix + "/" + item):
            cmd += " --recursive"

        if upload:
            cmd += f" {item} {destination}"
        else:
            cmd += f" {destination} {item}"
        sh.run(cmd)

        if not upload:
            to_chmod = []
            print(item)
            if Path(item).is_dir():
                to_chmod = [f for f in Path(item).glob("**/*") if not f.is_dir()]
            else:
                to_chmod = [item]

            # Add execute bit for downloads
            to_chmod = [str(f) for f in to_chmod]
            logging.info(f"Adding execute bit for files: {to_chmod}")
            if len(to_chmod) > 0:
                sh.run(f"chmod +x {' '.join(to_chmod)}")

        if not upload:
            # Show md5 after downloading
            show_md5(item)
