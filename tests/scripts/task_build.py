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
import shutil
import os
import logging
import sys
import multiprocessing

from pathlib import Path

# Hackery to enable importing of utils from ci/scripts/jenkins
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT / "ci" / "scripts" / "jenkins"))
from cmd_utils import Sh, init_log, REPO_ROOT

if __name__ == "__main__":
    init_log()

    parser = argparse.ArgumentParser(description="List pytest nodeids for a folder")
    parser.add_argument("--sccache-bucket", required=False, help="sccache bucket name")
    parser.add_argument("--sccache-region", required=False, help="sccache region")
    parser.add_argument("--build-dir", default="build", help="build folder")
    parser.add_argument("--cmake-target", help="optional build target")
    parser.add_argument("--debug", required=False, action="store_true", help="build in debug mode")
    args = parser.parse_args()
    sccache_exe = shutil.which("sccache")

    if args.cmake_target in ["standalone_crt", "crttest"]:
        logging.info("Skipping standalone_crt build")
        exit(0)

    use_sccache = sccache_exe is not None
    build_dir = Path(os.getcwd()) / args.build_dir
    build_dir = build_dir.relative_to(REPO_ROOT)
    env = {}

    if use_sccache:
        if args.sccache_bucket and "AWS_ACCESS_KEY_ID" in os.environ:
            env["SCCACHE_BUCKET"] = args.sccache_bucket
            env["SCCACHE_REGION"] = args.sccache_region if args.sccache_region else "us-west-2"
            logging.info(f"Using sccache bucket: {args.sccache_bucket}")
            logging.info(f"Using sccache region: {env['SCCACHE_REGION']}")
        else:
            logging.info(f"No sccache bucket set, using local cache")
        env["CXX"] = "/opt/sccache/c++"
        env["CC"] = "/opt/sccache/cc"

    else:
        if sccache_exe is None:
            reason = "'sccache' executable not found"
        else:
            reason = "<unknown>"
        logging.info(f"Not using sccache, reason: {reason}")

    sh = Sh(env)

    if use_sccache:
        sh.run("sccache --start-server", check=False)
        logging.info("===== sccache stats =====")
        sh.run("sccache --show-stats")

    executors = int(os.environ.get("CI_NUM_EXECUTORS", 1))
    build_platform = os.environ.get("PLATFORM", None)

    nproc = multiprocessing.cpu_count()

    available_cpus = nproc // executors
    num_cpus = max(available_cpus, 1)

    if build_platform == "i386":
        sh.run("cmake ..", cwd=build_dir)
    else:
        if args.debug:
            sh.run("cmake -GNinja -DCMAKE_BUILD_TYPE=Debug ..", cwd=build_dir)
        else:
            sh.run("cmake -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo ..", cwd=build_dir)

    target = ""
    if args.cmake_target:
        target = args.cmake_target

    verbose = os.environ.get("VERBOSE", "true").lower() in {"1", "true", "yes"}
    ninja_args = [target, f"-j{num_cpus}"]
    if verbose:
        ninja_args.append("-v")

    if build_platform == "i386":
        if args.cmake_target:
            sh.run(f"make {args.cmake_target} -j{num_cpus}", cwd=build_dir)
        else:
            sh.run(f"make -j{num_cpus}", cwd=build_dir)
    else:
        sh.run(f"cmake --build . -- " + " ".join(ninja_args), cwd=build_dir)

    if use_sccache:
        logging.info("===== sccache stats =====")
        sh.run("sccache --show-stats")
