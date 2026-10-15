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
import multiprocessing
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(filename)s:%(lineno)d %(levelname)s] %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser(
        description="Build TVM with optional settings from the CMAKE_ARGS environment variable"
    )
    parser.add_argument("--sccache-bucket", required=False, help="sccache bucket name")
    parser.add_argument("--sccache-region", required=False, help="sccache region")
    parser.add_argument("--build-dir", default="build", help="build folder")
    parser.add_argument(
        "--cmake-target", action="append", default=[], help="build target (repeatable)"
    )
    parser.add_argument(
        "--ffi-only", action="store_true", help="prepare the source-pinned FFI package"
    )
    parser.add_argument("--debug", required=False, action="store_true", help="build in debug mode")
    args = parser.parse_args()
    sccache_exe = shutil.which("sccache")

    if args.cmake_target and all(
        target in {"standalone_crt", "crttest"} for target in args.cmake_target
    ):
        logging.info("Skipping standalone_crt build")
        exit(0)

    use_sccache = sccache_exe is not None
    build_dir = Path(os.getcwd()) / args.build_dir
    build_dir = build_dir.relative_to(REPO_ROOT)
    env = os.environ.copy()
    # CMAKE_ARGS configures TVM, not the separate source-FFI Python package.
    cmake_args = shlex.split(env.pop("CMAKE_ARGS", ""))
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO_ROOT / "tirx" / "python"),
            str(REPO_ROOT / "python"),
            os.environ.get("PYTHONPATH", ""),
        ]
    )

    if use_sccache:
        if args.sccache_bucket and "AWS_ACCESS_KEY_ID" in os.environ:
            env["SCCACHE_BUCKET"] = args.sccache_bucket
            env["SCCACHE_REGION"] = args.sccache_region if args.sccache_region else "us-west-2"
            logging.info(f"Using sccache bucket: {args.sccache_bucket}")
            logging.info(f"Using sccache region: {env['SCCACHE_REGION']}")
        else:
            logging.info("No sccache bucket set, using local cache")
        if "SCCACHE_SERVER_PORT" in os.environ:
            env["SCCACHE_SERVER_PORT"] = os.getenv("SCCACHE_SERVER_PORT")
        env["CXX"] = "/opt/sccache/c++"
        env["CC"] = "/opt/sccache/cc"

    else:
        if sccache_exe is None:
            reason = "'sccache' executable not found"
        else:
            reason = "<unknown>"
        logging.info(f"Not using sccache, reason: {reason}")

    # Build from this checkout's pinned submodule once, then transfer the exact
    # Python/native package together with the TVM libraries to test workers.
    ffi_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT / "tirx" / "3rdparty" / "tvm-ffi"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    ffi_identity = f"{ffi_commit} {sys.implementation.cache_tag}"
    ffi_marker = REPO_ROOT / "python" / "tvm_ffi" / ".source-commit"
    if not ffi_marker.is_file() or ffi_marker.read_text().strip() != ffi_identity:
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "-v",
                "--no-deps",
                "--reinstall",
                "--target=python",
                "./tirx/3rdparty/tvm-ffi/",
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )
        ffi_marker.write_text(ffi_identity + "\n")
    if args.ffi_only:
        sys.exit(0)

    if use_sccache:
        subprocess.run(["sccache", "--start-server"], env=env, check=False)
        logging.info("===== sccache stats =====")
        subprocess.run(["sccache", "--show-stats"], env=env, check=True)

    executors = int(os.environ.get("CI_NUM_EXECUTORS", 1))
    build_platform = os.environ.get("PLATFORM", None)

    nproc = multiprocessing.cpu_count()

    available_cpus = nproc // executors
    num_cpus = max(available_cpus, 1)

    build_type = "Debug" if args.debug else "RelWithDebInfo"
    subprocess.run(
        ["cmake", "-GNinja", f"-DCMAKE_BUILD_TYPE={build_type}", *cmake_args, ".."],
        cwd=build_dir,
        env=env,
        check=True,
    )

    ninja_args = [*args.cmake_target, f"-j{num_cpus}"]
    if os.environ.get("VERBOSE", "true").lower() in {"1", "true", "yes"}:
        ninja_args.append("-v")
    subprocess.run(["cmake", "--build", ".", "--", *ninja_args], cwd=build_dir, env=env, check=True)

    if use_sccache:
        logging.info("===== sccache stats =====")
        subprocess.run(["sccache", "--show-stats"], env=env, check=True)
