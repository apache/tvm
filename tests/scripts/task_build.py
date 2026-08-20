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
# ruff: noqa: E402
import argparse
import json
import logging
import multiprocessing
import os
import re
import shutil
import sys
from pathlib import Path

# Hackery to enable importing of utils from ci/scripts/jenkins
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT / "ci" / "scripts" / "jenkins"))
from cmd_utils import REPO_ROOT, Sh, init_log


def _check_cxx_standard(build_dir):
    compile_commands_path = REPO_ROOT / build_dir / "compile_commands.json"
    with open(compile_commands_path, encoding="utf-8") as compile_commands_file:
        compile_commands = json.load(compile_commands_file)

    standard_flag_pattern = re.compile(r"^(?:-std=(?:c|gnu)\+\+|/std:c\+\+)(.+)$", re.IGNORECASE)
    tvm_ffi_commands = []
    tvm_commands = []

    for compile_command in compile_commands:
        source = Path(compile_command["file"])
        if not source.is_absolute():
            source = Path(compile_command["directory"]) / source
        try:
            source = source.resolve().relative_to(REPO_ROOT)
        except ValueError:
            continue

        if source.suffix.lower() not in {".cc", ".cpp", ".cxx"}:
            continue

        arguments = compile_command.get("arguments")
        if arguments is None:
            arguments = compile_command["command"].split()
        standard_flags = []
        for argument in arguments:
            argument = argument.strip("\"'")
            match = standard_flag_pattern.match(argument)
            if match:
                standard_flags.append((argument, match.group(1).lower()))

        if source.parts[:2] == ("3rdparty", "tvm-ffi"):
            tvm_ffi_commands.append((source, standard_flags, compile_command))
        elif source.parts[0] == "src":
            tvm_commands.append((source, standard_flags, compile_command))

    if not tvm_ffi_commands:
        raise RuntimeError("No C++ compile commands found for 3rdparty/tvm-ffi")
    if not tvm_commands:
        raise RuntimeError("No C++ compile commands found for TVM src/")

    for source, standard_flags, compile_command in tvm_ffi_commands:
        if standard_flags and standard_flags[-1][1] not in {"17", "1z"}:
            raise RuntimeError(
                f"tvm-ffi source {source} must use C++17, but its compile command contains "
                f"{', '.join(flag for flag, _ in standard_flags)}: {compile_command}"
            )

    for source, standard_flags, compile_command in tvm_commands:
        if not standard_flags or standard_flags[-1][1] not in {"20", "2a"}:
            raise RuntimeError(
                f"TVM source {source} must use C++20, but its compile command is: {compile_command}"
            )

    logging.info(
        "Verified C++ standards in %s tvm-ffi and %s TVM compile commands",
        len(tvm_ffi_commands),
        len(tvm_commands),
    )


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

    if args.debug:
        sh.run("cmake -GNinja -DCMAKE_BUILD_TYPE=Debug ..", cwd=build_dir)
    else:
        sh.run("cmake -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo ..", cwd=build_dir)

    _check_cxx_standard(build_dir)

    target = ""
    if args.cmake_target:
        target = args.cmake_target

    verbose = os.environ.get("VERBOSE", "true").lower() in {"1", "true", "yes"}
    ninja_args = [target, f"-j{num_cpus}"]
    if verbose:
        ninja_args.append("-v")

    sh.run("cmake --build . -- " + " ".join(ninja_args), cwd=build_dir)

    if use_sccache:
        logging.info("===== sccache stats =====")
        sh.run("sccache --show-stats")
