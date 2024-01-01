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

"""
This is the global script that set the version information of TVM.
This script runs and update all the locations that related to versions

List of affected files:
- tvm-root/python/tvm/_ffi/libinfo.py
- tvm-root/include/tvm/runtime/c_runtime_api.h
- tvm-root/conda/recipe/meta.yaml
- tvm-root/web/package.json
"""
import os
import re
import argparse
import logging
import subprocess

# Modify the following value during release
# ---------------------------------------------------
# Current version:
# We use the version of the incoming release for code
# that is under development.
#
# It is also fallback version to be used when --git-describe
# is not invoked, or when the repository does not present the
# git tags in a format that this script can use.
#
# Two tag formats are supported:
# - vMAJ.MIN.PATCH (e.g. v0.8.0) or
# - vMAJ.MIN.devN (e.g. v0.8.dev0)
__version__ = "0.16.dev0"

# ---------------------------------------------------

PROJ_ROOT = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


def py_str(cstr):
    return cstr.decode("utf-8")


def git_describe_version():
    """Get PEP-440 compatible public and local version using git describe.

    Returns
    -------
    pub_ver: str
        Public version.

    local_ver: str
        Local version (with additional label appended to pub_ver).

    Notes
    -----
    - We follow PEP 440's convention of public version
      and local versions.
    - Only tags conforming to vMAJOR.MINOR.REV (e.g. "v0.7.0")
      are considered in order to generate the version string.
      See the use of `--match` in the `git` command below.

    Here are some examples:

    - pub_ver = '0.7.0', local_ver = '0.7.0':
      We are at the 0.7.0 release.
    - pub_ver =  '0.8.dev94', local_ver = '0.8.dev94+g0d07a329e':
      We are at the 0.8 development cycle.
      The current source contains 94 additional commits
      after the most recent tag(v0.7.0),
      the git short hash tag of the current commit is 0d07a329e.
    """
    cmd = [
        "git",
        "describe",
        "--tags",
        "--match",
        "v[0-9]*.[0-9]*.[0-9]*",
        "--match",
        "v[0-9]*.[0-9]*.dev[0-9]*",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=PROJ_ROOT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = py_str(out)
        if msg.find("not a git repository") != -1:
            return __version__, __version__
        logging.warning("git describe: %s, use %s", msg, __version__)
        return __version__, __version__
    describe = py_str(out).strip()
    arr_info = describe.split("-")

    # Remove the v prefix, mainly to be robust
    # to the case where v is not presented as well.
    if arr_info[0].startswith("v"):
        arr_info[0] = arr_info[0][1:]

    # hit the exact tag
    if len(arr_info) == 1:
        return arr_info[0], arr_info[0]

    if len(arr_info) != 3:
        logging.warning("Invalid output from git describe %s", describe)
        return __version__, __version__

    dev_pos = arr_info[0].find(".dev")

    # Development versions:
    # The code will reach this point in case it can't match a full release version, such as v0.7.0.
    #
    # 1. in case the last known label looks like vMAJ.MIN.devN e.g. v0.8.dev0, we use
    # the current behaviour of just using vMAJ.MIN.devNNNN+gGIT_REV
    if dev_pos != -1:
        dev_version = arr_info[0][: arr_info[0].find(".dev")]
    # 2. in case the last known label looks like vMAJ.MIN.PATCH e.g. v0.8.0
    # then we just carry on with a similar version to what git describe provides, which is
    # vMAJ.MIN.PATCH.devNNNN+gGIT_REV
    else:
        dev_version = arr_info[0]

    pub_ver = "%s.dev%s" % (dev_version, arr_info[1])
    local_ver = "%s+%s" % (pub_ver, arr_info[2])
    return pub_ver, local_ver


# Implementations
def update(file_name, pattern, repl, dry_run=False):
    update = []
    hit_counter = 0
    need_update = False
    with open(file_name) as file:
        for l in file:
            result = re.findall(pattern, l)
            if result:
                assert len(result) == 1
                hit_counter += 1
                if result[0] != repl:
                    l = re.sub(pattern, repl, l)
                    need_update = True
                    print("%s: %s -> %s" % (file_name, result[0], repl))
                else:
                    print("%s: version is already %s" % (file_name, repl))

            update.append(l)
    if hit_counter != 1:
        raise RuntimeError("Cannot find version in %s" % file_name)

    if need_update and not dry_run:
        with open(file_name, "w") as output_file:
            for l in update:
                output_file.write(l)


def sync_version(pub_ver, local_ver, dry_run):
    """Synchronize version."""
    # python uses the PEP-440: local version
    update(
        os.path.join(PROJ_ROOT, "python", "tvm", "_ffi", "libinfo.py"),
        r"(?<=__version__ = \")[.0-9a-z\+]+",
        local_ver,
        dry_run,
    )
    # Use public version for other parts for now
    # Note that full git hash is already available in libtvm
    # C++ header
    update(
        os.path.join(PROJ_ROOT, "include", "tvm", "runtime", "c_runtime_api.h"),
        r'(?<=TVM_VERSION ")[.0-9a-z\+]+',
        pub_ver,
        dry_run,
    )
    # conda
    update(
        os.path.join(PROJ_ROOT, "conda", "recipe", "meta.yaml"),
        r"(?<=version = ')[.0-9a-z\+]+",
        pub_ver,
        dry_run,
    )
    # web
    # change to pre-release convention by npm
    dev_pos = pub_ver.find(".dev")
    npm_ver = pub_ver if dev_pos == -1 else "%s.0-%s" % (pub_ver[:dev_pos], pub_ver[dev_pos + 1 :])
    update(
        os.path.join(PROJ_ROOT, "web", "package.json"),
        r'(?<="version": ")[.0-9a-z\-\+]+',
        npm_ver,
        dry_run,
    )


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Detect and synchronize version.")
    parser.add_argument(
        "--print-version",
        action="store_true",
        help="Print version to the command line. No changes is applied to files.",
    )
    parser.add_argument(
        "--git-describe",
        action="store_true",
        help="Use git describe to generate development version.",
    )
    parser.add_argument("--dry-run", action="store_true")

    opt = parser.parse_args()
    pub_ver, local_ver = __version__, __version__
    if opt.git_describe:
        pub_ver, local_ver = git_describe_version()
    if opt.print_version:
        print(local_ver)
    else:
        sync_version(pub_ver, local_ver, opt.dry_run)


if __name__ == "__main__":
    main()
