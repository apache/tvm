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
"""Helper tool to check file types that are allowed to checkin."""

import subprocess
import sys
from pathlib import Path

# List of file types we allow
ALLOW_EXTENSION = {
    # source code
    "cc",
    "c",
    "h",
    "S",
    "cu",
    "cuh",
    "m",
    "mm",
    "java",
    "gradle",
    "groovy",
    "js",
    "cjs",
    "mjs",
    "ts",
    "sh",
    "py",
    # configurations
    "mk",
    "in",
    "cmake",
    "xml",
    "toml",
    "yml",
    "yaml",
    "json",
    "cfg",
    "ini",
    # docs
    "txt",
    "md",
    "rst",
    "css",
    "html",
    # ios
    "pbxproj",
    "plist",
    "xcworkspacedata",
    "storyboard",
    "xcscheme",
    # interface definition
    "idl",
    # Jinja2 templates
    "j2",
    # images
    "png",
    # misc
    "properties",
    "template",
}

# List of file names allowed
ALLOW_FILE_NAME = {
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
    ".clang-format",
    "README",
    "Makefile",
    "Doxyfile",
    "CODEOWNERSHIP",
    "condarc",
    "with_the_same_user",
}

# List of specific files allowed in relpath to <proj_root>
ALLOW_SPECIFIC_FILE = {"LICENSE", "NOTICE", "KEYS"}


def filename_allowed(name: str) -> bool:
    """Check if name is allowed by the current policy.

    Parameters
    ----------
    name : str
        Input name

    Returns
    -------
    allowed : bool
        Whether the filename is allowed.
    """
    arr = name.rsplit(".", 1)
    if arr[-1] in ALLOW_EXTENSION:
        return True

    basename = Path(name).name
    if basename in ALLOW_FILE_NAME:
        return True

    # Dockerfile and variants like Dockerfile.ci_gpu
    if basename == "Dockerfile" or basename.startswith("Dockerfile."):
        return True

    if name.startswith("3rdparty"):
        return True

    if name in ALLOW_SPECIFIC_FILE:
        return True

    return False


def main() -> None:
    cmd = ["git", "ls-files"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    res = out.decode("utf-8")
    assert proc.returncode == 0, f"{' '.join(cmd)} errored: {res}"
    error_list = [f for f in res.split() if not filename_allowed(f)]

    if error_list:
        report = "------File type check report----\n"
        report += "\n".join(error_list)
        report += f"\nFound {len(error_list)} files that are not allowed\n"
        report += "We do not check in binary files into the repo.\n"
        report += (
            "If necessary, please discuss with committers and "
            "modify tests/lint/check_file_type.py to enable the file you need.\n"
        )
        sys.stderr.write(report)
        sys.stderr.flush()
        sys.exit(-1)

    print("check_file_type.py: all checks passed..")


if __name__ == "__main__":
    main()
