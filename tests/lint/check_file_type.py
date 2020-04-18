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
import os
import sys
import subprocess

# List of file types we allow
ALLOW_EXTENSION = {
    # source code
    "cc",
    "c",
    "h",
    "s",
    "rs",
    "m",
    "mm",
    "g4",
    "gradle",
    "js",
    "tcl",
    "scala",
    "java",
    "go",
    "sh",
    "py",
    "pyi",
    "pxi",
    "pyd",
    "pyx",
    "cu",
    # relay text format
    "rly",
    # configurations
    "mk",
    "in",
    "cmake",
    "xml",
    "toml",
    "yml",
    "yaml",
    "json",
    # docs
    "txt",
    "md",
    "rst",
    # sgx
    "edl",
    "lds",
    # ios
    "pbxproj",
    "plist",
    "xcworkspacedata",
    "storyboard",
    # hw/chisel
    "sbt",
    "properties",
    "v",
    "sdc",
    # generated parser
    "interp",
    "tokens",
    # interface definition
    "idl",
    }

# List of file names allowed
ALLOW_FILE_NAME = {
    ".gitignore",
    ".gitattributes",
    "README",
    "Makefile",
    "Doxyfile",
    "pylintrc",
    "rat-excludes",
    "log4j.properties",
    ".clang-format",
    ".gitmodules",
    "CODEOWNERS",
    ".scalafmt.conf",
    "Cargo.lock",
    "with_the_same_user",
   }

# List of specific files allowed in relpath to <proj_root>
ALLOW_SPECIFIC_FILE = {
    "LICENSE",
    "NOTICE",
    "KEYS",
    "DISCLAIMER",
    "Jenkinsfile",
    # sgx config
    "apps/sgx/.cargo/config",
    # html for demo purposes
    "tests/webgl/test_static_webgl_library.html",
    "web/example_rpc.html",
    # images are normally not allowed
    # discuss with committers before add more images
    "apps/android_rpc/app/src/main/res/mipmap-hdpi/ic_launcher.png",
    "apps/android_rpc/app/src/main/res/mipmap-mdpi/ic_launcher.png",
    # documentation related files
    "docs/_static/css/tvm_theme.css",
    "docs/_static/img/tvm-logo-small.png",
    "docs/_static/img/tvm-logo-square.png",
   }


def filename_allowed(name):
    """Check if name is allowed by the current policy.

    Paramaters
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

    if os.path.basename(name) in ALLOW_FILE_NAME:
        return True

    if os.path.basename(name).startswith("Dockerfile"):
        return True

    if name.startswith("3rdparty"):
        return True

    if name in ALLOW_SPECIFIC_FILE:
        return True

    return False


def copyright_line(line):
    # Following two items are intentionally break apart
    # so that the copyright detector won't detect the file itself.
    if line.find("Copyright " + "(c)") != -1:
        return True
    if (line.find("Copyright") != -1 and
        line.find(" by") != -1):
        return True
    return False


def check_asf_copyright(fname):
    if fname.endswith(".png"):
        return True
    if not os.path.isfile(fname):
        return True
    has_asf_header = False
    has_copyright = False
    try:
        for line in open(fname):
            if line.find("Licensed to the Apache Software Foundation") != -1:
                has_asf_header = True
            if copyright_line(line):
                has_copyright = True
            if has_asf_header and has_copyright:
                return False
    except UnicodeDecodeError:
        pass
    return True


def main():
    cmd = ["git", "ls-files"]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    assert proc.returncode == 0
    res = out.decode("utf-8")
    flist = res.split()
    error_list = []

    for fname in flist:
        if not filename_allowed(fname):
            error_list.append(fname)

    if error_list:
        report = "------File type check report----\n"
        report += "\n".join(error_list)
        report += "\nFound %d files that are now allowed\n" % len(error_list)
        report += ("We do not check in binary files into the repo.\n"
                   "If necessary, please discuss with committers and"
                   "modify tests/lint/check_file_type.py to enable the file you need.\n")
        sys.stderr.write(report)
        sys.stderr.flush()
        sys.exit(-1)

    asf_copyright_list = []

    for fname in res.split():
        if not check_asf_copyright(fname):
            asf_copyright_list.append(fname)

    if asf_copyright_list:
        report = "------File type check report----\n"
        report += "\n".join(asf_copyright_list) + "\n"
        report += "------Found %d files that has ASF header with copyright message----\n" % len(asf_copyright_list)
        report += "--- Files with ASF header do not need Copyright lines.\n"
        report += "--- Contributors retain copyright to their contribution by default.\n"
        report += "--- If a file comes with a different license, consider put it under the 3rdparty folder instead.\n"
        report += "---\n"
        report += "--- You can use the following steps to remove the copyright lines\n"
        report += "--- Create file_list.txt in your text editor\n"
        report += "--- Copy paste the above content in file-list into file_list.txt\n"
        report += "--- python3 tests/lint/add_asf_header.py file_list.txt\n"
        sys.stderr.write(report)
        sys.stderr.flush()
        sys.exit(-1)

    print("check_file_type.py: all checks passed..")

if __name__ == "__main__":
    main()
