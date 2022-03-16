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

import sys

import pytest

from tvm.contrib import utils

from test_utils import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "docs"))
from script_convert import (
    bash_to_python,
    BASH,
    BASH_IGNORE,
    BASH_MULTILINE_COMMENT_START,
    BASH_MULTILINE_COMMENT_END,
)


def test_bash_cmd():
    temp = utils.tempdir()
    src_path = temp / "src.sh"
    dest_path = temp / "dest.py"

    with open(src_path, "w") as src_f:
        src_f.write(BASH)
        src_f.write("\n")
        src_f.write("tvmc\n")
        src_f.write(BASH)
        src_f.write("\n")

    bash_to_python(src_path, dest_path)

    with open(dest_path, "r") as dest_f:
        generated_cmd = dest_f.read()

    expected_cmd = "# .. code-block:: bash\n" "#\n" "#\t  tvmc\n" "#\n"

    assert generated_cmd == expected_cmd


def test_bash_ignore_cmd():
    temp = utils.tempdir()
    src_path = temp / "src.sh"
    dest_path = temp / "dest.py"

    with open(src_path, "w") as src_f:
        src_f.write("# start\n")
        src_f.write(BASH_IGNORE)
        src_f.write("\n")
        src_f.write("tvmc\n")
        src_f.write(BASH_IGNORE)
        src_f.write("\n")
        src_f.write("# end\n")

    bash_to_python(src_path, dest_path)

    with open(dest_path, "r") as dest_f:
        generated_cmd = dest_f.read()

    expected_cmd = "# start\n" "# end\n"
    assert generated_cmd == expected_cmd


def test_no_command():
    temp = utils.tempdir()
    src_path = temp / "src.sh"
    dest_path = temp / "dest.py"

    with open(src_path, "w") as src_f:
        src_f.write("# start\n")
        src_f.write("# description\n")
        src_f.write("end\n")

    bash_to_python(src_path, dest_path)

    with open(dest_path, "r") as dest_f:
        generated_cmd = dest_f.read()

    expected_cmd = "# start\n" "# description\n" "end\n"
    assert generated_cmd == expected_cmd


def test_text_and_bash_command():
    temp = utils.tempdir()
    src_path = temp / "src.sh"
    dest_path = temp / "dest.py"

    with open(src_path, "w") as src_f:
        src_f.write("# start\n")
        src_f.write(BASH)
        src_f.write("\n")
        src_f.write("tvmc\n")
        src_f.write(BASH)
        src_f.write("\n")
        src_f.write("# end\n")

    bash_to_python(src_path, dest_path)

    with open(dest_path, "r") as dest_f:
        generated_cmd = dest_f.read()

    expected_cmd = "# start\n" "# .. code-block:: bash\n" "#\n" "#\t  tvmc\n" "#\n" "# end\n"

    assert generated_cmd == expected_cmd


def test_last_line_break():
    temp = utils.tempdir()
    src_path = temp / "src.sh"
    dest_path = temp / "dest.py"

    with open(src_path, "w") as src_f:
        src_f.write("# start\n")
        src_f.write("# end\n")

    bash_to_python(src_path, dest_path)

    with open(dest_path, "r") as dest_f:
        generated_cmd = dest_f.read()

    expected_cmd = "# start\n" "# end\n"

    assert generated_cmd == expected_cmd


def test_multiline_comment():
    temp = utils.tempdir()
    src_path = temp / "src.sh"
    dest_path = temp / "dest.py"

    with open(src_path, "w") as src_f:
        src_f.write(BASH_MULTILINE_COMMENT_START)
        src_f.write("\n")
        src_f.write("comment\n")
        src_f.write(BASH_MULTILINE_COMMENT_END)
        src_f.write("\n")

    bash_to_python(src_path, dest_path)

    with open(dest_path, "r") as dest_f:
        generated_cmd = dest_f.read()

    expected_cmd = '"""\n' "comment\n" '"""\n'

    assert generated_cmd == expected_cmd
