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
Test the conversion of bash to rst
"""

import sys

import tvm
from tvm.contrib import utils

# this has to be after the sys.path patching, so ignore pylint
# pylint: disable=wrong-import-position,wrong-import-order
from .test_utils import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "docs"))
from script_convert import bash_to_python, BASH, BASH_IGNORE, BASH_MULTILINE_COMMENT

# pylint: enable=wrong-import-position,wrong-import-order


def test_bash_cmd():
    """Test that a bash command gets turned into a rst code block"""
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

    expected_cmd = "# .. code-block:: bash\n" "#\n" "# \t  tvmc\n" "#\n"

    assert generated_cmd == expected_cmd


def test_bash_ignore_cmd():
    """Test that ignored bash commands are not turned into code blocks"""
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
    """Test a file with no code blocks"""
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
    """Test a file with a bash code block"""
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

    expected_cmd = "# start\n" "# .. code-block:: bash\n" "#\n" "# \t  tvmc\n" "#\n" "# end\n"

    assert generated_cmd == expected_cmd


def test_last_line_break():
    """Test that line endings are correct"""
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
    """Test that bash comments are inserted correctly"""
    temp = utils.tempdir()
    src_path = temp / "src.sh"
    dest_path = temp / "dest.py"

    with open(src_path, "w") as src_f:
        src_f.write(BASH_MULTILINE_COMMENT)
        src_f.write("\n")
        src_f.write('# """\n')
        src_f.write("# comment\n")
        src_f.write(BASH_MULTILINE_COMMENT)
        src_f.write("\n")

    bash_to_python(src_path, dest_path)

    with open(dest_path, "r") as dest_f:
        generated_cmd = dest_f.read()

    expected_cmd = '"""\ncomment\n'

    assert generated_cmd == expected_cmd


if __name__ == "__main__":
    tvm.testing.main()
