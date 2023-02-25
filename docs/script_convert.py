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
import pathlib

BASH = "# bash"
BASH_IGNORE = "# bash-ignore"
BASH_MULTILINE_COMMENT = "# bash-comment"


def bash_to_python(src_path: pathlib.Path, dest_path: pathlib.Path):
    """Convert a bash script file to a Python format compatible with Sphinx doc."""
    with open(src_path, "r") as src_f:
        with open(dest_path, "w") as dest_f:
            line = src_f.readline()
            bash_block = []
            bash_detected = False
            bash_ignore_detected = False
            new_line_required = False
            bash_multiline_comment_detected = False

            while line:
                line = line.strip("\n").strip("\r")
                if bash_detected:
                    if line == BASH:
                        # write the bash block to destination
                        if new_line_required:
                            dest_f.write("\n")
                        python_code = "# .. code-block:: bash\n#\n"
                        for bash_line in bash_block:
                            python_code += f"# \t  {bash_line}\n"
                        python_code += "#"
                        dest_f.write(python_code)

                        bash_detected = False
                        bash_block = []
                        new_line_required = True
                    else:
                        # add new bash command line
                        bash_block.append(line)
                elif bash_ignore_detected:
                    if line == BASH_IGNORE:
                        bash_ignore_detected = False
                    else:
                        new_line_required = False
                        pass
                elif bash_multiline_comment_detected:
                    if line == BASH_MULTILINE_COMMENT:
                        bash_multiline_comment_detected = False
                    else:
                        if line != "#":
                            assert len(line) > 2, "Detected empty line."
                            dest_f.write(line[2:])
                        new_line_required = True
                else:
                    if line == BASH:
                        bash_detected = True
                    elif line == BASH_IGNORE:
                        bash_ignore_detected = True
                    elif line == BASH_MULTILINE_COMMENT:
                        bash_multiline_comment_detected = True
                    else:
                        if new_line_required:
                            dest_f.write("\n")
                        dest_f.write(f"{line}")
                        new_line_required = True

                line = src_f.readline()
                if new_line_required:
                    dest_f.write("\n")
                    new_line_required = False


def main():
    parser = argparse.ArgumentParser(description="Convert tutorial script to Python.")
    parser.add_argument("script", type=str, help="Path to script file.")
    args = parser.parse_args()

    src_path = pathlib.Path(args.script)
    dest_path = src_path.parent / f"{src_path.stem}.py"
    bash_to_python(src_path, dest_path)


if __name__ == "__main__":
    main()
