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
import fnmatch
import re
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPECTED = """
# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore
""".rstrip()
IGNORE_PATTERNS = ["*/micro_tvmc.py", "*/micro_train.py"]
APACHE_HEADER_LINES = 16


def find_code_block_line(lines: List[str]) -> Optional[int]:
    """
    This returns the index in 'lines' of the first line of code in the tutorial
    or none if there are no code blocks.
    """
    in_multiline_string = False
    in_sphinx_directive = False

    i = 0
    lines = lines[APACHE_HEADER_LINES:]
    while i < len(lines):
        line = lines[i].strip()
        if '"""' in line:
            in_multiline_string = not in_multiline_string
        elif "# sphinx_gallery_" in line:
            in_sphinx_directive = not in_sphinx_directive
        elif line.startswith("#") or in_sphinx_directive or in_multiline_string or line == "":
            pass
        else:
            return i
        i += 1

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check that all tutorials/docs override urllib.request.Request"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Insert expected code into erroring files"
    )
    args = parser.parse_args()

    gallery_files = (REPO_ROOT / "gallery").glob("**/*.py")
    # gallery_files = [x for x in gallery_files if "cross_compi" in str(x)]

    errors = []
    for file in gallery_files:
        skip = False
        for ignored_file in IGNORE_PATTERNS:
            if fnmatch.fnmatch(str(file), ignored_file):
                skip = True
                break
        if skip:
            continue

        with open(file) as f:
            content = f.read()

        if EXPECTED not in content:
            errors.append((file, None))
            continue

        index = content.index(EXPECTED)
        line = content.count("\n", 0, index) + EXPECTED.count("\n") + 2
        expected = find_code_block_line(content.split("\n"))

        if expected is not None and line < expected:
            errors.append((file, (line, expected)))

    if args.fix:
        for error, line_info in errors:
            with open(error) as f:
                content = f.read()

            # Note: There must be a little bit of care taken here since inserting
            # the block between a comment and multiline string will lead to an
            # empty code block in the HTML output
            if "from __future__" in content:
                # Place after the last __future__ import
                new_content = re.sub(
                    r"((?:from __future__.*?\n)+)", r"\1\n" + EXPECTED, content, flags=re.MULTILINE
                )
            else:
                # Place in the first codeblock
                lines = content.split("\n")
                position = find_code_block_line(lines)
                if position is None:
                    new_content = "\n".join(lines) + EXPECTED + "\n"
                else:
                    print(position)
                    new_content = (
                        "\n".join(lines[:position])
                        + EXPECTED
                        + "\n\n"
                        + "\n".join(lines[position:])
                    )

            with open(error, "w") as f:
                f.write(new_content)
    else:
        # Don't fix, just check and print an error message
        if len(errors) > 0:
            print(
                f"These {len(errors)} file(s) did not contain the expected text to "
                "override urllib.request.Request, it was at the wrong position, or "
                "the whitespace is incorrect.\n"
                "You can run 'python3 tests/lint/check_request_hook.py --fix' to "
                "automatically fix these errors:\n"
                f"{EXPECTED}\n\nFiles:"
            )
            for file, line_info in errors:
                if line_info is None:
                    print(f"{file} (missing hook)")
                else:
                    actual, expected = line_info
                    print(f"{file} (misplaced hook at {actual}, expected at {expected})")
            exit(1)
        else:
            print("All files successfully override urllib.request.Request")
            exit(0)
