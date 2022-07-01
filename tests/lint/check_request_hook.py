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


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPECTED = """
# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore
""".rstrip()
IGNORE_PATTERNS = ["*/micro_tvmc.py", "*/micro_train.py"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check that all tutorials/docs override urllib.request.Request"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Insert expected code into erroring files"
    )
    args = parser.parse_args()

    gallery_files = (REPO_ROOT / "gallery").glob("**/*.py")

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
            errors.append(file)

    if args.fix:
        for error in errors:
            with open(error) as f:
                content = f.read()

            if "from __future__" in content:
                # Place after the last __future__ import
                new_content = re.sub(
                    r"((?:from __future__.*?\n)+)", r"\1\n" + EXPECTED, content, flags=re.MULTILINE
                )
            else:
                # Place after the module doc comment
                new_content = re.sub(
                    r"(\"\"\"(?:.*\n)+\"\"\")", r"\1\n" + EXPECTED, content, flags=re.MULTILINE
                )

            with open(error, "w") as f:
                f.write(new_content)
    else:
        # Don't fix, just check and print an error message
        if len(errors) > 0:
            print(
                f"These {len(errors)} files did not contain the expected text to "
                "override urllib.request.Request.\n"
                "You can run 'python3 tests/lint/check_request_hook.py --fix' to "
                "automatically fix these errors:\n"
                f"{EXPECTED}\n\nFiles:\n" + "\n".join([str(error_path) for error_path in errors])
            )
            exit(1)
        else:
            print("All files successfully override urllib.request.Request")
            exit(0)
