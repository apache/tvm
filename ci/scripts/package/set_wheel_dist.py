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
"""Apply optional distribution name/version overrides to ``[project]``.

Used by the publishing flow (e.g. TestPyPI validation builds) so the wheel is
produced with the desired name/version directly by the build backend, instead
of rewriting the finished wheel. Reads ``TVM_WHEEL_DIST_NAME`` and
``TVM_WHEEL_DIST_VERSION`` from the environment; missing/empty values are left
unchanged. A no-op when neither is set.
"""

import os
import re
import sys


def _replace(text, key, value):
    # Replace the first ``key = "..."`` entry in the [project] table.
    pattern = re.compile(rf'(?m)^(\s*{key}\s*=\s*)"[^"]*"')
    new, n = pattern.subn(rf'\g<1>"{value}"', text, count=1)
    if n != 1:
        raise SystemExit(f"set_wheel_dist: could not find a single '{key}' entry to override")
    return new


def main():
    pyproject = sys.argv[1]
    name = os.environ.get("TVM_WHEEL_DIST_NAME", "").strip()
    version = os.environ.get("TVM_WHEEL_DIST_VERSION", "").strip()
    if not name and not version:
        return 0

    with open(pyproject, encoding="utf-8") as f:
        text = f.read()
    if name:
        text = _replace(text, "name", name)
    if version:
        text = _replace(text, "version", version)
    with open(pyproject, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"set_wheel_dist: name={name or '(unchanged)'} version={version or '(unchanged)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
