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
# ruff: noqa: E741
"""Helper tool to check and fix ASF headers in source files tracked by git.

Ported from tvm-ffi. Replaces the previous RAT-based check_asf_header.sh
(which required Java) with a self-contained Python implementation.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

header_cstyle = """
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
""".strip()

header_mdstyle = """
<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->
""".strip()

header_pystyle = """
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
""".strip()

header_rststyle = """
..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
""".strip()

header_groovystyle = """
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
""".strip()

header_cmdstyle = """
:: Licensed to the Apache Software Foundation (ASF) under one
:: or more contributor license agreements.  See the NOTICE file
:: distributed with this work for additional information
:: regarding copyright ownership.  The ASF licenses this file
:: to you under the Apache License, Version 2.0 (the
:: "License"); you may not use this file except in compliance
:: with the License.  You may obtain a copy of the License at
::
::   http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing,
:: software distributed under the License is distributed on an
:: "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
:: KIND, either express or implied.  See the License for the
:: specific language governing permissions and limitations
:: under the License.
""".strip()

FMT_MAP = {
    "sh": header_pystyle,
    "cc": header_cstyle,
    "c": header_cstyle,
    "cu": header_cstyle,
    "cuh": header_cstyle,
    "mm": header_cstyle,
    "m": header_cstyle,
    "go": header_cstyle,
    "java": header_cstyle,
    "h": header_cstyle,
    "pyi": header_pystyle,
    "py": header_pystyle,
    "pyx": header_pystyle,
    "toml": header_pystyle,
    "yml": header_pystyle,
    "yaml": header_pystyle,
    "rs": header_cstyle,
    "md": header_mdstyle,
    "cmake": header_pystyle,
    "mk": header_pystyle,
    "rst": header_rststyle,
    "gradle": header_groovystyle,
    "groovy": header_groovystyle,
    "tcl": header_pystyle,
    "xml": header_mdstyle,
    "storyboard": header_mdstyle,
    "pbxproj": header_cstyle,
    "plist": header_mdstyle,
    "xcworkspacedata": header_mdstyle,
    "html": header_mdstyle,
    "bat": header_cmdstyle,
}

# Files and patterns to skip during header checking.
# Files and patterns to skip (3rdparty, generated files, etc.).
SKIP_LIST: list[str] = [
    "3rdparty/*",
    "ffi/3rdparty/*",
    ".github/*",
    "*.json",
    "*.txt",
    "*.svg",
    "*.lst",
    "*.lds",
    "*.in",
    "*.diff",
    "*.edl",
    "*.md5",
    "*.csv",
    "*.log",
    "*.interp",
    "*.tokens",
    "*.ipynb",
    "*.conf",
    "*.ini",
    "*.lock",
    "*.properties",
    "*.j2",
]


def should_skip_file(filepath: str) -> bool:
    """Check if file should be skipped based on SKIP_LIST."""
    for pattern in SKIP_LIST:
        if fnmatch.fnmatch(filepath, pattern):
            return True
    return False


def get_git_files() -> list[str] | None:
    """Get list of files tracked by git."""
    try:
        result = subprocess.run(
            ["git", "ls-files"], check=False, capture_output=True, text=True, cwd=Path.cwd()
        )
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.split("\n") if line.strip()]
        else:
            print("Error: Could not get git files. Make sure you're in a git repository.")
            print("Git command failed:", result.stderr.strip())
            return None
    except FileNotFoundError:
        print("Error: Git not found. This tool requires git to be installed.")
        return None


def copyright_line(line: str) -> bool:
    # Following two items are intentionally break apart
    # so that the copyright detector won"t detect the file itself.
    if line.find("Copyright " + "(c)") != -1:
        return True
    # break pattern into two lines to avoid false-negative check
    spattern1 = "Copyright"
    if line.find(spattern1) != -1 and line.find("by") != -1:
        return True
    return False


def check_header(fname: str, header: str) -> bool:
    """Check header status of file without modifying it."""
    if not Path(fname).exists():
        print(f"ERROR: Cannot find {fname}")
        return False

    lines = Path(fname).open().readlines()

    has_asf_header = False
    has_copyright = False

    for i, l in enumerate(lines):
        if l.find("Licensed to the Apache Software Foundation") != -1:
            has_asf_header = True
        elif copyright_line(l):
            has_copyright = True

    if has_asf_header and not has_copyright:
        return True  # File is good

    if not has_asf_header:
        print(f"ERROR: Missing ASF header in {fname}")
        print("Run `python tests/lint/check_asf_header.py --fix` to add the header")
        return False

    if has_copyright:
        print(f"ERROR: Has copyright line that should be removed in {fname}")
        return False

    return True


def collect_files() -> list[str] | None:
    """Collect all files that need header checking from git."""
    files = []

    # Get files from git (required)
    git_files = get_git_files()

    if git_files is None:
        # Git is required, so exit if we can't get files
        return None

    # Filter git files based on supported types and skip list
    for git_file in git_files:
        # Check if file should be skipped
        if should_skip_file(git_file):
            continue

        # Check if this file type is supported
        suffix = git_file.split(".")[-1] if "." in git_file else ""
        basename = Path(git_file).name

        if (
            suffix in FMT_MAP
            or basename == "gradle.properties"
            or (suffix == "" and basename in ["CMakeLists", "Makefile"])
        ):
            files.append(git_file)

    return files


def add_header(fname: str, header: str) -> None:
    """Add header to file."""
    if not Path(fname).exists():
        print(f"Cannot find {fname} ...")
        return

    lines = Path(fname).open().readlines()

    has_asf_header = False
    has_copyright = False

    for i, l in enumerate(lines):
        if l.find("Licensed to the Apache Software Foundation") != -1:
            has_asf_header = True
        elif copyright_line(l):
            has_copyright = True
            lines[i] = ""

    if has_asf_header and not has_copyright:
        return

    with Path(fname).open("w") as outfile:
        skipline = False
        if not lines:
            skipline = False  # File is empty
        elif lines[0][:2] == "#!":
            skipline = True
        elif lines[0][:2] == "<?":
            skipline = True
        elif lines[0].startswith("<html>"):
            skipline = True
        elif lines[0].startswith("// !$"):
            skipline = True

        if skipline:
            outfile.write(lines[0])
            if not has_asf_header:
                outfile.write(header + "\n\n")
            outfile.write("".join(lines[1:]))
        else:
            if not has_asf_header:
                outfile.write(header + "\n\n")
            outfile.write("".join(lines))
    if not has_asf_header:
        print(f"Add header to {fname}")
    if has_copyright:
        print(f"Removed copyright line from {fname}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check and fix ASF headers in source files tracked by git",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool processes all files tracked by git (using 'git ls-files') and automatically
skips files matching patterns in SKIP_LIST (build artifacts, third-party files, etc.).

One of --check, --fix, or --dry-run must be specified.

Examples:
  # Show all files that would be processed (dry run)
  python check_asf_header.py --dry-run

  # Check all git-tracked files (report errors)
  python check_asf_header.py --check

  # Fix all git-tracked files (add/update headers)
  python check_asf_header.py --fix
        """,
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: report errors without modifying files",
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix mode: fix files with missing or incorrect headers (default)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: show files that would be processed without doing anything",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.check and args.fix:
        print("Error: Cannot specify both --check and --fix")
        return 1

    if not args.check and not args.fix and not args.dry_run:
        print("Error: Must specify one of --check, --fix, or --dry-run")
        return 1

    # Always use git-based approach
    files = collect_files()

    if files is None:
        return 1  # Error already printed by get_git_files()

    if not files:
        print("No files found to process")
        return 0

    # Handle dry-run mode
    if args.dry_run:
        print(f"Would process {len(files)} files:")
        for fname in sorted(files):
            print(f"  {fname}")
        return 0

    # Determine mode
    check_only = args.check

    error_count = 0
    processed_count = 0

    for fname in files:
        processed_count += 1
        suffix = fname.split(".")[-1] if "." in fname else ""
        basename = Path(fname).name

        # Determine header type
        if suffix in FMT_MAP:
            header = FMT_MAP[suffix]
        elif basename == "gradle.properties":
            header = FMT_MAP["h"]
        elif suffix == "" and basename in ["CMakeLists", "Makefile"]:
            header = FMT_MAP["cmake"] if basename == "CMakeLists" else FMT_MAP["mk"]
        else:
            if check_only:
                print(f"ERROR: Cannot handle file type for {fname}")
                error_count += 1
            else:
                print(f"Cannot handle {fname} ...")
            continue

        if check_only:
            # Check mode
            if not check_header(fname, header):
                error_count += 1
        else:
            # Fix mode
            add_header(fname, header)

    if check_only:
        if error_count > 0:
            print(f"\nFound {error_count} errors in {processed_count} files")
            return 1
        else:
            print(f"\nAll {processed_count} files are compliant")
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
