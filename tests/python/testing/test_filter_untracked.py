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

import os
import shutil
import subprocess
import sys
import tempfile


def setup_git_repo(worktree=False):
    git_repo_dir = tempfile.mkdtemp()
    to_rm = [git_repo_dir]
    try:
        subprocess.check_output(["git", "init", "."], cwd=git_repo_dir)

        with open(f"{git_repo_dir}/committed", "w") as committed_f:
            committed_f.write("normal committed file\n")

        subprocess.check_output(["git", "add", "committed"], cwd=git_repo_dir)

        with open(f"{git_repo_dir}/committed-ignored", "w") as gitignore_f:
            gitignore_f.write("this file is gitignored, but committed already")

        subprocess.check_output(["git", "add", "committed-ignored"], cwd=git_repo_dir)

        with open(f"{git_repo_dir}/.gitignore", "w") as gitignore_f:
            gitignore_f.write("ignored\n" "committed-ignored\n")

        subprocess.check_output(["git", "add", ".gitignore"], cwd=git_repo_dir)

        # NOTE: explicitly set the author so this test passes in the CI.
        subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=Unit Test",
                "-c",
                "user.email=unit.test@testing.tvm.ai",
                "commit",
                "-m",
                "initial commit",
            ],
            cwd=git_repo_dir,
        )

        if worktree:
            worktree_dir = tempfile.mkdtemp()
            to_rm.append(worktree_dir)
            subprocess.check_output(["git", "worktree", "add", worktree_dir], cwd=git_repo_dir)
            git_repo_dir = worktree_dir

        with open(f"{git_repo_dir}/ignored", "w") as gitignore_f:
            gitignore_f.write("this file is gitignored")

        with open(f"{git_repo_dir}/added-to-index", "w") as added_f:
            added_f.write("only added to git index\n")

        subprocess.check_output(["git", "add", "added-to-index"], cwd=git_repo_dir)

        with open(f"{git_repo_dir}/ignored-added-to-index", "w") as ignored_f:
            ignored_f.write("this file is gitignored but in the index already\n")

        subprocess.check_output(["git", "add", "-f", "ignored-added-to-index"], cwd=git_repo_dir)

        with open(f"{git_repo_dir}/untracked", "w") as untracked_f:
            untracked_f.write("this file is untracked\n")

        os.mkdir(f"{git_repo_dir}/subdir")
        with open(f"{git_repo_dir}/subdir/untracked", "w") as untracked_f:
            untracked_f.write("this file is untracked\n")

        with open(f"{git_repo_dir}/subdir/untracked2", "w") as untracked_f:
            untracked_f.write("this file is also untracked\n")

        return git_repo_dir, to_rm

    except Exception:
        for rm_dir in to_rm:
            shutil.rmtree(rm_dir)
        raise


def run_test(repo_path, passed_files, filtered_files):
    test_input = (
        "\n".join(
            passed_files
            + filtered_files
            + [f"./{f}" for f in passed_files]
            + [f"./{f}" for f in filtered_files]
        )
        + "\n"
    )

    test_script_dir = f"{repo_path}/test-script-dir"
    os.mkdir(test_script_dir)

    filter_script_path = f"{test_script_dir}/filter_untracked.py"
    test_script_dirname = os.path.dirname(__file__) or os.getcwd()
    shutil.copy(
        os.path.realpath(f"{test_script_dirname}/../../lint/filter_untracked.py"),
        filter_script_path,
    )
    filter_proc = subprocess.Popen(
        [sys.executable, filter_script_path],
        cwd=repo_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        encoding="utf-8",
    )
    filter_output, _ = filter_proc.communicate(test_input)
    filter_output_lines = [l for l in filter_output.split("\n") if l]

    for pass_f in passed_files:
        assert (
            pass_f in filter_output_lines
        ), f"expected in filter output: {pass_f}\filter output: {filter_output}"
        assert (
            f"./{pass_f}" in filter_output_lines
        ), f"expected in filter output: ./{pass_f}\filter output: {filter_output}"

    for filter_f in filtered_files:
        assert (
            filter_f not in filter_output_lines
        ), f"expected not in filter output: {filter_f}\nfilter_output: {filter_output}"
        assert (
            f"./{filter_f}" not in filter_output_lines
        ), f"expected not in filter output: ./{filter_f}\nfilter_output: {filter_output}"

    assert len(filter_output_lines) == 2 * len(
        passed_files
    ), f"expected {len(filter_output_lines)} == 2 * {len(passed_files)}"


def test_filter_untracked():
    repo_path, to_rm = setup_git_repo()
    try:
        passed_files = [
            "committed",
            "committed-ignored",
            "added-to-index",
            "ignored-added-to-index",
        ]
        filtered_files = [
            "ignored",
            "untracked",
            "subdir/untracked",
            "subdir/untracked2",
        ]
        run_test(repo_path, passed_files, filtered_files)

    finally:
        for rm_dir in to_rm:
            shutil.rmtree(rm_dir)


def test_worktree():
    repo_path, to_rm = setup_git_repo(worktree=True)
    try:
        passed_files = [
            "committed",
            "committed-ignored",
            "added-to-index",
            "ignored-added-to-index",
        ]
        filtered_files = [
            "ignored",
            "untracked",
            "subdir/untracked",
            "subdir/untracked2",
            ".git",
        ]
        run_test(repo_path, passed_files, filtered_files)

    finally:
        for rm_dir in to_rm:
            shutil.rmtree(rm_dir)


if __name__ == "__main__":
    test_filter_untracked()
    test_worktree()
