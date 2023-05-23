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
Test the @tvm-bot merge code
"""

import json
from pathlib import Path
from typing import Dict, Any

import tvm
from .test_utils import GITHUB_SCRIPT_ROOT, TempGit, run_script


SUCCESS_EXPECTED_OUTPUT = """
Dry run, would have merged with url=pulls/10786/merge and data={
  "commit_title": "[Hexagon] 2-d allocation cleanup (#10786)",
  "commit_message": "- Added device validity check in allocation. HexagonDeviceAPI should only be called for CPU/Hexagon types.\\n\\n- Check for \\"global.vtcm\\" scope instead of \\"vtcm\\".  The ccope of N-d allocations produced by `LowerVtcmAlloc` should be `\\"global.vtcm\\"`.  The previous check allowed unsupported scope such as `\\"local.vtcm\\"`.\\n\\n- Remove `vtcmallocs` entry after calling free.\\n\\nPreviously, the vtcm allocation map kept dangling pointers to `HexagonBuffer` objects after they had been freed.\\n\\n- Rename N-d alloc and free packed functions.  Since most of the similar device functions use snake case, renaming `*.AllocND` to `*.alloc_nd` and `*.FreeND` to `*.free_nd`.\\n\\n\\ncc someone\\n\\n\\nCo-authored-by: Adam Straw <astraw@octoml.ai>",
  "sha": "6f04bcf57d07f915a98fd91178f04d9e92a09fcd",
  "merge_method": "squash"
}
""".strip()


class _TvmBotTest:
    NUMBER = 10786

    def preprocess_data(self, data: Dict[str, Any]):
        """
        Used to pre-process PR data before running the test. Override as
        necessary to edit data for specific test cases.
        """
        return data

    @tvm.testing.skip_if_wheel_test
    def test(self, tmpdir_factory):
        """
        Run the tvm-bot script using the data from preprocess_data
        """
        mergebot_script = GITHUB_SCRIPT_ROOT / "github_tvmbot.py"
        test_json_dir = Path(__file__).resolve().parent / "sample_prs"
        with open(test_json_dir / f"pr{self.NUMBER}.json") as f:
            test_data = json.load(f)

        # Update testing data with replacements / additions
        test_data = self.preprocess_data(test_data)

        git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))

        comment = {
            "body": self.COMMENT,
            "id": 123,
            "user": {
                "login": self.USER,
            },
        }
        allowed_users = [{"login": "abc"}, {"login": "other-abc"}]

        proc = run_script(
            [
                mergebot_script,
                "--pr",
                self.NUMBER,
                "--dry-run",
                "--run-url",
                "https://example.com",
                "--testing-pr-json",
                json.dumps(test_data),
                "--testing-collaborators-json",
                json.dumps(allowed_users),
                "--testing-mentionable-users-json",
                json.dumps(allowed_users),
                "--trigger-comment-json",
                json.dumps(comment),
            ],
            env={
                "TVM_BOT_JENKINS_TOKEN": "123",
                "GH_ACTIONS_TOKEN": "123",
            },
            cwd=git.cwd,
        )

        if self.EXPECTED not in proc.stderr:
            raise RuntimeError(f"{proc.stderr}\ndid not contain\n{self.EXPECTED}")


class TestNoRequest(_TvmBotTest):
    """
    A PR for which the mergebot runs but no merge is requested
    """

    COMMENT = "@tvm-bot do something else"
    USER = "abc"
    EXPECTED = "Command 'do something else' did not match anything"

    def preprocess_data(self, data: Dict[str, Any]):
        data["reviews"]["nodes"][0]["body"] = "nothing"
        return data


class TestSuccessfulMerge(_TvmBotTest):
    """
    Everything is fine so this PR will merge
    """

    COMMENT = "@tvm-bot merge"
    USER = "abc"
    EXPECTED = SUCCESS_EXPECTED_OUTPUT


class TestBadCI(_TvmBotTest):
    """
    A PR which failed CI and cannot merge
    """

    COMMENT = "@tvm-bot merge"
    USER = "abc"
    EXPECTED = "Cannot merge, these CI jobs are not successful on"

    def preprocess_data(self, data: Dict[str, Any]):
        # Mark the Jenkins build as failed
        contexts = data["commits"]["nodes"][0]["commit"]["statusCheckRollup"]["contexts"]["nodes"]
        for context in contexts:
            if "context" in context and context["context"] == "tvm-ci/pr-head":
                context["state"] = "FAILED"
        return data


class TestOldReview(_TvmBotTest):
    """
    A PR with passing CI and approving reviews on an old commit so it cannot merge
    """

    COMMENT = "@tvm-bot merge"
    USER = "abc"
    EXPECTED = "Cannot merge, did not find any approving reviews"

    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["reviews"]["nodes"][0]["commit"]["oid"] = "abc12345"
        return data


class TestMissingJob(_TvmBotTest):
    """
    PR missing an expected CI job and cannot merge
    """

    COMMENT = "@tvm-bot merge"
    USER = "abc"
    EXPECTED = "Cannot merge, missing expected jobs"

    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        contexts = data["commits"]["nodes"][0]["commit"]["statusCheckRollup"]["contexts"]["nodes"]
        for context in contexts:
            if "context" in context and context["context"] == "tvm-ci/pr-head":
                context["context"] = "something"
        return data


class TestInvalidAuthor(_TvmBotTest):
    """
    Merge requester is not a committer and cannot merge
    """

    COMMENT = "@tvm-bot merge"
    USER = "not-abc"
    EXPECTED = "Failed auth check 'collaborators', quitting"


class TestUnauthorizedComment(_TvmBotTest):
    """
    Check that a merge comment not from a CONTRIBUTOR is rejected
    """

    COMMENT = "@tvm-bot merge"
    USER = "not-abc2"
    EXPECTED = "Failed auth check 'collaborators'"


class TestNoReview(_TvmBotTest):
    """
    Check that a merge request without any reviews is rejected
    """

    COMMENT = "@tvm-bot merge"
    USER = "abc"
    EXPECTED = "Cannot merge, did not find any approving reviews from users with write access"

    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["reviews"]["nodes"] = []
        return data


class TestChangesRequested(_TvmBotTest):
    """
    Check that a merge request with a 'Changes Requested' review is rejected
    """

    COMMENT = "@tvm-bot merge"
    USER = "abc"
    EXPECTED = "Cannot merge, found [this review]"

    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["reviews"]["nodes"][0]["state"] = "CHANGES_REQUESTED"
        data["reviews"]["nodes"][0]["url"] = "http://example.com"
        return data


class TestCoAuthors(_TvmBotTest):
    """
    Check that a merge request with co-authors generates the correct commit message
    """

    COMMENT = "@tvm-bot merge"
    USER = "abc"
    EXPECTED = "Co-authored-by: Some One <someone@email.com>"

    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["authorCommits"]["nodes"][0]["commit"]["authors"]["nodes"].append(
            {"name": "Some One", "email": "someone@email.com"}
        )
        return data


class TestRerunCI(_TvmBotTest):
    """
    Start a new CI job
    """

    COMMENT = "@tvm-bot rerun"
    USER = "abc"
    EXPECTED = "Rerunning ci with"


class TestRerunPermissions(_TvmBotTest):
    """
    Start a new CI job as an unauthorized user
    """

    COMMENT = "@tvm-bot rerun"
    USER = "someone"
    EXPECTED = "Failed auth check 'mentionable_users', quitting"


class TestRerunNonAuthor(_TvmBotTest):
    """
    Start a new CI job as a mentionable user
    """

    COMMENT = "@tvm-bot rerun"
    USER = "other-abc"
    EXPECTED = "Passed auth check 'mentionable_users', continuing"


class TestIgnoreJobs(_TvmBotTest):
    """
    Ignore GitHub Actions jobs that don't start with CI /
    """

    COMMENT = "@tvm-bot merge"
    USER = "abc"
    EXPECTED = "Dry run, would have merged"


if __name__ == "__main__":
    tvm.testing.main()
