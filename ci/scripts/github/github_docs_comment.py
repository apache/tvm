#!/usr/bin/env python3
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

from typing import Dict, Any


def build_docs_url(base_url_docs, pr_number, build_number):
    return f"{base_url_docs}/PR-{str(pr_number)}/{str(build_number)}/docs/index.html"


def find_target_url(pr_head: Dict[str, Any]):
    for status in pr_head["statusCheckRollup"]["contexts"]["nodes"]:
        if status.get("context", "") == "tvm-ci/pr-head":
            return status["targetUrl"]

    raise RuntimeError(f"Unable to find tvm-ci/pr-head status in {pr_head}")


def get_pr_and_build_numbers(target_url):
    target_url = target_url[target_url.find("PR-") : len(target_url)]
    split = target_url.split("/")
    pr_number = split[0].strip("PR-")
    build_number = split[1]
    return {"pr_number": pr_number, "build_number": build_number}


def get_doc_url(pr: Dict[str, Any], base_docs_url: str = "https://pr-docs.tlcpack.ai") -> str:
    pr_head = pr["commits"]["nodes"][0]["commit"]
    target_url = find_target_url(pr_head)
    pr_and_build = get_pr_and_build_numbers(target_url)

    commit_sha = pr_head["oid"]

    docs_url = build_docs_url(
        base_docs_url, pr_and_build["pr_number"], pr_and_build["build_number"]
    )

    return f"Built docs for commit {commit_sha} can be found [here]({docs_url})."
