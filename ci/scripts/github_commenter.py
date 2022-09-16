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

import re
import logging
from typing import Dict, Tuple, Any, Optional, List, Union

from git_utils import GitHubRepo

BOT_COMMENT_START = "<!---bot-comment-->"
WELCOME_TEXT = "Thanks for contributing to TVM! Please refer to the [contributing guidelines](https://tvm.apache.org/docs/contribute/) for useful information and tips. Please request code reviews from [Reviewers](https://github.com/apache/incubator-tvm/blob/master/CONTRIBUTORS.md#reviewers) by @-ing them in a comment."


class BotCommentBuilder:
    ALLOWLIST_USERS = {"driazati", "gigiblender", "areusch"}

    def __init__(self, github: GitHubRepo, data: Dict[str, Any], run_url: str):
        self.github = github
        self.pr_number = data["number"]
        self.comment_data = data["comments"]["nodes"]
        self.author = data["author"]["login"]
        self.run_url = run_url

    def find_bot_comment(self) -> Optional[Dict[str, Any]]:
        """
        Return the existing bot comment or None if it does not exist
        """
        for comment in self.comment_data:
            logging.info(f"Checking comment {comment}")
            if (
                comment["author"]["login"] == "github-actions"
                and BOT_COMMENT_START in comment["body"]
            ):
                logging.info("Found existing comment")
                return comment
        logging.info("No existing comment found")
        return None

    def find_existing_body(self) -> Dict[str, str]:
        """
        Find existing dynamic bullet point items
        """
        existing_comment = self.find_bot_comment()
        if existing_comment is None:
            logging.info(f"No existing comment while searching for body items")
            return {}

        matches = re.findall(
            r"<!--bot-comment-([a-z][a-z-]+)-start-->([\S\s]*?)<!--bot-comment-([a-z-]+)-end-->",
            existing_comment["body"],
            flags=re.MULTILINE,
        )
        logging.info(f"Fetch body item matches: {matches}")

        items = {}
        for start, text, end in matches:
            if start != end:
                raise RuntimeError(
                    f"Malformed comment found: {start} marker did not have matching end, found instead {end}"
                )
            items[start] = text.strip().lstrip("* ")

        logging.info(f"Found body items: {items}")
        return items

    def _post_comment(self, body_items: Dict[str, str]):
        comment = BOT_COMMENT_START + "\n\n" + WELCOME_TEXT + "\n\n"
        for key, content in body_items.items():
            # Indent every line but the first
            lines = content.strip().split("\n")
            for i in range(1, len(lines)):
                lines[i] = "    " + lines[i]
            content = "\n".join(lines)

            # Add the bullet point
            line = self.start_key(key) + "\n * " + content.strip() + self.end_key(key)
            logging.info(f"Adding line {line}")
            comment += line
        comment += (
            f"\n\n<sub>[Generated]({self.run_url}) by "
            "[tvm-bot](https://github.com/apache/tvm/blob/main/ci/README.md#github-actions)</sub>"
        )

        data = {"body": comment}
        url = f"issues/{self.pr_number}/comments"

        logging.info(f"Commenting {comment} on {url}")

        if self.author not in self.ALLOWLIST_USERS:
            logging.info(f"Skipping comment for author {self.author}")
            return

        existing_comment = self.find_bot_comment()
        if existing_comment is None:
            # Comment does not exist, post it
            r = self.github.post(url, data)
        else:
            # Comment does exist, update it
            comment_url = f"issues/comments/{existing_comment['databaseId']}"
            r = self.github.patch(comment_url, data)

        logging.info(f"Got response from posting comment: {r}")

    def start_key(self, key: str) -> str:
        return f"<!--bot-comment-{key}-start-->"

    def end_key(self, key: str) -> str:
        return f"<!--bot-comment-{key}-end-->"

    def post_items(self, items: List[Tuple[str, str]]):
        """
        Update or post bullet points in the PR based on 'items' which is a
        list of (key, text) pairs
        """
        # Find the existing bullet points
        body_items = self.find_existing_body()

        # Add or update the requested items
        for key, text in items:
            if text is None or text.strip() == "":
                logging.info(f"Skipping {key} since it was empty")
                continue
            logging.info(f"Updating comment items {key} with {text}")
            body_items[key] = text.strip()

        # Post or update the comment
        # print(body_items)
        self._post_comment(body_items=body_items)
