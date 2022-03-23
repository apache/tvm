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

import argparse
import re
import json
from collections import defaultdict

from cmd_utils import REPO_ROOT

CONTRIBUTORS = REPO_ROOT / "CONTRIBUTORS.md"


if __name__ == "__main__":
    help = "List out users by team from CONTRIBUTORS.md"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--format", default="github", help="'github' or 'json' for output")
    args = parser.parse_args()

    with open(CONTRIBUTORS) as f:
        content = f.readlines()

    parse_line = False
    all_topics = defaultdict(lambda: [])
    for line in content:
        if line.strip().startswith("## Committers"):
            parse_line = True
        elif line.strip().startswith("##"):
            parse_line = False

        if parse_line and line.startswith("- ["):
            line = line.strip()
            m = re.search("github.com\/(.*?)\).* - (.*)", line)
            user = m.groups()[0]
            topics = [t.lower().strip() for t in m.groups()[1].split(",")]
            for t in topics:
                all_topics[t].append(user)

    all_topics = dict(all_topics)
    if args.format == "json":
        print(json.dumps(all_topics, indent=2))
    else:
        items = sorted(all_topics.items(), key=lambda x: x[0])
        for topic, users in items:
            users = [f"@{user}" for user in users]
            users = " ".join(users)
            print(f"- {topic} {users}")
