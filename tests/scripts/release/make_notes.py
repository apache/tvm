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
import os
import pickle
from pathlib import Path
import csv
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(REPO_ROOT / "tests" / "scripts"))
sys.path.append(str(REPO_ROOT / "tests" / "scripts" / "github"))
sys.path.append(str(REPO_ROOT / "tests" / "scripts" / "jenkins"))


def strip_header(title: str, header: str) -> str:
    pos = title.lower().find(header.lower())
    if pos == -1:
        return title

    return title[0:pos] + title[pos + len(header) :].strip()


def sprint(*args):
    print(*args, file=sys.stderr)


if __name__ == "__main__":
    help = "List out commits with attached PRs since a certain commit"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--notes-csv", required=True, help="csv file of categorized PRs in order")
    args = parser.parse_args()
    user = "apache"
    repo = "tvm"

    cache = Path("out.pkl")
    if not cache.exists():
        sprint("run gather_prs.py first to generate out.pkl")
        exit(1)

    with open(cache, "rb") as f:
        data = pickle.load(f)

    sprint(data[1])
    reverse = {}
    for item in data:
        prs = item["associatedPullRequests"]["nodes"]
        if len(prs) != 1:
            continue

        pr = prs[0]
        reverse[pr["number"]] = pr

    def pr_title(number, heading):
        title = reverse[int(number)]["title"]
        title = strip_header(title, heading)
        return title

    headings = defaultdict(lambda: defaultdict(list))
    output = ""

    sprint("Opening CSV")
    with open(args.notes_csv) as f:
        # Skip header stuff
        f.readline()
        f.readline()
        f.readline()

        input_file = csv.DictReader(f)

        i = 0
        for row in input_file:
            category = row["category"].strip()
            subject = row["subject"].strip()
            pr_number = row["url"].split("/")[-1]
            if category == "" or subject == "":
                sprint(f"Skipping {pr_number}")
                continue

            headings[category][subject].append(pr_number)
            i += 1
            # if i > 30:
            #     break

    def sorter(x):
        if x == "Misc":
            return 10
        return 0

    keys = list(headings.keys())
    keys = list(sorted(keys))
    keys = list(sorted(keys, key=sorter))
    for key in keys:
        value = headings[key]
        if key == "DO NOT INCLUDE":
            continue
        value = dict(value)
        output += f"### {key}\n"

        misc = []
        misc += value.get("n/a", [])
        misc += value.get("Misc", [])
        for pr_number in misc:
            output += f" * #{pr_number} - {pr_title(pr_number, '[' + key + ']')}\n"

        for subheading, pr_numbers in value.items():
            if subheading == "DO NOT INCLUDE":
                continue
            if subheading == "n/a" or subheading == "Misc":
                continue
            else:
                output += f" * {subheading} - " + ", ".join([f"#{n}" for n in pr_numbers]) + "\n"
            # print(value)

        output += "\n"

    print(output)
