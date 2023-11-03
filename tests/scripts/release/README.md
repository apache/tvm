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

These scripts can be helpful when creating release notes.

```bash
# example: create a csv file of all PRs since the v0.8 and v0.9.0 releases
# the result will be in 2 CSV files based on the --threshold arg (small PRs vs large PRs)
export GITHUB_TOKEN=<github oauth token>
python release/gather_prs.py --from-commit $(git rev-parse v0.9.0) --to-commit $(git merge-base origin/main v0.8.0)
```

After running the commands above, you will get a csv file named `out_pr_gathered.csv`. You can then import this CSV into a collaborative spreadsheet editor to distribute the work of categorizing PRs for the notes, **especially check the `pr_title_tags` column for each row and correct it if it's wrong**.

Once done, you can download the csv file assuming with name `out_pr_gathered_corrected.csv` and convert it to readable release notes using commands below:

```bash
# example: use a csv of tags-corrected PRs to create a markdown file

# Export monthly report on forum:
python make_notes.py --notes out_pr_gathered_corrected.csv --is-pr-with-link true > monthly_report.md

# Export release report on Github:
python make_notes.py --notes out_pr_gathered_corrected.csv --is-pr-with-link true > release_report.md

# If release report exported but forget set `--is-pr-with-link true`,
# you can append arg `--convert-with-link true`.
python3 make_notes.py --notes ./release_report.md --convert-with-link true
```

You can also create a list of RFCs

```bash
git clone https://github.com/apache/tvm-rfcs.git

# example: list RFCs since a specific commit in the tvm-rfcs repo
python list_rfcs.py --since-commit <hash> --rfcs-repo ./tvm-rfcs > rfc.md
```

Finally, combine `rfc.md` and `out.md` along with some prose to create the final release notes.
