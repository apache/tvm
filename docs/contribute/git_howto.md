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

# Git Usage Tips

Here are some tips for git workflow.

## How to resolve conflict with master
- First rebase to most recent master
```bash
# The first two steps can be skipped after you do it once.
git remote add upstream [url to tvm repo]
git fetch upstream
git rebase upstream/master
```
- The git may show some conflicts it cannot merge, say ```conflicted.py```.
  - Manually modify the file to resolve the conflict.
  - After you resolved the conflict, mark it as resolved by
```bash
git add conflicted.py
```
- Then you can continue rebase by
```bash
git rebase --continue
```
- Finally push to your fork, you may need to force push here.
```bash
git push --force
```

## How to combine multiple commits into one
Sometimes we want to combine multiple commits, especially when later commits are only fixes to previous ones,
to create a PR with set of meaningful commits. You can do it by following steps.
- Before doing so, configure the default editor of git if you haven't done so before.
```bash
git config core.editor the-editor-you-like
```
- Assume we want to merge last 3 commits, type the following commands
```bash
git rebase -i HEAD~3
```
- It will pop up an text editor. Set the first commit as ```pick```, and change later ones to ```squash```.
- After you saved the file, it will pop up another text editor to ask you modify the combined commit message.
- Push the changes to your fork, you need to force push.
```bash
git push --force
```

## Reset to the most recent master
You can always use git reset to reset your version to the most recent master.
Note that all your ***local changes will get lost***.
So only do it when you do not have local changes or when your pull request just get merged.
```bash
git reset --hard [hash tag of master]
git push --force
```

## What is the consequence of force push
The previous two tips requires force push, this is because we altered the path of the commits.
It is fine to force push to your own fork, as long as the commits changed are only yours.
