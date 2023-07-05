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

# TVM CI

TVM runs CI jobs on every commit to an open pull request and to branches in the apache/tvm repo (such as `main`). These jobs are essential to keeping the TVM project in a healthy state and preventing breakages.

## Jenkins

Jenkins runs all of the linux-based TVM CI-enabled regression tests. This includes tests against accelerated hardware such as GPUs. It excludes those regression tests that run against hardware not available in the cloud (those tests aren't currently exercised in TVM CI). The tests run by Jenkins represent most of the merge-blocking tests (and passing Jenkins should mostly correlate with passing the remaining Windows/Mac builds).

## GitHub Actions

GitHub Actions is used to run Windows jobs, MacOS jobs, and various on-GitHub automations. These are defined in [`.github/workflows`](../../.github/workflows/). These automations include bots to:
* [cc people based on subscribed teams/topics](https://github.com/apache/tvm/issues/10317)
* [allow non-committers to merge approved / CI passing PRs](https://discuss.tvm.apache.org/t/rfc-allow-merging-via-pr-comments/12220)
* [add cc-ed people as reviewers on GitHub](https://discuss.tvm.apache.org/t/rfc-remove-codeowners/12095)
* [ping languishing PRs after no activity for a week (currently opt-in only)](https://github.com/apache/tvm/issues/9983)
* [push a `last-successful` branch to GitHub with the last `main` commit that passed CI](https://github.com/apache/tvm/tree/last-successful)

https://github.com/apache/tvm/actions has the logs for each of these workflows. Note that when debugging these workflows changes from PRs from forked repositories won't be reflected in the PR. These should be tested in the forked repository first and linked in the PR body.

# Jenkins CI

TVM uses Jenkins for running Linux continuous integration (CI) tests on
[branches](https://ci.tlcpack.ai/job/tvm/) and
[pull requests](https://ci.tlcpack.ai/job/tvm/view/change-requests/) through a
build configuration specified in a [`Jenkinsfile`](../../Jenkinsfile).
Other jobs run in GitHub Actions for Windows and MacOS jobs.

## `Jenkinsfile`

The template files in this directory are used to generate the [`Jenkinsfile`](../../Jenkinsfile) used by Jenkins to run CI jobs for each commit to PRs and branches.

To regenerate the `Jenkinsfile`, run

```bash
python3 -mvenv _venv
_venv/bin/pip3 install -r ci/jenkins/requirements.txt
_venv/bin/python3 ci/jenkins/generate.py
```

# Infrastructure

While all TVM tests are contained within the apache/tvm repository, the infrastructure used to run the tests is donated by the TVM Community. To encourage collaboration, the configuration for TVM's CI infrastructure is stored in a public GitHub repository. TVM community members are encouraged to contribute improvements. The configuration, along with documentation of TVM's CI infrastructure, is in the [tlc-pack/ci](https://github.com/tlc-pack/ci) repo.
