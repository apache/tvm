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
* [allow non-committers to merge approved / CI passing PRs](https://discuss.tvm.apache.org/t/rfc-allow-merging-via-pr-comments/12220)

https://github.com/apache/tvm/actions has the logs for each of these workflows. Note that when debugging these workflows changes from PRs from forked repositories won't be reflected in the PR. These should be tested in the forked repository first and linked in the PR body.

# Jenkins CI

TVM uses Jenkins for running Linux continuous integration (CI) tests on
[branches](https://ci.tlcpack.ai/job/tvm/) and
[pull requests](https://ci.tlcpack.ai/job/tvm/view/change-requests/) through a
build configuration specified in a [`Jenkinsfile`](generated/).
Other jobs run in GitHub Actions for Windows and MacOS jobs.

## Pipeline entry points

The five `generated/*_jenkinsfile.groovy` paths remain Jenkins job entry points.
Despite the historical directory name, these are native Groovy sources: edit
and review them directly. `generated/common.groovy` shares preparation, checkout,
Docker setup, artifacts and spot fallback policy. Platform commands and deployment
stay visible in their respective job files. Image tags remain in
`docker-images.ini`; artifact bundles remain in `data.py`.

Host-side commands live in `scripts/`. GitHub bot and release API helpers live
in `../github/`. The retained `determine_docker_images.py` supports both
Jenkins image selection and local `--lookup-only` lookup; `should_rebuild_docker.py`
and the standalone skip commands retain their command-line interfaces.

### Trust and rollout

Each entry point evaluates the shared helper returned by `readTrusted` at
Jenkins' selected SCM revision. Untrusted PRs that modify or delete the helper
fail immediately, before node allocation, checkout, or credentials. Missing
helpers, interrupted reads, and evaluation errors also propagate directly.
No alternative helper is fetched or evaluated.

Trusted helper changes use Jenkins' selected revision directly. Cancellation
runs once before preparation. During preparation, the normal source merge pins
the PR target revision for subsequent stages, including release targets.
Host scripts and artifact metadata follow that pinned target revision for PR
authors outside `CONTRIBUTORS.md`.

The legacy `ci/scripts/` hierarchy and `tests/scripts/task_show_node_info.sh`
remain alongside the new paths because fork PR jobs can select older trusted
Jenkinsfiles from their target branch. Those files keep the old docs-only and
slow-test gates and their helper imports; the native pipelines use the new
layout without those gates. Both layouts share the artifact registry and image
configuration. Keep the legacy commands until the trusted targets no longer
call them. Use a trusted branch build to exercise native helper changes; do not
bypass a refused read by evaluating PR-owned code or a different revision.

Before enabling the migration on production jobs, validate it with the existing
Jenkins plugin versions and sandbox settings on a trusted branch: run all five
entry points, suspend/resume across a node step, and exercise spot loss, a normal
build-command failure, superseded builds, and a failed on-demand retry. Check
trusted helper changes, untrusted PRs without helper changes, and refusal of
untrusted modified/deleted helpers, including a release-target PR. Verify image
overrides and disabled deployment gates before
checking enabled docs/Docker deployment in the normal trusted deployment context.
Local mocked execution checks pipeline control flow but cannot certify Jenkins
CPS serialization, sandbox approvals, SCM trust configuration, or real agent loss.
No job path or controller reconfiguration is part of this migration.

# Infrastructure

The community's infrastructure configuration and documentation live in
[tlc-pack/ci](https://github.com/tlc-pack/ci). Coordinate external callers when
changing any retained script interface.
