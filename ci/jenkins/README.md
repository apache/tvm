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

Each entry point first uses `readTrusted` to read the helper at Jenkins' pinned
SCM revision. When an untrusted PR modifies that file, Jenkins refuses the read.
The entry point then checks out a fresh bootstrap workspace and reads the helper
from the fetched **PR target branch** in the canonical Apache repository. It pins
that exact target commit for all subsequent source merges, including release
branch targets. It never evaluates workspace helper code or substitutes `main`.
The bootstrap runs only built-in checkout/Git commands, without repository scripts
or credentials. Missing target files and evaluation errors fail the build.

This fallback adds a bootstrap node allocation for a refused helper read; it has
the same spot-to-on-demand Exception fallback as preparation. Cancellation still
runs once before preparation. Trusted helper changes use Jenkins' selected
revision directly. Host scripts and artifact metadata follow the pinned target
revision for PR authors outside `CONTRIBUTORS.md`.

Migrate each trusted target branch as a unit before testing untrusted PRs against
it. A target predating the shared helper cannot supply the fallback. An older
helper or monolithic entry point may also call a renamed host script before the
merge and trusted-file restoration, so a PR deleting that old path can fail
preparation. This includes the node-information command moved into `scripts/`.
A trusted branch build exercises the complete new layout; it does not establish
compatibility with an unmigrated target. Do not bypass these failures by loading
PR-owned helpers or a different target revision.

Before enabling the migration on production jobs, validate it with the existing
Jenkins plugin versions and sandbox settings on a trusted branch: run all five
entry points, suspend/resume across a node step, and exercise spot loss, a normal
build-command failure, superseded builds, and a failed on-demand retry. Check
trusted and untrusted PRs both with and without a common-helper change, and a
release-target PR. Verify image overrides and disabled deployment gates before
checking enabled docs/Docker deployment in the normal trusted deployment context.
Local mocked execution checks pipeline control flow but cannot certify Jenkins
CPS serialization, sandbox approvals, SCM trust configuration, or real agent loss.
No job path or controller reconfiguration is part of this migration.

# Infrastructure

The community's infrastructure configuration and documentation live in
[tlc-pack/ci](https://github.com/tlc-pack/ci). Coordinate external callers when
changing any retained script interface.
