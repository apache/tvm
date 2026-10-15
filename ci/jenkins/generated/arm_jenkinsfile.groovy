// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// Keep trusted branches on their selected helper; recover rejected PR helpers
// only from the fixed upstream main ref. Never evaluate the checkout's copy.
def helperPath = 'ci/jenkins/generated/common.groovy'
def helperSource
try {
  helperSource = readTrusted(helperPath)
} catch (hudson.AbortException failure) {
  def rejected = failure.message == "‘${helperPath}’ has been modified in an untrusted revision"
  def missing = failure.message?.endsWith("/${helperPath} not found")
  if (!env.CHANGE_ID || !(rejected || missing)) {
    throw failure
  }
} catch (java.io.FileNotFoundException failure) {
  if (!env.CHANGE_ID || failure.message != helperPath) {
    throw failure
  }
}
if (helperSource == null) {
  node('CPU-SMALL') {
    helperSource = sh(
      returnStdout: true,
      label: 'Read trusted helper from upstream main',
      script: '''
        set -eu
        helper_repo=$(mktemp -d)
        trap 'rm -rf "$helper_repo"' EXIT
        git init --bare --quiet "$helper_repo"
        git --git-dir="$helper_repo" fetch --quiet --no-tags --depth=1 \
          https://github.com/apache/tvm.git +refs/heads/main:refs/heads/trusted-main
        revision=$(git --git-dir="$helper_repo" rev-parse --verify 'refs/heads/trusted-main^{commit}')
        git --git-dir="$helper_repo" show "$revision:ci/jenkins/generated/common.groovy"
      '''
    )
  }
}
// Evaluation and subsequent pipeline failures must never trigger a fallback.
ci = evaluate(helperSource)
ci.start()

ci.build('ARM-GRAVITON3', 'BUILD: arm', 'tvm/build-arm', images.ci_arm, 'arm') {
  sh(
    script: "${docker_run} ${images.ci_arm} ./tests/scripts/task_config_build_arm.sh build",
    label: 'Create ARM CMake config',
  )
  ci.cmake_build(images.ci_arm, 'build')
  ci.make_cpp_tests(images.ci_arm, 'build')
  ci.upload('arm', '--bundle tvm_lib --bundle cpptest')
}
