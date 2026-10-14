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

// readTrusted pins trusted revisions, but rejects helper edits in untrusted PRs.
// In that case use the PR target, pinning the same commit for all source merges.
def helper
String target_revision = null
try {
  helper = readTrusted('ci/jenkins/generated/common.groovy')
} catch (hudson.AbortException | java.io.FileNotFoundException ex) {
  if (!env.CHANGE_TARGET) throw ex
  echo 'Loading shared pipeline from the pinned PR target: ' + ex.toString()
  def from_target = { label ->
    node(label) {
      ws("workspace/exec_${env.EXECUTOR_NUMBER}/tvm/pipeline") {
        // No PR-owned executable or credentials run during this bootstrap.
        deleteDir()
        retry(5) { checkout scm }
        // Quote the branch as data and fetch only from the canonical repository.
        def target = env.CHANGE_TARGET.replace("'", "'\"'\"'")
        sh(script: "git fetch https://github.com/apache/tvm.git 'refs/heads/${target}'",
           label: 'Fetch trusted pipeline target')
        def revision = sh(script: 'git rev-parse FETCH_HEAD', returnStdout: true).trim()
        def source = sh(script: "git show ${revision}:ci/jenkins/generated/common.groovy",
                        returnStdout: true)
        return [revision, source]
      }
    }
  }
  def pinned
  try {
    pinned = from_target('CPU-SMALL-SPOT')
  } catch (Exception failure) {
    pinned = from_target('CPU-SMALL')
  }
  target_revision = pinned[0]
  helper = pinned[1]
}
// Evaluation is outside the read/fetch fallback: script errors must propagate.
ci = evaluate(helper)
if (target_revision != null) ci.upstream_revision = target_revision
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
