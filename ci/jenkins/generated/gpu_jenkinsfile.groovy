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

ci.build('CPU', 'BUILD: GPU', 'tvm/build-gpu', images.ci_gpu, 'gpu') {
  sh "${docker_run} --no-gpu ${images.ci_gpu} ./tests/scripts/task_config_build_gpu.sh build"
  ci.cmake_build("${images.ci_gpu} --no-gpu", 'build')
  ci.upload('gpu', '--bundle tvm_lib --bundle tvm_lib_gpu_extra')

  // Check the compiler-only GPU configuration after publishing the test build.
  sh 'rm -rf build'
  sh "${docker_run} --no-gpu ${images.ci_gpu} ./tests/scripts/task_config_build_gpu_other.sh build"
  sh "${docker_run} --env CI_NUM_EXECUTORS ${images.ci_gpu} --no-gpu ./tests/scripts/task_build.py --sccache-bucket tvm-sccache-prod --sccache-region us-west-2 --cmake-target tvm_runtime --build-dir build"
}

stage('Test') {
  parallel(
    'unittest: GPU': {
      ci.test('GPU', 'unittest: GPU', 'tvm/ut-python-gpu', images.ci_gpu, 'gpu') {
        ci.download('gpu')
        sh(
          script: "${docker_run} ${images.ci_gpu} ./tests/scripts/task_java_unittest.sh",
          label: 'Run Java unit tests',
        )
        sh(
          script: "${docker_run} ${images.ci_gpu} ./tests/scripts/task_python_unittest_gpuonly.sh",
          label: 'Run Python GPU unit tests',
        )
      }
    },
    'docs: GPU': {
      ci.test('GPU', 'docs: GPU', 'tvm/docs-python-gpu', images.ci_gpu, 'gpu') {
        ci.download('gpu')
        sh(
          script: "${docker_run} ${images.ci_gpu} ./tests/scripts/task_python_docs.sh",
          label: 'Build docs',
        )
        ci.upload('docs', '--items docs.tgz')
        sh(
          script: "aws s3 cp --no-progress _docs s3://${s3_bucket}/${s3_prefix}/docs --recursive",
          label: 'Upload docs to S3',
        )
      }
    },
  )
}

def deploy_docs() {
  // Note: This code must stay in the Jenkinsfile to ensure that it runs
  // from a trusted context only
  sh(
    script: '''
      set -eux
      rm -rf tvm-site
      git clone -b $DOCS_DEPLOY_BRANCH --depth=1 https://github.com/apache/tvm-site
      cd tvm-site
      git status
      git checkout -B $DOCS_DEPLOY_BRANCH

      git ls-tree HEAD docs/ --name-only | grep -vP '^docs/v\\d' | xargs rm -rf
      mkdir -p docs
      tar xf ../docs.tgz -C docs
      COMMIT=$(cat docs/commit_hash)
      git add .
      git config user.name tvm-bot
      git config user.email 95660001+tvm-bot@users.noreply.github.com
      git commit -m"deploying docs (apache/tvm@$COMMIT)"
      git status
    ''',
    label: 'Unpack docs and update tvm-site'
  )

  withCredentials([string(
    credentialsId: 'docs-push-token',
    variable: 'GITHUB_TOKEN',
    )]) {
    sh(
      script: '''
        cd tvm-site
        git remote add deploy https://$GITHUB_TOKEN:x-oauth-basic@github.com/apache/tvm-site.git
        git push deploy $DOCS_DEPLOY_BRANCH || true
      ''',
      label: 'Upload docs to apache/tvm-site'
    )
  }
}

stage('Deploy') {
  if (env.BRANCH_NAME == 'main') {
    parallel('Deploy Docs': {
      ci.deploy_step('Deploy Docs', env.DOCS_DEPLOY_ENABLED == 'yes', 'tvm/deploy-docs') {
        ci.init_git()
        sh(
          script: "./${jenkins_scripts_root}/s3.py --action download --bucket ${s3_bucket} --prefix ${s3_prefix}/docs --items docs.tgz",
          label: 'Download docs folder from S3',
        )
        deploy_docs()
      }
    })
  }
}
