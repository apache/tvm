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

// Reject untrusted helper edits before allocating nodes or running pipeline steps.
ci = evaluate(readTrusted('ci/jenkins/generated/common.groovy'))
ci.start()

stage('Build') {
  ci.with_node(instance: 'CPU-SPOT', retry_instance: 'CPU',
    name: 'BUILD: GPU', folder: 'tvm/build-gpu', variables: ['PLATFORM=gpu']) {
    ci.docker_init(images.ci_gpu)
    sh "${docker_run} --no-gpu ${images.ci_gpu} ./tests/scripts/task_config_build_gpu.sh build"
    sh (
      script: "${docker_run} --env CI_NUM_EXECUTORS ${images.ci_gpu} --no-gpu ./tests/scripts/task_build.py --sccache-bucket tvm-sccache-prod --sccache-region us-west-2 --build-dir build",
      label: 'Run CMake build',
    )
    ci.upload_artifacts('gpu', '--artifact tvm_lib --artifact tvm_lib_gpu_extra')

    // Check the compiler-only GPU configuration after publishing the test build.
    sh 'rm -rf build'
    sh "${docker_run} --no-gpu ${images.ci_gpu} ./tests/scripts/task_config_build_gpu_other.sh build"
    sh "${docker_run} --env CI_NUM_EXECUTORS ${images.ci_gpu} --no-gpu ./tests/scripts/task_build.py --sccache-bucket tvm-sccache-prod --sccache-region us-west-2 --cmake-target tvm_runtime --build-dir build"
  }
}

stage('Test') {
  parallel(
    'unittest: GPU': {
      ci.with_node(instance: 'GPU-SPOT', retry_instance: 'GPU',
        name: 'unittest: GPU', folder: 'tvm/ut-python-gpu',
        variables: ['PLATFORM=gpu', 'TEST_STEP_NAME=unittest: GPU']) {
        ci.docker_init(images.ci_gpu)
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
      ci.with_node(instance: 'GPU-SPOT', retry_instance: 'GPU',
        name: 'docs: GPU', folder: 'tvm/docs-python-gpu',
        variables: ['PLATFORM=gpu', 'TEST_STEP_NAME=docs: GPU']) {
        ci.docker_init(images.ci_gpu)
        ci.download('gpu')
        sh(
          script: "${docker_run} ${images.ci_gpu} ./tests/scripts/task_python_docs.sh",
          label: 'Build docs',
        )
        ci.upload_artifacts('docs', '--path docs.tgz')
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
          script: "./${jenkins_scripts_root}/s3.py --action download --bucket ${s3_bucket} --prefix ${s3_prefix}/docs --path docs.tgz",
          label: 'Download docs folder from S3',
        )
        deploy_docs()
      }
    })
  }
}
