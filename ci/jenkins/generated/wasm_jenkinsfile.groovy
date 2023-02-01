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
// -*- mode: groovy -*-

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

// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// Docker env used for testing
// Different image may have different version tag
// because some of them are more stable than anoter.
//
// Docker images are maintained by PMC, cached in dockerhub
// and remains relatively stable over the time.
// Flow for upgrading docker env(need commiter)
//
// - Send PR to upgrade build script in the repo
// - Build the new docker image
// - Tag the docker image with a new version and push to a binary cache.
// - Update the version in the Jenkinsfile, send a PR
// - Fix any issues wrt to the new image version in the PR
// - Merge the PR and now we are in new version
// - Tag the new version as the lates
// - Periodically cleanup the old versions on local workers
//

// unity: Skip less relevant tests
// to reduce ci time and resource cost
// (DO NOT UPSTREAM TO MAIN)
return

// ============================= IMPORTANT NOTE =============================
// This file is generated by 'jenkins/generate.py'. Do not edit this file directly!
// Make edits to 'jenkins/Jenkinsfile.j2' and regenerate this with
// 'python3 jenkins/generate.py'
// Note: This timestamp is here to ensure that updates to the Jenkinsfile are
// always rebased on main before merging:
// Generated at 2022-12-09T15:39:24.526394

import org.jenkinsci.plugins.pipeline.modeldefinition.Utils
// These are set at runtime from data in ci/jenkins/docker-images.yml, update
// image tags in that file
ci_lint = ''
ci_gpu = ''
ci_cpu = ''
ci_minimal = ''
ci_wasm = ''
ci_i386 = ''
ci_cortexm = ''
ci_arm = ''
ci_hexagon = ''
ci_riscv = ''

// Parameters to allow overriding (in Jenkins UI), the images
// to be used by a given build. When provided, they take precedence
// over default values above.
properties([
  parameters([
    string(name: 'ci_arm_param', defaultValue: ''),
    string(name: 'ci_cortexm_param', defaultValue: ''),
    string(name: 'ci_cpu_param', defaultValue: ''),
    string(name: 'ci_gpu_param', defaultValue: ''),
    string(name: 'ci_hexagon_param', defaultValue: ''),
    string(name: 'ci_i386_param', defaultValue: ''),
    string(name: 'ci_lint_param', defaultValue: ''),
    string(name: 'ci_minimal_param', defaultValue: ''),
    string(name: 'ci_riscv_param', defaultValue: ''),
    string(name: 'ci_wasm_param', defaultValue: ''),
  ])
])

// Placeholders for newly built Docker image names (if rebuild_docker_images
// is used)
  built_ci_arm = null;
  built_ci_cortexm = null;
  built_ci_cpu = null;
  built_ci_gpu = null;
  built_ci_hexagon = null;
  built_ci_i386 = null;
  built_ci_lint = null;
  built_ci_minimal = null;
  built_ci_riscv = null;
  built_ci_wasm = null;

// Global variable assigned during Sanity Check that holds the sha1 which should be
// merged into the PR in all branches.
upstream_revision = null

// command to start a docker container
docker_run = 'docker/bash.sh --env CI --env TVM_SHARD_INDEX --env TVM_NUM_SHARDS --env RUN_DISPLAY_URL --env PLATFORM --env SKIP_SLOW_TESTS --env TEST_STEP_NAME'
docker_build = 'docker/build.sh'
// timeout in minutes
max_time = 180
rebuild_docker_images = false

s3_bucket = 'tvm-jenkins-artifacts-prod'
s3_prefix = "tvm/${env.BRANCH_NAME}/${env.BUILD_NUMBER}"

// Jenkins script root directory
jenkins_scripts_root = "ci/scripts/jenkins"


// General note: Jenkins has limits on the size of a method (or top level code)
// that are pretty strict, so most usage of groovy methods in these templates
// are purely to satisfy the JVM
def per_exec_ws(folder) {
  return "workspace/exec_${env.EXECUTOR_NUMBER}/" + folder
}

// initialize source codes
def init_git() {
  retry(5) {
    checkout scm
  }

  // Add more info about job node
  sh (
    script: './tests/scripts/task_show_node_info.sh',
    label: 'Show executor node info',
  )

  // Determine merge commit to use for all stages
  if (env.BRANCH_NAME == 'main') {
    // Only set upstream_revision to HEAD and skip merging to avoid a race with another commit merged to main.
    update_upstream_revision("HEAD")
  } else {
    // This is PR branch so merge with latest main.
    merge_with_main()
  }

  sh(
    script: """
      set -eux
      . ${jenkins_scripts_root}/retry.sh
      retry 3 timeout 5m git submodule update --init -f --jobs 0
    """,
    label: 'Update git submodules',
  )
  checkout_trusted_files()
}

def update_upstream_revision(git_ref) {
  if (upstream_revision == null) {
    upstream_revision = sh(
      script: "git log -1 ${git_ref} --format=\'%H\'",
      label: 'Determine upstream revision',
      returnStdout: true,
    ).trim()
  }
}

def merge_with_main() {
  sh (
    script: 'git fetch origin main',
    label: 'Fetch upstream',
  )
  update_upstream_revision("FETCH_HEAD")
  sh (
    script: "git -c user.name=TVM-Jenkins -c user.email=jenkins@tvm.apache.org merge ${upstream_revision}",
    label: 'Merge to origin/main'
  )
}

def docker_init(image) {
  // Clear out all Docker images that aren't going to be used
  sh(
    script: """
    set -eux
    docker image ls --all
    IMAGES=\$(docker image ls --all --format '{{.Repository}}:{{.Tag}}  {{.ID}}')

    echo -e "Found images:\\n\$IMAGES"
    echo "\$IMAGES" | { grep -vE '${image}' || test \$? = 1; } | { xargs docker rmi || test \$? = 123; }

    docker image ls --all
    """,
    label: 'Clean old Docker images',
  )

  if (image.contains("amazonaws.com")) {
    // If this string is in the image name it's from ECR and needs to be pulled
    // with the right credentials
    ecr_pull(image)
  } else {
    sh(
      script: """
      set -eux
      . ${jenkins_scripts_root}/retry.sh
      retry 5 docker pull ${image}
      """,
      label: 'Pull docker image',
    )
  }
}

def ecr_pull(full_name) {
  aws_account_id = sh(
    returnStdout: true,
    script: 'aws sts get-caller-identity | grep Account | cut -f4 -d\\"',
    label: 'Get AWS ID'
  ).trim()

  try {
    withEnv([
      "AWS_ACCOUNT_ID=${aws_account_id}",
      'AWS_DEFAULT_REGION=us-west-2',
      "AWS_ECR_REPO=${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com"]) {
      sh(
        script: '''
          set -eux
          aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ECR_REPO
        ''',
        label: 'Log in to ECR'
      )
      sh(
        script: """
          set -eux
          . ${jenkins_scripts_root}/retry.sh
          retry 5 docker pull ${full_name}
        """,
        label: 'Pull image from ECR'
      )
    }
  } finally {
    withEnv([
      "AWS_ACCOUNT_ID=${aws_account_id}",
      'AWS_DEFAULT_REGION=us-west-2',
      "AWS_ECR_REPO=${aws_account_id}.dkr.ecr.us-west-2.amazonaws.com"]) {
      sh(
        script: 'docker logout $AWS_ECR_REPO',
        label: 'Clean up login credentials'
      )
    }
  }
}

def should_skip_slow_tests(pr_number) {
  withCredentials([string(
    credentialsId: 'tvm-bot-jenkins-reader',
    variable: 'GITHUB_TOKEN',
  )]) {
    // Exit code of 1 means run slow tests, exit code of 0 means skip slow tests
    result = sh (
      returnStatus: true,
      script: "./${jenkins_scripts_root}/should_run_slow_tests.py --pr '${pr_number}'",
      label: 'Check if CI should run slow tests',
    )
  }
  return result == 0
}

def cancel_previous_build() {
  // cancel previous build if it is not on main.
  if (env.BRANCH_NAME != 'main') {
    def buildNumber = env.BUILD_NUMBER as int
    // Milestone API allows us to cancel previous build
    // with the same milestone number
    if (buildNumber > 1) milestone(buildNumber - 1)
    milestone(buildNumber)
  }
}

def checkout_trusted_files() {
  // trust everything from branch builds
  if (env.BRANCH_NAME == null || !env.BRANCH_NAME.startsWith('PR-')) {
    return;
  }

  // trust peoople listed in CONTRIBUTING.md
  grep_code = sh(
    returnStatus: true,
    script: "git show '${upstream_revision}:CONTRIBUTORS.md' | grep '@${env.CHANGE_AUTHOR}'",
    label: 'Check if change is from a contributor',
  )

  if (grep_code == 1) {
    // Any scripts that run on the bare host and not inside a Docker container
    // (especially those that access secrets) should be checked out here so
    // only trusted versions are used in CI
    sh(
      script: "git checkout ${upstream_revision} ${jenkins_scripts_root}/.",
      label: 'Check out trusted files',
    )
  }
}

def should_skip_ci(pr_number) {
  if (env.BRANCH_NAME == null || !env.BRANCH_NAME.startsWith('PR-')) {
    // never skip CI on build sourced from a branch
    return false
  }
  glob_skip_ci_code = sh (
    returnStatus: true,
    script: "./${jenkins_scripts_root}/git_skip_ci_globs.py",
    label: 'Check if CI should be skipped due to changed files',
  )
  if (glob_skip_ci_code == 0) {
    return true
  }
  withCredentials([string(
    credentialsId: 'tvm-bot-jenkins-reader',
    variable: 'GITHUB_TOKEN',
    )]) {
    // Exit code of 1 means run full CI (or the script had an error, so run
    // full CI just in case). Exit code of 0 means skip CI.
    git_skip_ci_code = sh (
      returnStatus: true,
      script: "./${jenkins_scripts_root}/git_skip_ci.py --pr '${pr_number}'",
      label: 'Check if CI should be skipped',
    )
  }
  return git_skip_ci_code == 0
}

def check_pr(pr_number) {
  if (env.BRANCH_NAME == null || !env.BRANCH_NAME.startsWith('PR-')) {
    // never skip CI on build sourced from a branch
    return false
  }
  withCredentials([string(
    credentialsId: 'tvm-bot-jenkins-reader',
    variable: 'GITHUB_TOKEN',
    )]) {
    sh (
      script: "python3 ${jenkins_scripts_root}/check_pr.py --pr ${pr_number}",
      label: 'Check PR title and body',
    )
  }

}

def prepare() {
  stage('Prepare') {
    node('CPU-SMALL') {
      ws("workspace/exec_${env.EXECUTOR_NUMBER}/tvm/prepare") {
        init_git()

        check_pr(env.CHANGE_ID)

        if (env.DETERMINE_DOCKER_IMAGES == 'yes') {
          sh(
            script: "./${jenkins_scripts_root}/determine_docker_images.py ci_arm ci_cortexm ci_cpu ci_gpu ci_hexagon ci_i386 ci_lint ci_minimal ci_riscv ci_wasm ",
            label: 'Decide whether to use tlcpack or tlcpackstaging for Docker images',
          )
          // Pull image names from the results of should_rebuild_docker.py
          ci_arm = sh(
            script: "cat .docker-image-names/ci_arm",
            label: "Find docker image name for ci_arm",
            returnStdout: true,
          ).trim()
          ci_cortexm = sh(
            script: "cat .docker-image-names/ci_cortexm",
            label: "Find docker image name for ci_cortexm",
            returnStdout: true,
          ).trim()
          ci_cpu = sh(
            script: "cat .docker-image-names/ci_cpu",
            label: "Find docker image name for ci_cpu",
            returnStdout: true,
          ).trim()
          ci_gpu = sh(
            script: "cat .docker-image-names/ci_gpu",
            label: "Find docker image name for ci_gpu",
            returnStdout: true,
          ).trim()
          ci_hexagon = sh(
            script: "cat .docker-image-names/ci_hexagon",
            label: "Find docker image name for ci_hexagon",
            returnStdout: true,
          ).trim()
          ci_i386 = sh(
            script: "cat .docker-image-names/ci_i386",
            label: "Find docker image name for ci_i386",
            returnStdout: true,
          ).trim()
          ci_lint = sh(
            script: "cat .docker-image-names/ci_lint",
            label: "Find docker image name for ci_lint",
            returnStdout: true,
          ).trim()
          ci_minimal = sh(
            script: "cat .docker-image-names/ci_minimal",
            label: "Find docker image name for ci_minimal",
            returnStdout: true,
          ).trim()
          ci_riscv = sh(
            script: "cat .docker-image-names/ci_riscv",
            label: "Find docker image name for ci_riscv",
            returnStdout: true,
          ).trim()
          ci_wasm = sh(
            script: "cat .docker-image-names/ci_wasm",
            label: "Find docker image name for ci_wasm",
            returnStdout: true,
          ).trim()
        }

        ci_arm = params.ci_arm_param ?: ci_arm
        ci_cortexm = params.ci_cortexm_param ?: ci_cortexm
        ci_cpu = params.ci_cpu_param ?: ci_cpu
        ci_gpu = params.ci_gpu_param ?: ci_gpu
        ci_hexagon = params.ci_hexagon_param ?: ci_hexagon
        ci_i386 = params.ci_i386_param ?: ci_i386
        ci_lint = params.ci_lint_param ?: ci_lint
        ci_minimal = params.ci_minimal_param ?: ci_minimal
        ci_riscv = params.ci_riscv_param ?: ci_riscv
        ci_wasm = params.ci_wasm_param ?: ci_wasm

        sh (script: """
          echo "Docker images being used in this build:"
          echo " ci_arm = ${ci_arm}"
          echo " ci_cortexm = ${ci_cortexm}"
          echo " ci_cpu = ${ci_cpu}"
          echo " ci_gpu = ${ci_gpu}"
          echo " ci_hexagon = ${ci_hexagon}"
          echo " ci_i386 = ${ci_i386}"
          echo " ci_lint = ${ci_lint}"
          echo " ci_minimal = ${ci_minimal}"
          echo " ci_riscv = ${ci_riscv}"
          echo " ci_wasm = ${ci_wasm}"
        """, label: 'Docker image names')

        is_docs_only_build = sh (
          returnStatus: true,
          script: "./${jenkins_scripts_root}/git_change_docs.sh",
          label: 'Check for docs only changes',
        )
        skip_ci = should_skip_ci(env.CHANGE_ID)
        skip_slow_tests = should_skip_slow_tests(env.CHANGE_ID)
        rebuild_docker_images = sh (
          returnStatus: true,
          script: "./${jenkins_scripts_root}/git_change_docker.sh",
          label: 'Check for any docker changes',
        )

        if (skip_ci) {
          // Don't rebuild when skipping CI
          rebuild_docker_images = false
        }
      }
    }
  }
}
def ci_setup(image) {
  sh (
    script: "${docker_run} ${image} ./tests/scripts/task_clear_pytest.sh",
    label: 'Clean up old workspace',
  )
}

def python_unittest(image) {
  sh (
    script: "${docker_run} ${image} ./tests/scripts/task_python_unittest.sh",
    label: 'Run Python unit tests',
  )
}

def fsim_test(image) {
  sh (
    script: "${docker_run} ${image} ./tests/scripts/task_python_vta_fsim.sh",
    label: 'Run VTA tests in FSIM',
  )
}

def make_standalone_crt(image, build_dir) {
  sh (
    script: """
      set -eux
      ${docker_run} ${image} python3 ./tests/scripts/task_build.py \
        --sccache-bucket tvm-sccache-prod \
        --cmake-target standalone_crt \
        --build-dir build
      ${docker_run} ${image} python3 ./tests/scripts/task_build.py \
        --sccache-bucket tvm-sccache-prod \
        --cmake-target crttest \
        --build-dir build
      """,
    label: 'Make standalone CRT',
  )
}

def make_cpp_tests(image, build_dir) {
  sh (
    script: """
      set -eux
      ${docker_run} ${image} python3 ./tests/scripts/task_build.py \
        --sccache-bucket tvm-sccache-prod \
        --cmake-target cpptest \
        --build-dir ${build_dir}
      """,
    label: 'Make C++ tests',
  )
}

def cmake_build(image, path, make_flag) {
  sh (
    script: "${docker_run} --env CI_NUM_EXECUTORS ${image} ./tests/scripts/task_build.py --sccache-bucket tvm-sccache-prod",
    label: 'Run cmake build',
  )
}
def cpp_unittest(image) {
  sh (
    script: "${docker_run} --env CI_NUM_EXECUTORS ${image} ./tests/scripts/task_cpp_unittest.sh",
    label: 'Run C++ tests',
  )
}

def micro_cpp_unittest(image) {
  sh (
    script: "${docker_run} --env CI_NUM_EXECUTORS ${image} ./tests/scripts/task_microtvm_cpp_tests.sh build",
    label: 'Run microTVM C++ tests',
  )
}

cancel_previous_build()

prepare()
def build() {
  stage('Build') {
    if (!skip_ci && is_docs_only_build != 1) {
      node('CPU-SMALL') {
        ws("workspace/exec_${env.EXECUTOR_NUMBER}/tvm/build-wasm") {
          init_git()
          docker_init(ci_wasm)
          timeout(time: max_time, unit: 'MINUTES') {
            sh (
          script: "${docker_run} ${ci_wasm} ./tests/scripts/task_config_build_wasm.sh build",
          label: 'Create WASM cmake config',
        )
        cmake_build(ci_wasm, 'build', '-j2')
        make_standalone_crt(ci_wasm, 'build')
        make_cpp_tests(ci_wasm, 'build')
        cpp_unittest(ci_wasm)
        ci_setup(ci_wasm)
        sh (
          script: "${docker_run} ${ci_wasm} ./tests/scripts/task_web_wasm.sh",
          label: 'Run WASM lint and tests',
        )
          }
        }
      }
    } else {
      Utils.markStageSkippedForConditional('BUILD: WASM')
    }
  }
}
build()
