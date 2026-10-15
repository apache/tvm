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

import org.jenkinsci.plugins.pipeline.modeldefinition.Utils

// Keep the image names aligned with docker-images.ini. Tags are resolved in prepare.
image_platforms = [ci_arm: 'ARM', ci_cpu: 'CPU', ci_gpu: 'GPU', ci_wasm: 'CPU']
images = image_platforms.collectEntries { name, platform -> [(name): ''] }
built_images = [:]
upstream_revision = null
max_time = 180
rebuild_docker_images = false
docker_run = 'docker/bash.sh --env CI --env PLATFORM --env TEST_STEP_NAME'
docker_build = 'docker/build.sh'
s3_bucket = 'tvm-jenkins-artifacts-prod'
s3_prefix = "tvm/${env.BRANCH_NAME}/${env.BUILD_NUMBER}"
jenkins_scripts_root = 'ci/jenkins/scripts'

properties([parameters(image_platforms.collect { name, platform ->
  string(name: "${name}_param", defaultValue: '')
})])

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
    script: './ci/jenkins/scripts/show_node_info.sh',
    label: 'Show executor node info',
  )

  // Determine merge commit to use for all stages
  if (env.CHANGE_TARGET) {
    // This is a PR build, so merge with the latest of the PR's target branch
    // (e.g. main or a release branch like v0.25.0).
    merge_with_target()
  } else {
    // This is a branch build (main or a release branch). Only set
    // upstream_revision to HEAD and skip merging to avoid a race with another
    // commit merged to the branch.
    update_upstream_revision("HEAD")
  }

  sh(
    script: """
      set -eux
      . ${jenkins_scripts_root}/retry.sh
      retry 3 timeout 5m git submodule update --init --recursive -f --jobs 0
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

def merge_with_target() {
  def target = env.CHANGE_TARGET
  sh (
    script: "git fetch origin ${target}",
    label: "Fetch target branch ${target}",
  )
  update_upstream_revision("FETCH_HEAD")
  sh (
    script: "git -c user.name=TVM-Jenkins -c user.email=jenkins@tvm.apache.org merge ${upstream_revision}",
    label: "Merge to origin/${target}"
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

def is_last_build() {
  // check whether it is last build
  try {
    return currentBuild.number == currentBuild.rawBuild.project.getLastBuild().number
  } catch (Throwable ex) {
    echo 'Error during check is_last_build ' + ex.toString()
    return false
  }
}

def checkout_trusted_files() {
  // trust everything from branch builds
  if (env.BRANCH_NAME == null || !env.BRANCH_NAME.startsWith('PR-')) {
    return;
  }

  // Trust contributors listed in the pinned target revision.
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
      script: "git checkout ${upstream_revision} ${jenkins_scripts_root}/. ci/jenkins/data.py",
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

def prepare(node_type) {
  stage('Prepare') {
    node(node_type) {
      ws("workspace/exec_${env.EXECUTOR_NUMBER}/tvm/prepare") {
        init_git()

        if (env.DETERMINE_DOCKER_IMAGES == 'yes') {
          sh(
            script: "./${jenkins_scripts_root}/determine_docker_images.py ${images.keySet().join(' ')}",
            label: 'Decide whether to use tlcpack or tlcpackstaging for Docker images',
          )
          images.keySet().each { name ->
            images[name] = sh(
              script: "cat .docker-image-names/${name}",
              label: "Find docker image name for ${name}",
              returnStdout: true,
            ).trim()
          }
        }
        images.keySet().each { name -> images[name] = params["${name}_param"] ?: images[name] }
        sh(
          script: 'echo "Docker images being used in this build:"\n' +
            images.collect { name, image -> "echo \" ${name} = ${image}\"" }.join('\n'),
          label: 'Docker image names',
        )

        skip_ci = should_skip_ci(env.CHANGE_ID)
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
def python_unittest(image) {
  sh (
    script: "${docker_run} ${image} ./tests/scripts/task_python_unittest.sh",
    label: 'Run Python unit tests',
  )
}

def make_cpp_tests(image, build_dir) {
  sh (
    script: """
      set -eux
      ${docker_run} ${image} python3 ./tests/scripts/task_build.py \
        --sccache-bucket tvm-sccache-prod \
        --sccache-region us-west-2 \
        --cmake-target cpptest \
        --build-dir ${build_dir}
      """,
    label: 'Make C++ tests',
  )
}

def cmake_build(image, path) {
  sh (
    script: "${docker_run} --env CI_NUM_EXECUTORS ${image} ./tests/scripts/task_build.py --sccache-bucket tvm-sccache-prod --sccache-region us-west-2 --build-dir ${path}",
    label: 'Run CMake build',
  )
}
def cpp_unittest(image) {
  sh (
    script: "${docker_run} --env CI_NUM_EXECUTORS ${image} ./tests/scripts/task_cpp_unittest.sh",
    label: 'Run C++ tests',
  )
}

// Preparation retries Exceptions unconditionally; Errors and cancellation of the
// previous build are deliberately outside this fallback.
def start() {
  cancel_previous_build()
  try {
    prepare('CPU-SMALL-SPOT')
  } catch (Exception ex) {
    prepare('CPU-SMALL')
  }
}

// A failed build command is final. Infrastructure failures retry only while
// this is the newest build; tests intentionally also retry AbortException.
def build(label, name, folder, image, platform, body) {
  stage('Build') {
    try {
      run_on_node("${label}-SPOT", name, folder, image, ["PLATFORM=${platform}"], body)
    } catch (hudson.AbortException ex) {
      echo 'Received normal AbortException, exit now. Details:' + ex.toString()
      throw ex
    } catch (Throwable ex) {
      retry_latest(ex) {
        run_on_node(label, name, folder, image, ["PLATFORM=${platform}"], body)
      }
    }
  }
}

def retry_latest(ex, body) {
  echo 'Exception during SPOT run ' + ex.toString()
  if (!is_last_build()) {
    echo 'Exit since it is not last build'
    throw ex
  }
  echo 'Retry on-demand given it is last build'
  currentBuild.result = 'SUCCESS'
  // Do not catch failure of the on-demand retry.
  body()
}

def test(label, name, folder, image, platform, body) {
  def variables = ["PLATFORM=${platform}", "TEST_STEP_NAME=${name}"]
  try {
    run_on_node("${label}-SPOT", name, folder, image, variables, body)
  } catch (Throwable ex) {
    retry_latest(ex) { run_on_node(label, name, folder, image, variables, body) }
  }
}

def run_on_node(label, name, folder, image, variables, body) {
  if (skip_ci) {
    Utils.markStageSkippedForConditional(name)
    return
  }
  echo 'Begin running on node_type ' + label
  node(label) {
    ws(per_exec_ws(folder)) {
      init_git()
      docker_init(image)
      timeout(time: max_time, unit: 'MINUTES') {
        withEnv(variables) { body() }
      }
    }
  }
  echo 'End running on node_type ' + label
}

def upload(tag, options) {
  sh(
    script: "./${jenkins_scripts_root}/s3.py --action upload --bucket ${s3_bucket} --prefix ${s3_prefix}/${tag} ${options}",
    label: 'Upload artifacts to S3',
  )
}

def download(tag) {
  sh(
    script: "./${jenkins_scripts_root}/s3.py --action download --bucket ${s3_bucket} --prefix ${s3_prefix}/${tag}",
    label: 'Download artifacts from S3',
  )
}

def deploy_step(name, enabled, folder, body) {
  if (enabled) {
    node('CPU') {
      ws(per_exec_ws(folder)) {
        timeout(time: max_time, unit: 'MINUTES') { body() }
      }
    }
  } else {
    Utils.markStageSkippedForConditional(name)
  }
}

return this
