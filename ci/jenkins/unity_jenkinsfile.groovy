#!groovy
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

// ============================= IMPORTANT NOTE =============================
// To keep things simple
// This file is manually updated to maintain unity branch specific builds.
// Please do not send this file to main


import org.jenkinsci.plugins.pipeline.modeldefinition.Utils

// NOTE: these lines are scanned by docker/dev_common.sh. Please update the regex as needed. -->
ci_lint = 'tlcpack/ci-lint:20221025-182121-e41d0ed6e'
ci_gpu = 'tlcpack/ci-gpu:20221128-070141-ae4fd7df7'
ci_cpu = 'tlcpack/ci-cpu:20230110-070003-d00168ffb'
ci_wasm = 'tlcpack/ci-wasm:v0.72'
ci_i386 = 'tlcpack/ci-i386:v0.75'
ci_qemu = 'tlcpack/ci-qemu:v0.11'
ci_arm = 'tlcpack/ci-arm:v0.08'
ci_hexagon = 'tlcpack/ci-hexagon:20221025-182121-e41d0ed6e'
// <--- End of regex-scanned config.

// Parameters to allow overriding (in Jenkins UI), the images
// to be used by a given build. When provided, they take precedence
// over default values above.
properties([
  parameters([
    string(name: 'ci_lint_param', defaultValue: ''),
    string(name: 'ci_cpu_param',  defaultValue: ''),
    string(name: 'ci_gpu_param',  defaultValue: ''),
    string(name: 'ci_wasm_param', defaultValue: ''),
    string(name: 'ci_i386_param', defaultValue: ''),
    string(name: 'ci_qemu_param', defaultValue: ''),
    string(name: 'ci_arm_param',  defaultValue: ''),
    string(name: 'ci_hexagon_param', defaultValue: '')
  ])
])

// tvm libraries
tvm_runtime = 'build/libtvm_runtime.so, build/config.cmake'
tvm_lib = 'build/libtvm.so, ' + tvm_runtime
// LLVM upstream lib
tvm_multilib = 'build/libtvm.so, ' +
               'build/libvta_fsim.so, ' +
               tvm_runtime

tvm_multilib_tsim = 'build/libvta_tsim.so, ' +
               tvm_multilib

// command to start a docker container
docker_run = 'docker/bash.sh'
// timeout in minutes
max_time = 240

def per_exec_ws(folder) {
  return "workspace/exec_${env.EXECUTOR_NUMBER}/" + folder
}

// initialize source codes
def init_git() {
  checkout scm
  // Add more info about job node
  sh (
    script: './tests/scripts/task_show_node_info.sh',
    label: 'Show executor node info',
  )
  retry(5) {
    timeout(time: 2, unit: 'MINUTES') {
      sh (script: 'git submodule update --init -f', label: 'Update git submodules')
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
      script: "./tests/scripts/should_run_slow_tests.py --pr '${pr_number}'",
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

def should_skip_ci(pr_number) {
  withCredentials([string(
    credentialsId: 'tvm-bot-jenkins-reader',
    variable: 'TOKEN',
    )]) {
    // Exit code of 1 means run full CI (or the script had an error, so run
    // full CI just in case). Exit code of 0 means skip CI.
    git_skip_ci_code = sh (
      returnStatus: true,
      script: "./tests/scripts/git_skip_ci.py --pr '${pr_number}'",
      label: 'Check if CI should be skipped',
    )
    }
  return git_skip_ci_code == 0
}

cancel_previous_build()

def lint() {
stage('Prepare') {
  node('CPU-SMALL') {
    // When something is provided in ci_*_param, use it, otherwise default with ci_*
    ci_lint = params.ci_lint_param ?: ci_lint
    ci_cpu = params.ci_cpu_param ?: ci_cpu
    ci_gpu = params.ci_gpu_param ?: ci_gpu
    ci_wasm = params.ci_wasm_param ?: ci_wasm
    ci_i386 = params.ci_i386_param ?: ci_i386
    ci_qemu = params.ci_qemu_param ?: ci_qemu
    ci_arm = params.ci_arm_param ?: ci_arm
    ci_hexagon = params.ci_hexagon_param ?: ci_hexagon

    sh (script: """
      echo "Docker images being used in this build:"
      echo " ci_lint = ${ci_lint}"
      echo " ci_cpu  = ${ci_cpu}"
      echo " ci_gpu  = ${ci_gpu}"
      echo " ci_wasm = ${ci_wasm}"
      echo " ci_i386 = ${ci_i386}"
      echo " ci_qemu = ${ci_qemu}"
      echo " ci_arm  = ${ci_arm}"
      echo " ci_hexagon  = ${ci_hexagon}"
    """, label: 'Docker image names')
  }
}

stage('Sanity Check') {
  timeout(time: max_time, unit: 'MINUTES') {
    node('CPU') {
      ws(per_exec_ws('tvm/sanity')) {
        init_git()
        is_docs_only_build = sh (
          returnStatus: true,
          script: './tests/scripts/git_change_docs.sh',
          label: 'Check for docs only changes',
        )
        skip_ci = should_skip_ci(env.CHANGE_ID)
        skip_slow_tests = should_skip_slow_tests(env.CHANGE_ID)
        sh (
          script: "${docker_run} ${ci_lint}  ./tests/scripts/task_lint.sh",
          label: 'Run lint',
        )
        sh (
          script: "${docker_run} ${ci_lint}  ./tests/scripts/unity/task_extra_lint.sh",
          label: 'Run extra lint',
        )
      }
    }
  }
}
}

lint()

// Run make. First try to do an incremental make from a previous workspace in hope to
// accelerate the compilation. If something is wrong, clean the workspace and then
// build from scratch.
def make(docker_type, path, make_flag) {
  timeout(time: max_time, unit: 'MINUTES') {
    try {
      cmake_build(docker_type, path, make_flag)
      // always run cpp test when build
      // sh "${docker_run} ${docker_type} ./tests/scripts/task_cpp_unittest.sh"
    } catch (hudson.AbortException ae) {
      // script exited due to user abort, directly throw instead of retry
      if (ae.getMessage().contains('script returned exit code 143')) {
        throw ae
      }
      echo 'Incremental compilation failed. Fall back to build from scratch'
      sh (
        script: "${docker_run} ${docker_type} ./tests/scripts/task_clean.sh ${path}",
        label: 'Clear old cmake workspace',
      )
      cmake_build(docker_type, path, make_flag)
      cpp_unittest(docker_type)
    }
  }
}

// Specifications to Jenkins "stash" command for use with various pack_ and unpack_ functions.
tvm_runtime = 'build/libtvm_runtime.so, build/config.cmake'  // use libtvm_runtime.so.
tvm_lib = 'build/libtvm.so, ' + tvm_runtime  // use libtvm.so to run the full compiler.
// LLVM upstream lib
tvm_multilib = 'build/libtvm.so, ' +
               'build/libvta_fsim.so, ' +
               tvm_runtime

tvm_multilib_tsim = 'build/libvta_tsim.so, ' +
                    tvm_multilib

microtvm_tar_gz = 'build/microtvm_template_projects.tar.gz'

// pack libraries for later use
def pack_lib(name, libs) {
  sh (script: """
     echo "Packing ${libs} into ${name}"
     echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
     """, label: 'Stash libraries and show md5')
  stash includes: libs, name: name
}

// unpack libraries saved before
def unpack_lib(name, libs) {
  unstash name
  sh (script: """
     echo "Unpacked ${libs} from ${name}"
     echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
     """, label: 'Unstash libraries and show md5')
}

// compress microtvm template projects and pack the tar.
def pack_microtvm_template_projects(name) {
  sh(
    script: 'cd build && tar -czvf microtvm_template_projects.tar.gz microtvm_template_projects/',
    label: 'Compress microtvm_template_projects'
  )
  pack_lib(name + '-microtvm-libs', microtvm_tar_gz)
}

def unpack_microtvm_template_projects(name) {
  unpack_lib(name + '-microtvm-libs', microtvm_tar_gz)
  sh(
    script: 'cd build && tar -xzvf microtvm_template_projects.tar.gz',
    label: 'Unpack microtvm_template_projects'
  )
}

def ci_setup(image) {
  sh (
    script: "${docker_run} ${image} ./tests/scripts/task_ci_setup.sh",
    label: 'Set up CI environment',
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

def cmake_build(image, path, make_flag) {
  sh (
    script: "${docker_run} ${image} ./tests/scripts/task_build.py --sccache-bucket tvm-sccache-prod",
    label: 'Run cmake build',
  )
}

def cpp_unittest(image) {
  sh (
    script: "${docker_run} ${image} ./tests/scripts/task_cpp_unittest.sh",
    label: 'Build and run C++ tests',
  )
}

def add_hexagon_permissions() {
  sh(
    script: 'find build/hexagon_api_output -type f | xargs chmod +x',
    label: 'Add execute permissions for hexagon files',
  )
}

// NOTE: limit tests to relax folder for now to allow us to skip some of the tests
// that are mostly related to changes in main.
// This helps to speedup CI time and reduce CI cost.
stage('Build and Test') {
  if (is_docs_only_build != 1) {
    parallel 'BUILD: GPU': {
      node('GPU') {
        ws(per_exec_ws('tvm/build-gpu')) {
          init_git()
          sh "${docker_run} ${ci_gpu} nvidia-smi"
          sh "${docker_run}  ${ci_gpu} ./tests/scripts/task_config_build_gpu.sh build"
          make("${ci_gpu}", 'build', '-j2')
          sh "${docker_run} ${ci_gpu} ./tests/scripts/unity/task_python_relax_gpuonly.sh"
        }
      }
    },
    'BUILD: CPU': {
      node('CPU') {
        ws(per_exec_ws('tvm/build-cpu')) {
          init_git()
          sh "${docker_run} ${ci_cpu} ./tests/scripts/task_config_build_cpu.sh build"
          make(ci_cpu, 'build', '-j2')
          sh "${docker_run} ${ci_cpu} ./tests/scripts/unity/task_python_relax.sh"
        }
      }
    }
  } else {
    Utils.markStageSkippedForConditional('BUILD: CPU')
  }
}
