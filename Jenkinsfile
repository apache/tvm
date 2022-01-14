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

// Hashtag in the source to build current CI docker builds
//
//
import org.jenkinsci.plugins.pipeline.modeldefinition.Utils

// NOTE: these lines are scanned by docker/dev_common.sh. Please update the regex as needed. -->
ci_lint = "tlcpack/ci-lint:v0.67"
ci_gpu = "tlcpackstaging/ci_gpu:20220113-230019-74854135a"
ci_cpu = "tlcpack/ci-cpu:v0.80"
ci_wasm = "tlcpack/ci-wasm:v0.71"
ci_i386 = "tlcpack/ci-i386:v0.74"
ci_qemu = "tlcpackstaging/ci_qemu:20220113-230019-74854135a"
ci_arm = "tlcpack/ci-arm:v0.06"
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
    string(name: 'ci_arm_param',  defaultValue: '')
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
  // Add more info about job node
  sh (
    script: """
     echo "INFO: NODE_NAME=${NODE_NAME} EXECUTOR_NUMBER=${EXECUTOR_NUMBER}"
     """,
     label: 'Show executor node info',
  )
  checkout scm
  retry(5) {
    timeout(time: 2, unit: 'MINUTES') {
      sh (script: 'git submodule update --init -f', label: 'Update git submodules')
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

stage('Prepare') {
  node('CPU') {
    // When something is provided in ci_*_param, use it, otherwise default with ci_*
    ci_lint = params.ci_lint_param ?: ci_lint
    ci_cpu = params.ci_cpu_param ?: ci_cpu
    ci_gpu = params.ci_gpu_param ?: ci_gpu
    ci_wasm = params.ci_wasm_param ?: ci_wasm
    ci_i386 = params.ci_i386_param ?: ci_i386
    ci_qemu = params.ci_qemu_param ?: ci_qemu
    ci_arm = params.ci_arm_param ?: ci_arm

    sh (script: """
      echo "Docker images being used in this build:"
      echo " ci_lint = ${ci_lint}"
      echo " ci_cpu  = ${ci_cpu}"
      echo " ci_gpu  = ${ci_gpu}"
      echo " ci_wasm = ${ci_wasm}"
      echo " ci_i386 = ${ci_i386}"
      echo " ci_qemu = ${ci_qemu}"
      echo " ci_arm  = ${ci_arm}"
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
        sh (
          script: "${docker_run} ${ci_lint}  ./tests/scripts/task_lint.sh",
          label: 'Run lint',
        )
      }
    }
  }
}

// Run make. First try to do an incremental make from a previous workspace in hope to
// accelerate the compilation. If something is wrong, clean the workspace and then
// build from scratch.
def make(docker_type, path, make_flag) {
  timeout(time: max_time, unit: 'MINUTES') {
    try {
      cmake_build(docker_type, path, make_flag)
      // always run cpp test when build
      cpp_unittest(docker_type)
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
    label: 'Run VTA tests in FSIM ',
  )
}

def cmake_build(image, path, make_flag) {
  sh (
    script: "${docker_run} ${image} ./tests/scripts/task_build.sh ${path} ${make_flag}",
    label: 'Run cmake build',
  )
}

def cpp_unittest(image) {
  sh (
    script: "${docker_run} ${image} ./tests/scripts/task_cpp_unittest.sh",
    label: 'Build and run C++ tests',
  )
}

stage('Build') {
    parallel 'BUILD: GPU': {
      if (!skip_ci) {
        node('GPUBUILD') {
          ws(per_exec_ws('tvm/build-gpu')) {
            init_git()
            sh "${docker_run} ${ci_gpu} ./tests/scripts/task_config_build_gpu.sh"
            make(ci_gpu, 'build', '-j2')
            pack_lib('gpu', tvm_multilib)
            // compiler test
            sh "${docker_run} ${ci_gpu} ./tests/scripts/task_config_build_gpu_other.sh"
            make(ci_gpu, 'build2', '-j2')
          }
        }
      }
  },
  'BUILD: CPU': {
    if (!skip_ci && is_docs_only_build != 1) {
      node('CPU') {
        ws(per_exec_ws('tvm/build-cpu')) {
          init_git()
          sh (
            script: "${docker_run} ${ci_cpu} ./tests/scripts/task_config_build_cpu.sh",
            label: 'Create CPU cmake config',
          )
          make(ci_cpu, 'build', '-j2')
          pack_lib('cpu', tvm_multilib_tsim)
          timeout(time: max_time, unit: 'MINUTES') {
            ci_setup(ci_cpu)
            // sh "${docker_run} ${ci_cpu} ./tests/scripts/task_golang.sh"
            // TODO(@jroesch): need to resolve CI issue will turn back on in follow up patch
            sh (script: "${docker_run} ${ci_cpu} ./tests/scripts/task_rust.sh", label: "Rust build and test")
          }
        }
      }
    } else {
      Utils.markStageSkippedForConditional('BUILD: CPU')
    }
  },
  'BUILD: WASM': {
    if (!skip_ci && is_docs_only_build != 1) {
      node('CPU') {
        ws(per_exec_ws('tvm/build-wasm')) {
          init_git()
          sh (
            script: "${docker_run} ${ci_wasm} ./tests/scripts/task_config_build_wasm.sh",
            label: 'Create WASM cmake config',
          )
          make(ci_wasm, 'build', '-j2')
          timeout(time: max_time, unit: 'MINUTES') {
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
  },
  'BUILD: i386': {
    if (!skip_ci && is_docs_only_build != 1) {
      node('CPU') {
        ws(per_exec_ws('tvm/build-i386')) {
          init_git()
          sh (
            script: "${docker_run} ${ci_i386} ./tests/scripts/task_config_build_i386.sh",
            label: 'Create i386 cmake config',
          )
          make(ci_i386, 'build', '-j2')
          pack_lib('i386', tvm_multilib_tsim)
        }
      }
    } else {
      Utils.markStageSkippedForConditional('BUILD: i386')
    }
  },
  'BUILD: arm': {
    if (!skip_ci && is_docs_only_build != 1) {
      node('ARM') {
        ws(per_exec_ws('tvm/build-arm')) {
          init_git()
          sh (
            script: "${docker_run} ${ci_arm} ./tests/scripts/task_config_build_arm.sh",
            label: 'Create ARM cmake config',
          )
          make(ci_arm, 'build', '-j4')
          pack_lib('arm', tvm_multilib)
        }
      }
     } else {
      Utils.markStageSkippedForConditional('BUILD: arm')
    }
  },
  'BUILD: QEMU': {
    if (!skip_ci && is_docs_only_build != 1) {
      node('CPU') {
        ws(per_exec_ws('tvm/build-qemu')) {
          init_git()
          sh (
            script: "${docker_run} ${ci_qemu} ./tests/scripts/task_config_build_qemu.sh",
            label: 'Create QEMU cmake config',
          )
          make(ci_qemu, 'build', '-j2')
          timeout(time: max_time, unit: 'MINUTES') {
            ci_setup(ci_qemu)
            sh (
              script: "${docker_run} ${ci_qemu} ./tests/scripts/task_python_microtvm.sh",
              label: 'Run microTVM tests',
            )
            junit 'build/pytest-results/*.xml'
          }
        }
      }
     } else {
      Utils.markStageSkippedForConditional('BUILD: QEMU')
    }
  }
}

stage('Test') {
    parallel 'unittest: GPU': {
      if (!skip_ci && is_docs_only_build != 1) {
        node('TensorCore') {
          ws(per_exec_ws('tvm/ut-python-gpu')) {
            init_git()
            unpack_lib('gpu', tvm_multilib)
            timeout(time: max_time, unit: 'MINUTES') {
              ci_setup(ci_gpu)
              sh (
                script: "${docker_run} ${ci_gpu} ./tests/scripts/task_sphinx_precheck.sh",
                label: 'Check Sphinx warnings in docs',
              )
              sh (
                script: "${docker_run} ${ci_gpu} ./tests/scripts/task_java_unittest.sh",
                label: 'Run Java unit tests',
              )
              sh (
                script: "${docker_run} ${ci_gpu} ./tests/scripts/task_python_unittest_gpuonly.sh",
                label: 'Run Python GPU unit tests',
              )
              sh (
                script: "${docker_run} ${ci_gpu} ./tests/scripts/task_python_integration_gpuonly.sh",
                label: 'Run Python GPU integration tests',
              )
              junit 'build/pytest-results/*.xml'
            }
          }
        }
      } else {
        Utils.markStageSkippedForConditional('unittest: GPU')
      }
    },
    'integration: CPU': {
      if (!skip_ci && is_docs_only_build != 1) {
        node('CPU') {
          ws(per_exec_ws('tvm/ut-python-cpu')) {
            init_git()
            unpack_lib('cpu', tvm_multilib_tsim)
            timeout(time: max_time, unit: 'MINUTES') {
              ci_setup(ci_cpu)
              sh (
                script: "${docker_run} ${ci_cpu} ./tests/scripts/task_python_integration.sh",
                label: 'Run CPU integration tests',
              )
              junit 'build/pytest-results/*.xml'
            }
          }
        }
      } else {
        Utils.markStageSkippedForConditional('integration: CPU')
      }
    },
    'unittest: CPU': {
      if (!skip_ci && is_docs_only_build != 1) {
        node('CPU') {
          ws(per_exec_ws("tvm/ut-python-cpu")) {
            init_git()
            unpack_lib('cpu', tvm_multilib_tsim)
            timeout(time: max_time, unit: 'MINUTES') {
              ci_setup(ci_cpu)
              python_unittest(ci_cpu)
              fsim_test(ci_cpu)
              sh (
                script: "${docker_run} ${ci_cpu} ./tests/scripts/task_python_vta_tsim.sh",
                label: "Run VTA tests in TSIM",
              )
              junit "build/pytest-results/*.xml"
            }
          }
        }
      } else {
        Utils.markStageSkippedForConditional('unittest: CPU')
      }
    },
    'python3: i386': {
      if (!skip_ci && is_docs_only_build != 1) {
        node('CPU') {
          ws(per_exec_ws('tvm/ut-python-i386')) {
            init_git()
            unpack_lib('i386', tvm_multilib)
            timeout(time: max_time, unit: 'MINUTES') {
              ci_setup(ci_i386)
              python_unittest(ci_i386)
              sh (
                script: "${docker_run} ${ci_i386} ./tests/scripts/task_python_integration_i386only.sh",
                label: 'Run i386 integration tests',
              )
              fsim_test(ci_i386)
              junit 'build/pytest-results/*.xml'
            }
          }
        }
     } else {
        Utils.markStageSkippedForConditional('python3: i386')
      }
    },
    'python3: arm': {
      if (!skip_ci && is_docs_only_build != 1) {
        node('ARM') {
          ws(per_exec_ws('tvm/ut-python-arm')) {
            init_git()
            unpack_lib('arm', tvm_multilib)
            timeout(time: max_time, unit: 'MINUTES') {
              ci_setup(ci_arm)
              python_unittest(ci_arm)
              sh (
                script: "${docker_run} ${ci_arm} ./tests/scripts/task_python_arm_compute_library.sh",
                label: 'Run test_arm_compute_lib test',
              )
              junit 'build/pytest-results/*.xml'
            // sh "${docker_run} ${ci_arm} ./tests/scripts/task_python_integration.sh"
            }
          }
        }
      } else {
         Utils.markStageSkippedForConditional('python3: arm')
      }
    },
  'topi: GPU': {
  if (!skip_ci && is_docs_only_build != 1) {
    node('GPU') {
      ws(per_exec_ws('tvm/topi-python-gpu')) {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          ci_setup(ci_gpu)
          sh (
            script: "${docker_run} ${ci_gpu} ./tests/scripts/task_python_topi.sh",
            label: 'Run TOPI tests',
          )
          junit 'build/pytest-results/*.xml'
        }
      }
    }
    } else {
      Utils.markStageSkippedForConditional('topi: GPU')
  }
  },
  'frontend: GPU': {
    if (!skip_ci && is_docs_only_build != 1) {
      node('GPU') {
        ws(per_exec_ws('tvm/frontend-python-gpu')) {
          init_git()
          unpack_lib('gpu', tvm_multilib)
          timeout(time: max_time, unit: 'MINUTES') {
            ci_setup(ci_gpu)
            sh (
              script: "${docker_run} ${ci_gpu} ./tests/scripts/task_python_frontend.sh",
              label: 'Run Python frontend tests',
            )
            junit 'build/pytest-results/*.xml'
          }
        }
      }
     } else {
      Utils.markStageSkippedForConditional('frontend: GPU')
    }
  },
  'frontend: CPU': {
    if (!skip_ci && is_docs_only_build != 1) {
      node('CPU') {
        ws(per_exec_ws('tvm/frontend-python-cpu')) {
          init_git()
          unpack_lib('cpu', tvm_multilib)
          timeout(time: max_time, unit: 'MINUTES') {
            ci_setup(ci_cpu)
            sh (
              script: "${docker_run} ${ci_cpu} ./tests/scripts/task_python_frontend_cpu.sh",
              label: 'Run Python frontend tests',
            )
            junit 'build/pytest-results/*.xml'
          }
        }
      }
    } else {
      Utils.markStageSkippedForConditional('frontend: CPU')
    }
  },
  'docs: GPU': {
    if (!skip_ci) {
      node('TensorCore') {
        ws(per_exec_ws('tvm/docs-python-gpu')) {
          init_git()
          unpack_lib('gpu', tvm_multilib)
          timeout(time: max_time, unit: 'MINUTES') {
            ci_setup(ci_gpu)
            sh (
              script: "${docker_run} ${ci_gpu} ./tests/scripts/task_python_docs.sh",
              label: 'Build docs',
            )
          }
          pack_lib('docs', 'docs.tgz')
        }
      }
    }
  }
}

/*
stage('Build packages') {
  parallel 'conda CPU': {
    node('CPU') {
      sh "${docker_run} tlcpack/conda-cpu ./conda/build_cpu.sh
    }
  },
  'conda cuda': {
    node('CPU') {
      sh "${docker_run} tlcpack/conda-cuda90 ./conda/build_cuda.sh
      sh "${docker_run} tlcpack/conda-cuda100 ./conda/build_cuda.sh
    }
  }
// Here we could upload the packages to anaconda for releases
// and/or the main branch
}
*/

stage('Deploy') {
    node('doc') {
      ws(per_exec_ws('tvm/deploy-docs')) {
        if (env.BRANCH_NAME == 'main') {
        unpack_lib('docs', 'docs.tgz')
        sh 'cp docs.tgz /var/docs/docs.tgz'
        sh 'tar xf docs.tgz -C /var/docs'
        }
      }
    }
}
