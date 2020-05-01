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
// - Tag the docker image with a new version and push to tvmai
// - Update the version in the Jenkinsfile, send a PR
// - Fix any issues wrt to the new image version in the PR
// - Merge the PR and now we are in new version
// - Tag the new version as the lates
// - Periodically cleanup the old versions on local workers
//

// Hashtag in the source to build current CI docker builds
//
//

ci_lint = "tvmai/ci-lint:v0.60"
ci_gpu = "tvmai/ci-gpu:v0.63"
ci_cpu = "tvmai/ci-cpu:v0.61"
ci_i386 = "tvmai/ci-i386:v0.52"

// tvm libraries
tvm_runtime = "build/libtvm_runtime.so, build/config.cmake"
tvm_lib = "build/libtvm.so, " + tvm_runtime
// LLVM upstream lib
tvm_multilib = "build/libtvm.so, " +
               "build/libvta_tsim.so, " +
               "build/libvta_fsim.so, " +
               "build/libtvm_topi.so, " +
               tvm_runtime

// command to start a docker container
docker_run = 'docker/bash.sh'
// timeout in minutes
max_time = 120

def per_exec_ws(folder) {
  return "workspace/exec_${env.EXECUTOR_NUMBER}/" + folder
}

// initialize source codes
def init_git() {
  // Add more info about job node
  sh """
     echo "INFO: NODE_NAME=${NODE_NAME} EXECUTOR_NUMBER=${EXECUTOR_NUMBER}"
     """
  checkout scm
  retry(5) {
    timeout(time: 2, unit: 'MINUTES') {
      sh 'git submodule update --init'
    }
  }
}

def init_git_win() {
    checkout scm
    retry(5) {
        timeout(time: 2, unit: 'MINUTES') {
            bat 'git submodule update --init'
        }
    }
}

stage("Sanity Check") {
  timeout(time: max_time, unit: 'MINUTES') {
    node('CPU') {
      ws(per_exec_ws("tvm/sanity")) {
        init_git()
        sh "${docker_run} ${ci_lint}  ./tests/scripts/task_lint.sh"
      }
    }
  }
}

// Run make. First try to do an incremental make from a previous workspace in hope to
// accelerate the compilation. If something wrong, clean the workspace and then
// build from scratch.
def make(docker_type, path, make_flag) {
  timeout(time: max_time, unit: 'MINUTES') {
    try {
      sh "${docker_run} ${docker_type} ./tests/scripts/task_build.sh ${path} ${make_flag}"
      // always run cpp test when build
      sh "${docker_run} ${docker_type} ./tests/scripts/task_cpp_unittest.sh"
    } catch (exc) {
      echo 'Incremental compilation failed. Fall back to build from scratch'
      sh "${docker_run} ${docker_type} ./tests/scripts/task_clean.sh ${path}"
      sh "${docker_run} ${docker_type} ./tests/scripts/task_build.sh ${path} ${make_flag}"
      sh "${docker_run} ${docker_type} ./tests/scripts/task_cpp_unittest.sh"
    }
  }
}

// pack libraries for later use
def pack_lib(name, libs) {
  sh """
     echo "Packing ${libs} into ${name}"
     echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
     """
  stash includes: libs, name: name
}


// unpack libraries saved before
def unpack_lib(name, libs) {
  unstash name
  sh """
     echo "Unpacked ${libs} from ${name}"
     echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
     """
}

stage('Build') {
  parallel 'BUILD: GPU': {
    node('GPUBUILD') {
      ws(per_exec_ws("tvm/build-gpu")) {
        init_git()
        sh "${docker_run} ${ci_gpu} ./tests/scripts/task_config_build_gpu.sh"
        make(ci_gpu, 'build', '-j2')
        pack_lib('gpu', tvm_multilib)
        // compiler test
        sh "${docker_run} ${ci_gpu} ./tests/scripts/task_config_build_gpu_vulkan.sh"
        make(ci_gpu, 'build2', '-j2')
      }
    }
  },
  'BUILD: CPU': {
    node('CPU') {
      ws(per_exec_ws("tvm/build-cpu")) {
        init_git()
        sh "${docker_run} ${ci_cpu} ./tests/scripts/task_config_build_cpu.sh"
        make(ci_cpu, 'build', '-j2')
        pack_lib('cpu', tvm_lib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} ${ci_cpu} ./tests/scripts/task_python_unittest.sh"
          sh "${docker_run} ${ci_cpu} ./tests/scripts/task_python_integration.sh"
          sh "${docker_run} ${ci_cpu} ./tests/scripts/task_python_vta_fsim.sh"
          sh "${docker_run} ${ci_cpu} ./tests/scripts/task_python_vta_tsim.sh"
          sh "${docker_run} ${ci_cpu} ./tests/scripts/task_golang.sh"
          sh "${docker_run} ${ci_cpu} ./tests/scripts/task_rust.sh"
        }
      }
    }
  },
  'BUILD : i386': {
    node('CPU') {
      ws(per_exec_ws("tvm/build-i386")) {
        init_git()
        sh "${docker_run} ${ci_i386} ./tests/scripts/task_config_build_i386.sh"
        make(ci_i386, 'build', '-j2')
        pack_lib('i386', tvm_multilib)
      }
    }
  }
}

stage('Unit Test') {
  parallel 'python3: GPU': {
    node('TensorCore') {
      ws(per_exec_ws("tvm/ut-python-gpu")) {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} ${ci_gpu} ./tests/scripts/task_sphinx_precheck.sh"
          sh "${docker_run} ${ci_gpu} ./tests/scripts/task_python_unittest.sh"
          sh "${docker_run} ${ci_gpu} ./tests/scripts/task_python_integration.sh"
        }
      }
    }
  },
  'python3: i386': {
    node('CPU') {
      ws(per_exec_ws("tvm/ut-python-i386")) {
        init_git()
        unpack_lib('i386', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} ${ci_i386} ./tests/scripts/task_python_unittest.sh"
          sh "${docker_run} ${ci_i386} ./tests/scripts/task_python_integration.sh"
          sh "${docker_run} ${ci_i386} ./tests/scripts/task_python_vta_fsim.sh"
        }
      }
    }
  },
  'java: GPU': {
    node('GPU') {
      ws(per_exec_ws("tvm/ut-java")) {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} ${ci_gpu} ./tests/scripts/task_java_unittest.sh"
        }
      }
    }
  }
}

stage('Integration Test') {
  parallel 'topi: GPU': {
    node('GPU') {
      ws(per_exec_ws("tvm/topi-python-gpu")) {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} ${ci_gpu} ./tests/scripts/task_python_topi.sh"
        }
      }
    }
  },
  'frontend: GPU': {
    node('GPU') {
      ws(per_exec_ws("tvm/frontend-python-gpu")) {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} ${ci_gpu} ./tests/scripts/task_python_frontend.sh"
        }
      }
    }
  },
  'docs: GPU': {
    node('GPU') {
      ws(per_exec_ws("tvm/docs-python-gpu")) {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} ${ci_gpu} ./tests/scripts/task_python_docs.sh"
        }
        pack_lib('mydocs', 'docs.tgz')
      }
    }
  }
}

/*
stage('Build packages') {
  parallel 'conda CPU': {
    node('CPU') {
      sh "${docker_run} tvmai/conda-cpu ./conda/build_cpu.sh
    }
  },
  'conda cuda': {
    node('CPU') {
      sh "${docker_run} tvmai/conda-cuda90 ./conda/build_cuda.sh
      sh "${docker_run} tvmai/conda-cuda100 ./conda/build_cuda.sh
    }
  }
  // Here we could upload the packages to anaconda for releases
  // and/or the master branch
}
*/

stage('Deploy') {
    node('doc') {
      ws(per_exec_ws("tvm/deploy-docs")) {
        if (env.BRANCH_NAME == "master") {
           unpack_lib('mydocs', 'docs.tgz')
           sh "cp docs.tgz /var/docs/docs.tgz"
           sh "tar xf docs.tgz -C /var/docs"
        }
      }
    }
}
