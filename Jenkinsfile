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
ci_lint = "tvmai/ci-lint:v0.51"
ci_gpu = "tvmai/ci-gpu:v0.54"
ci_cpu = "tvmai/ci-cpu:v0.52"
ci_i386 = "tvmai/ci-i386:v0.52"

// tvm libraries
tvm_runtime = "build/libtvm_runtime.so, build/config.cmake"
tvm_lib = "build/libtvm.so, " + tvm_runtime
// LLVM upstream lib
tvm_multilib = "build/libtvm.so, " +
               "build/libvta_tsim.so, " +
               "build/libvta_fsim.so, " +
               "build/libtvm_topi.so, " +
               "build/libnnvm_compiler.so, " + tvm_runtime

// command to start a docker container
docker_run = 'docker/bash.sh'
// timeout in minutes
max_time = 120

// initialize source codes
def init_git() {
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
      ws('workspace/tvm/sanity') {
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
      ws('workspace/tvm/build-gpu') {
        init_git()
        sh """
           mkdir -p build
           cd build
           cp ../cmake/config.cmake .
           echo set\\(USE_CUBLAS ON\\) >> config.cmake
           echo set\\(USE_CUDNN ON\\) >> config.cmake
           echo set\\(USE_CUDA ON\\) >> config.cmake
           echo set\\(USE_OPENGL ON\\) >> config.cmake
           echo set\\(USE_MICRO ON\\) >> config.cmake
           echo set\\(USE_LLVM llvm-config-7\\) >> config.cmake
           echo set\\(USE_NNPACK ON\\) >> config.cmake
           echo set\\(NNPACK_PATH /NNPACK/build/\\) >> config.cmake
           echo set\\(USE_RPC ON\\) >> config.cmake
           echo set\\(USE_SORT ON\\) >> config.cmake
           echo set\\(USE_GRAPH_RUNTIME ON\\) >> config.cmake
           echo set\\(USE_STACKVM_RUNTIME ON\\) >> config.cmake
           echo set\\(USE_GRAPH_RUNTIME_DEBUG ON\\) >> config.cmake
           echo set\\(USE_VM_PROFILER ON\\) >> config.cmake
           echo set\\(USE_ANTLR ON\\) >> config.cmake
           echo set\\(USE_BLAS openblas\\) >> config.cmake
           echo set\\(CMAKE_CXX_COMPILER g++\\) >> config.cmake
           echo set\\(CMAKE_CXX_FLAGS -Werror\\) >> config.cmake
           """
        make(ci_gpu, 'build', '-j2')
        pack_lib('gpu', tvm_multilib)
        // compiler test
        sh """
           mkdir -p build2
           cd build2
           cp ../cmake/config.cmake .
           echo set\\(USE_OPENCL ON\\) >> config.cmake
           echo set\\(USE_ROCM ON\\) >> config.cmake
           echo set\\(USE_VULKAN ON\\) >> config.cmake
           echo set\\(USE_MICRO ON\\) >> config.cmake
           echo set\\(USE_GRAPH_RUNTIME_DEBUG ON\\) >> config.cmake
           echo set\\(USE_VM_PROFILER ON\\) >> config.cmake
           echo set\\(CMAKE_CXX_COMPILER clang-7\\) >> config.cmake
           echo set\\(CMAKE_CXX_FLAGS -Werror\\) >> config.cmake
           """
        make(ci_gpu, 'build2', '-j2')
      }
    }
  },
  'BUILD: CPU': {
    node('CPU') {
      ws('workspace/tvm/build-cpu') {
        init_git()
        sh """
           mkdir -p build
           cd build
           cp ../cmake/config.cmake .
           echo set\\(USE_SORT ON\\) >> config.cmake
           echo set\\(USE_MICRO ON\\) >> config.cmake
           echo set\\(USE_GRAPH_RUNTIME_DEBUG ON\\) >> config.cmake
           echo set\\(USE_VM_PROFILER ON\\) >> config.cmake
           echo set\\(USE_LLVM llvm-config-8\\) >> config.cmake
           echo set\\(USE_NNPACK ON\\) >> config.cmake
           echo set\\(NNPACK_PATH /NNPACK/build/\\) >> config.cmake
           echo set\\(USE_ANTLR ON\\) >> config.cmake
           echo set\\(CMAKE_CXX_COMPILER g++\\) >> config.cmake
           echo set\\(CMAKE_CXX_FLAGS -Werror\\) >> config.cmake
           echo set\\(HIDE_PRIVATE_SYMBOLS ON\\) >> config.cmake
           """
        make(ci_cpu, 'build', '-j2')
        pack_lib('cpu', tvm_lib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} ${ci_cpu} ./tests/scripts/task_golang.sh"
          sh "${docker_run} ${ci_cpu} ./tests/scripts/task_python_unittest.sh"
          sh "${docker_run} ${ci_cpu} ./tests/scripts/task_python_integration.sh"
          sh "${docker_run} ${ci_cpu} ./tests/scripts/task_python_vta.sh"
        }
      }
    }
  },
  'BUILD : i386': {
    node('CPU') {
      ws('workspace/tvm/build-i386') {
        init_git()
        sh """
           mkdir -p build
           cd build
           cp ../cmake/config.cmake .
           echo set\\(USE_SORT ON\\) >> config.cmake
           echo set\\(USE_RPC ON\\) >> config.cmake
           echo set\\(USE_GRAPH_RUNTIME_DEBUG ON\\) >> config.cmake
           echo set\\(USE_VM_PROFILER ON\\) >> config.cmake
           echo set\\(USE_LLVM llvm-config-4.0\\) >> config.cmake
           echo set\\(CMAKE_CXX_COMPILER g++\\) >> config.cmake
           echo set\\(CMAKE_CXX_FLAGS -Werror\\) >> config.cmake
           """
        make(ci_i386, 'build', '-j2')
        pack_lib('i386', tvm_multilib)
      }
    }
  }
}

stage('Unit Test') {
  parallel 'python3: GPU': {
    node('GPU') {
      ws('workspace/tvm/ut-python-gpu') {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} ${ci_gpu} ./tests/scripts/task_python_unittest.sh"
          sh "${docker_run} ${ci_gpu} ./tests/scripts/task_python_integration.sh"
        }
      }
    }
  },
  'python3: i386': {
    node('CPU') {
      ws('workspace/tvm/ut-python-i386') {
        init_git()
        unpack_lib('i386', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} ${ci_i386} ./tests/scripts/task_python_unittest.sh"
          sh "${docker_run} ${ci_i386} ./tests/scripts/task_python_integration.sh"
          sh "${docker_run} ${ci_i386} ./tests/scripts/task_python_vta.sh"
        }
      }
    }
  },
  'java: GPU': {
    node('GPU') {
      ws('workspace/tvm/ut-java') {
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
      ws('workspace/tvm/topi-python-gpu') {
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
      ws('workspace/tvm/frontend-python-gpu') {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} ${ci_gpu} ./tests/scripts/task_python_frontend.sh"
        }
      }
    }
  },
  'legacy: GPU': {
    node('GPU') {
      ws('workspace/tvm/legacy-python-gpu') {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} ${ci_gpu} ./tests/scripts/task_python_legacy.sh"
        }
      }
    }
  },
  'docs: GPU': {
    node('GPU') {
      ws('workspace/tvm/docs-python-gpu') {
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
      ws('workspace/tvm/deploy-docs') {
        if (env.BRANCH_NAME == "master") {
           unpack_lib('mydocs', 'docs.tgz')
           sh "tar xf docs.tgz -C /var/docs"
        }
      }
    }
}
