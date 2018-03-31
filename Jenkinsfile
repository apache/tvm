#!groovy
// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// tvm libraries
tvm_runtime = "lib/libtvm_runtime.so, config.mk"
tvm_lib = "lib/libtvm.so, " + tvm_runtime
// LLVM upstream lib
tvm_multilib = "lib/libtvm_llvm40.so, lib/libtvm_llvm50.so, lib/libtvm_llvm60.so, lib/libtvm_topi.so, " + tvm_runtime

// command to start a docker container
docker_run = 'tests/ci_build/ci_build.sh'
// timeout in minutes
max_time = 60

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
    node('linux') {
      ws('workspace/tvm/sanity') {
        init_git()
        sh "${docker_run} lint  ./tests/scripts/task_lint.sh"
      }
    }
  }
}

// Run make. First try to do an incremental make from a previous workspace in hope to
// accelerate the compilation. If something wrong, clean the workspace and then
// build from scratch.
def make(docker_type, make_flag) {
  timeout(time: max_time, unit: 'MINUTES') {
    try {
      sh "${docker_run} ${docker_type} make ${make_flag}"
    } catch (exc) {
      echo 'Incremental compilation failed. Fall back to build from scratch'
      sh "${docker_run} ${docker_type} make clean"
      sh "${docker_run} ${docker_type} make ${make_flag}"
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
  parallel 'GPU': {
    node('GPU' && 'linux') {
      ws('workspace/tvm/build-gpu') {
        init_git()
        sh """
           cp make/config.mk .
           echo USE_CUDNN=1 >> config.mk
           echo USE_CUDA=1 >> config.mk
           echo USE_OPENGL=1 >> config.mk
           echo LLVM_CONFIG=llvm-config-4.0 >> config.mk
           echo USE_RPC=1 >> config.mk
           echo USE_GRAPH_RUNTIME=1 >> config.mk
           echo USE_BLAS=openblas >> config.mk
           rm -f lib/libtvm_runtime.so lib/libtvm.so
           """
        make('gpu', '-j2')
        sh "mv lib/libtvm.so lib/libtvm_llvm40.so"
        sh "echo LLVM_CONFIG=llvm-config-5.0 >> config.mk"
        make('gpu', '-j2')
        sh "mv lib/libtvm.so lib/libtvm_llvm50.so"
        sh "echo LLVM_CONFIG=llvm-config-6.0 >> config.mk"
        make('gpu', '-j2')
        sh "mv lib/libtvm.so lib/libtvm_llvm60.so"
        pack_lib('gpu', tvm_multilib)
        sh """
           echo USE_OPENCL=1 >> config.mk
           echo USE_ROCM=1 >> config.mk
           echo ROCM_PATH=/opt/rocm >> config.mk
           echo USE_VULKAN=1 >> config.mk
           """
        make('gpu', '-j2')
      }
    }
  },
  'CPU': {
    node('CPU' && 'linux') {
      ws('workspace/tvm/build-cpu') {
        init_git()
        sh """
           cp make/config.mk .
           echo USE_CUDA=0 >> config.mk
           echo USE_OPENCL=0 >> config.mk
           echo USE_RPC=0 >> config.mk
           echo USE_OPENGL=1 >> config.mk
           echo LLVM_CONFIG=llvm-config-4.0 >> config.mk
           """
        make('cpu', '-j2')
        pack_lib('cpu', tvm_lib)
      }
    }
  },
  'i386': {
    node('CPU' && 'linux') {
      ws('workspace/tvm/build-i386') {
        init_git()
        sh """
           cp make/config.mk .
           echo USE_CUDA=0 >> config.mk
           echo USE_OPENCL=0 >> config.mk
           echo LLVM_CONFIG=llvm-config-4.0 >> config.mk
           echo USE_RPC=1 >> config.mk
           """
        make('i386', '-j2')
        sh "mv lib/libtvm.so lib/libtvm_llvm40.so"
        sh "echo LLVM_CONFIG=llvm-config-5.0 >> config.mk"
        make('i386', '-j2')
        sh "mv lib/libtvm.so lib/libtvm_llvm50.so"
        sh "echo LLVM_CONFIG=llvm-config-6.0 >> config.mk"
        make('i386', '-j2')
        sh "mv lib/libtvm.so lib/libtvm_llvm60.so"
        pack_lib('i386', tvm_multilib)
      }
    }
  }
}

stage('Unit Test') {
  parallel 'python2/3: GPU': {
    node('GPU' && 'linux') {
      ws('workspace/tvm/ut-python-gpu') {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        sh "cp lib/libtvm_llvm40.so lib/libtvm.so"
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} gpu ./tests/scripts/task_python_unittest.sh"
        }
        // Test on the lastest mainline.
        sh "cp lib/libtvm_llvm60.so lib/libtvm.so"
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} gpu ./tests/scripts/task_python_unittest.sh"
        }
      }
    }
  },
  'python2/3: i386': {
    node('CPU' && 'linux') {
      ws('workspace/tvm/ut-python-i386') {
        init_git()
        unpack_lib('i386', tvm_multilib)
        sh "cp lib/libtvm_llvm40.so lib/libtvm.so"
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} i386 ./tests/scripts/task_python_unittest.sh"
          sh "${docker_run} i386 ./tests/scripts/task_python_integration.sh"
        }
        // Test on llvm 5.0
        sh "cp lib/libtvm_llvm50.so lib/libtvm.so"
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} i386 ./tests/scripts/task_python_integration.sh"
        }
      }
    }
  },
  'cpp': {
    node('CPU' && 'linux') {
      ws('workspace/tvm/ut-cpp') {
        init_git()
        unpack_lib('cpu', tvm_lib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} cpu ./tests/scripts/task_cpp_unittest.sh"
        }
      }
    }
  },
  'java': {
    node('GPU' && 'linux') {
      ws('workspace/tvm/ut-java') {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        sh "cp lib/libtvm_llvm40.so lib/libtvm.so"
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} gpu ./tests/scripts/task_java_unittest.sh"
        }
      }
    }
  }
}

stage('Integration Test') {
  parallel 'python': {
    node('GPU' && 'linux') {
      ws('workspace/tvm/it-python-gpu') {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        sh "cp lib/libtvm_llvm40.so lib/libtvm.so"
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} gpu ./tests/scripts/task_python_integration.sh"
          sh "${docker_run} gpu ./tests/scripts/task_python_topi.sh"
          sh "${docker_run} gpu ./tests/scripts/task_cpp_topi.sh"
        }
      }
    }
  },
  'docs': {
    node('GPU' && 'linux') {
      ws('workspace/tvm/docs-python-gpu') {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        sh "cp lib/libtvm_llvm40.so lib/libtvm.so"
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} gpu ./tests/scripts/task_python_docs.sh"
        }
        pack_lib('mydocs', 'docs.tgz')
      }
    }
  }
}

stage('Deploy') {
    node('docker' && 'doc') {
      ws('workspace/tvm/deploy-docs') {
        if (env.BRANCH_NAME == "master") {
           unpack_lib('mydocs', 'docs.tgz')
           sh "tar xf docs.tgz -C /var/docs"
        }
      }
    }
}
