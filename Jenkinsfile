// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// tvm libraries
tvm_lib = 'lib/libtvm.so, lib/libtvm_runtime.so, config.mk'
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
def pack_lib(name, libs=tvm_lib) {
  sh """
     echo "Packing ${libs} into ${name}"
     echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
     """
  stash includes: libs, name: name
}


// unpack libraries saved before
def unpack_lib(name, libs=tvm_lib) {
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
           echo USE_CUDA=1 >> config.mk
           echo USE_OPENCL=1 >> config.mk
           echo LLVM_CONFIG=llvm-config-4.0 >> config.mk
           echo USE_RPC=1 >> config.mk
           echo USE_BLAS=openblas >> config.mk
           """
        make('gpu', '-j4')
        pack_lib('gpu')
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
           """
        make('cpu', '-j4')
        pack_lib('cpu')
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
        make('i386', '-j4')
        pack_lib('i386')
      }
    }
  }
}

stage('Unit Test') {
  parallel 'python2/3: GPU': {
    node('GPU' && 'linux') {
      ws('workspace/tvm/ut-python-gpu') {
        init_git()
        unpack_lib('gpu', tvm_lib)
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
        unpack_lib('i386', tvm_lib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} i386 ./tests/scripts/task_python_unittest.sh"
          sh "${docker_run} i386 ./tests/scripts/task_python_integration.sh"
        }
      }
    }
  },
  'cpp': {
    node('linux') {
      ws('workspace/tvm/ut-cpp') {
        init_git()
        unpack_lib('cpu', tvm_lib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} cpu ./tests/scripts/task_cpp_unittest.sh"
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
        unpack_lib('gpu')
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} gpu ./tests/scripts/task_python_integration.sh"
          sh "${docker_run} gpu ./tests/scripts/task_python_topi.sh"
        }
      }
    }
  },
  'docs': {
    node('GPU' && 'linux') {
      ws('workspace/tvm/docs-python-gpu') {
        init_git()
        unpack_lib('gpu')
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
