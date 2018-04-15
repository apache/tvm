#!groovy
// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// nnvm libraries
vta_lib += "lib/libvta.so, lib/libvta.so.json"
vta_lib += ", nnvm/tvm/lib/libtvm.so, nnvm/tvm/lib/libtopi.so, nnvm/lib/libnnvm_compiler.so"


// command to start a docker container
docker_run = 'tests/ci_build/ci_build.sh'
// timeout in minutes
max_time = 60

// initialize source codes
def init_git() {
  checkout scm
  retry(5) {
    timeout(time: 2, unit: 'MINUTES') {
      sh 'git submodule update --init --recursive'
    }
  }
}

def init_git_win() {
    checkout scm
    retry(5) {
        timeout(time: 2, unit: 'MINUTES') {
            bat 'git submodule update --init --recursive'
        }
    }
}

stage("Sanity Check") {
  timeout(time: max_time, unit: 'MINUTES') {
    node('linux') {
      ws('workspace/vta/sanity') {
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
    sh "${docker_run} ${docker_type} cp make/sim_sample.json config.json"
    try {
      sh "${docker_run} ${docker_type} ./tests/scripts/task_build.sh ${make_flag}"
    } catch (exc) {
      echo 'Incremental compilation failed. Fall back to build from scratch'
      sh "${docker_run} ${docker_type} ./tests/scripts/task_clean.sh"
      sh "${docker_run} ${docker_type} ./tests/scripts/task_build.sh ${make_flag}"
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
  timeout(time: max_time, unit: 'MINUTES') {
    node('linux') {
      ws('workspace/vta/build') {
        init_git()
        make('cpu', '-j2')
        pack_lib('cpu', vta_lib)
      }
    }
  }
}

stage('Tests') {
  parallel 'python': {
    node('linux') {
      ws('workspace/vta/it-python') {
        init_git()
        unpack_lib('cpu', vta_lib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} cpu ./tests/scripts/task_python_test.sh"
        }
      }
    }
  },
  'docs': {
    node('linux') {
      ws('workspace/vta/docs-python') {
        init_git()
        unpack_lib('cpu', vta_lib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} cpu ./tests/scripts/task_python_docs.sh"
        }
        pack_lib('mydocs', 'docs.tgz')
      }
    }
  }
}

stage('Deploy') {
    node('docker' && 'doc') {
      ws('workspace/vta/deploy-docs') {
        if (env.BRANCH_NAME == "master") {
           unpack_lib('mydocs', 'docs.tgz')
           sh "tar xf docs.tgz -C /var/vta-docs"
        }
      }
    }
}
