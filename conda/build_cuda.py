# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import sys
import subprocess

from jinja2 import Template

CUDA_VERSIONS = ['10.0', '9.0']


# Make sure that the cudnn version you set here is available
# for all the cuda versions that you want both from nvidia
# and from conda.

# These two must be in sync
CUDNN_FULL_VERSION = '7.3.1.20'
CUDNN_VERSION = '7.3.1'


condadir = os.path.dirname(sys.argv[0])
condadir = os.path.abspath(condadir)
srcdir = os.path.dirname(condadir)


with open(os.path.join(condadir, 'Dockerfile.template')) as f:
    docker_template = Template(f.read())


def render_dockerfile(version):
    txt = docker_template.render(cuda_version=version,
                                 cudnn_short_version=CUDNN_VERSION,
                                 cudnn_version=CUDNN_FULL_VERSION)
    fname = os.path.join(condadir,
                         'Dockerfile.cuda' + version.replace('.', ''))
    with open(fname, 'w') as f:
        f.write(txt)
    return fname


def build_docker(version):
    vv = version.replace('.', '')
    fname = render_dockerfile(version)
    tagname = f'tvm-cuda{ vv }-forge'
    subprocess.run(['docker', 'build', '-t', tagname,
                    condadir, '-f', fname], check=True)
    return tagname


def build_pkg(version):
    tagname = build_docker(version)
    subprocess.run(['docker', 'run', '--rm', '-v', f'{ srcdir }:/workspace',
                    tagname], check=True)


if __name__ == '__main__':
    build_versions = CUDA_VERSIONS
    if len(sys.argv) > 1:
        build_versions = sys.argv[1:]
    for version in build_versions:
        build_pkg(version)
