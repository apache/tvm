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
from __future__ import print_function
import sys
import os.path, re, StringIO

blacklist = [
    'Windows.h',
    'mach/clock.h', 'mach/mach.h',
    'malloc.h',
    'glog/logging.h', 'io/azure_filesys.h', 'io/hdfs_filesys.h', 'io/s3_filesys.h',
    'sys/stat.h', 'sys/types.h',
    'omp.h', 'execinfo.h', 'packet/sse-inl.h'
    ]


def get_sources(def_file):
    sources = []
    files = []
    visited = set()
    mxnet_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    for line in open(def_file):
        files = files + line.strip().split(' ')

    for f in files:
        f = f.strip()
        if not f or f.endswith('.o:') or f == '\\': continue
        fn = os.path.relpath(f)
        if os.path.abspath(f).startswith(mxnet_path) and fn not in visited:
            sources.append(fn)
            visited.add(fn)
    return sources

sources = get_sources(sys.argv[1])

def find_source(name, start):
    candidates = []
    for x in sources:
        if x == name or x.endswith('/' + name): candidates.append(x)
    if not candidates: return ''
    if len(candidates) == 1: return candidates[0]
    for x in candidates:
        if x.split('/')[1] == start.split('/')[1]: return x
    return ''


re1 = re.compile('<([./a-zA-Z0-9_-]*)>')
re2 = re.compile('"([./a-zA-Z0-9_-]*)"')

sysheaders = []
history = set([])
out = StringIO.StringIO()

def expand(x, pending):
    if x in history and x not in ['mshadow/mshadow/expr_scalar-inl.h']: # MULTIPLE includes
        return

    if x in pending:
        #print('loop found: %s in ' % x, pending)
        return

    print("//===== EXPANDING: %s =====\n" % x, file=out)
    for line in open(x):
        if line.find('#include') < 0:
            out.write(line)
            continue
        if line.strip().find('#include') > 0:
            print(line)
            continue
        m = re1.search(line)
        if not m: m = re2.search(line)
        if not m:
            print(line + ' not found')
            continue
        h = m.groups()[0].strip('./')
        source = find_source(h, x)
        if not source:
            if (h not in blacklist and
                h not in sysheaders and
                'mkl' not in h and
                'nnpack' not in h): sysheaders.append(h)
        else:
            expand(source, pending + [x])
    print("//===== EXPANDED: %s =====\n" % x, file=out)
    history.add(x)


expand(sys.argv[2], [])

f = open(sys.argv[3], 'wb')



for k in sorted(sysheaders):
    print("#include <%s>" % k, file=f)

print('', file=f)
print(out.getvalue(), file=f)

for x in sources:
    if x not in history and not x.endswith('.o'):
        print('Not processed:', x)

