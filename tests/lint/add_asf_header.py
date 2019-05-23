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

"""Helper tool to add ASF header to files that cannot be handled by Rat."""
import os
import sys

header_cstyle = """
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
""".strip()

header_mdstyle = """
<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->
""".strip()

header_pystyle = """
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
""".strip()

header_rststyle = """
..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
""".strip()

header_groovystyle = """
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
""".strip()

FMT_MAP = {
    "sh" : header_pystyle,
    "cc" : header_cstyle,
    "h" : header_cstyle,
    "py" : header_pystyle,
    "toml" : header_pystyle,
    "yml": header_pystyle,
    "yaml": header_pystyle,
    "rs" : header_cstyle,
    "md" : header_mdstyle,
    "cmake" : header_pystyle,
    "rst" : header_rststyle,
    "gradle" : header_groovystyle,
    "xml": header_mdstyle,
}

def add_header(fname, header):
    """Add header to file"""
    if not os.path.exists(fname):
        print("Cannot find %s ..." % fname)
        return

    orig = open(fname).read()
    if orig.find("Licensed to the Apache Software Foundation") != -1:
        print("Skip file %s ..." % fname)
        return

    with open(fname, "w") as outfile:
        skipline = False
        lines = orig.split('\n')
        ext = os.path.splitext(fname)[1][1:]
        if ext == 'sh' and lines[0][:2] == '#!':
            skipline = True
        elif ext == 'xml' and lines[0][:2] == '<?':
            skipline = True

        if skipline:
            outfile.write(lines[0] + "\n")
            outfile.write(header + "\n\n")
            outfile.write("\n".join(lines[1:]))
            outfile.write(header + "\n\n")
            outfile.write(orig)
        else:
            outfile.write(header + "\n\n")
            outfile.write(orig)
    print("Add header to %s" % fname)


def main(args):
    if len(args) != 2:
        print("Usage: python add_asf_header.py <file_list>")

    for l in open(args[1]):
        if l.startswith("-----"):
            continue
        if l.find("File:") != -1:
            l = l.split(":")[-1]
        fname = l.strip()
        suffix = fname.split(".")[-1]
        if suffix in FMT_MAP:
            add_header(fname, FMT_MAP[suffix])
        elif os.path.basename(fname) == 'gradle.properties':
            add_header(fname, FMT_MAP['h'])
        else:
            print("Cannot handle %s ..." % fname)


if __name__ == "__main__":
    main(sys.argv)
