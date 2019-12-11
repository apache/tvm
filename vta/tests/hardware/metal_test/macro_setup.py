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

import re

def header_parse(file, macros):
	with open(file) as infile:
		for line in infile:
			for name, value in re.findall(r'#define\s+(\w+)\s+(.*)', line):
				if len(value) != 0:
					macros.append(name + " = " + value)

def macro_gen(macros):
	with open('_macros_h.py', 'w') as outfile:
		for defs in macros:
			outfile.write("%s\n" % defs)

macros = []
with open('macros.txt') as infile:
    for line in infile:
        list = line.split(" ")
        for str in list:
            if re.search(r'=\d+$', str):
                macros.append(str[2:])

header_parse('../../../include/vta/driver.h', macros)
header_parse('../../../include/vta/hw_spec.h', macros)
header_parse('../../../src/pynq/pynq_driver.h', macros)

macro_gen(macros)
