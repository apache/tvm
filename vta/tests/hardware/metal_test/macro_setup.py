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
