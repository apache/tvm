#!/usr/bin/env python3

import os, sys
import argparse

thisdir = os.path.abspath(os.path.dirname(__file__))
ippath = os.path.join(thisdir, '..', 'ip', 'vta_adaptor')

def set_attrs(fname, fname_out):
    """Set attributes to precompiled verilog code to indicate synthesis preference.

    Parameters
    ----------
    fname : str
        The name of input verilog source code file.

    fname_out : str
        The name of output verilog source code file.
    """
    out = ""
    with open(fname, 'rt') as fp:
        module = ''
        for idx, line in enumerate(fp):
            if 'module' in line:
                module = line[line.find('module')+7:line.find('(')]
                out += line
            elif "reg " in line and "];" in line:
                print(fname_out+":"+str(idx+1)+": "+module+":"+line[1:line.find(";")+1])
                out += line
            elif " * " in line:
                line = line.replace(" * ", ' * (* multstyle="logic" *) ')
                print(fname_out+":"+str(idx+1)+": "+module+":"+line[1:line.find(";")+1])
                out += line
            elif "rA;" in line:
                line = line.replace("rA;", 'rA /* synthesis noprune */;')
                print(fname_out+":"+str(idx+1)+": "+module+":"+line[1:line.find(";")+1])
                out += line
            elif "rB;" in line:
                line = line.replace("rB;", 'rB /* synthesis noprune */;')
                print(fname_out+":"+str(idx+1)+": "+module+":"+line[1:line.find(";")+1])
                out += line
            elif "rC;" in line:
                line = line.replace("rC;", 'rC /* synthesis noprune */;')
                print(fname_out+":"+str(idx+1)+": "+module+":"+line[1:line.find(";")+1])
                out += line
            else:
                out += line
    with open(fname_out, 'wt') as fp:
        fp.write(out)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Set attributes to precompiled ' +
                                     'verilog code to indicate synthesis preference')
    parser.add_argument('-i', '--input', type=str,
                        help='input verilog file to be decorated',
                       default='VTA.DefaultDe10Config.v')
    parser.add_argument('-o', '--output', type=str,
                      help='decorated verilog file',
                      default='IntelShell.v')
    args = parser.parse_args()
    set_attrs(args.input, args.output)
