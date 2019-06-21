#!/usr/bin/env python3

def set_attrs():
    fname = "IntelShell.v"
    fname_out = "VTA.v"
    out = ""
    with open(fname, 'rt') as fp:
        for line in fp:
            if "reg " in line and "];" in line:
                print(line[:-1])
                line = line.replace("reg ", '(* ramstyle="M20K" *) reg ')
                print(line[:-1])
                out += line
            elif " * " in line:
                print(line[:-1])
                line = line.replace(" * ", ' * (* multstyle="dsp" *) ')
                print(line[:-1])
                out += line
            else:
                out += line
    with open(fname_out, 'wt') as fp:
        fp.write(out)

if __name__=="__main__":
    set_attrs()
