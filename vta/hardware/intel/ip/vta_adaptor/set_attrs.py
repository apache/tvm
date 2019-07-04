#!/usr/bin/env python3

def set_attrs():
    fname = "VTA.v"
    fname_out = "IntelShell.v"
    out = ""
    with open(fname, 'rt') as fp:
        module = ''
        for idx, line in enumerate(fp):
            if 'module' in line:
                module = line[line.find('module')+7:line.find('(')]
                out += line
            elif "reg " in line and "];" in line:
                # line = line.replace("];", '] /* synthesis ramstyle="M20K" */;')
                print(fname_out+":"+str(idx+1)+": "+module+":"+line[1:line.find(";")+1])
                out += line
            elif " * " in line:
                line = line.replace(" * ", ' * (* multstyle="logic" *) ')
                print(fname_out+":"+str(idx+1)+": "+module+":"+line[1:line.find(";")+1])
                out += line
            elif "rA;" in line:
                # line = line.replace("rA;", 'rA /* synthesis preserve */ /* synthesis noprune */;')
                line = line.replace("rA;", 'rA /* synthesis noprune */;')
                print(fname_out+":"+str(idx+1)+": "+module+":"+line[1:line.find(";")+1])
                out += line
            elif "rB;" in line:
                # line = line.replace("rB;", 'rB /* synthesis preserve */ /* synthesis noprune */;')
                line = line.replace("rB;", 'rB /* synthesis noprune */;')
                print(fname_out+":"+str(idx+1)+": "+module+":"+line[1:line.find(";")+1])
                out += line
            elif "rC;" in line:
                # line = line.replace("rC;", 'rC /* synthesis preserve */ /* synthesis noprune */;')
                line = line.replace("rC;", 'rC /* synthesis noprune */;')
                print(fname_out+":"+str(idx+1)+": "+module+":"+line[1:line.find(";")+1])
                out += line
            else:
                out += line
    with open(fname_out, 'wt') as fp:
        fp.write(out)

if __name__=="__main__":
    set_attrs()
