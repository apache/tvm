import json
import os,sys

base_dir=os.path.dirname(__file__)
sys.path.append(base_dir)
print(base_dir)

website="https://tvmchinese.github.io"          # $WEBSITE
mirror_website="https://tvmchinese.gitee.io"    # MIRROR_WEBSITE

html_page = ""
with open(os.path.join(base_dir,"declaration_zh_CN-template.html"), "r",encoding='utf-8') as f:
    html_page = f.read()

translators={}
with open(os.path.join(base_dir,"translators.json"), "r") as f:
    translators = json.load(f)

table_line_tempate="""
<tr>
    <td><a href="$PERSONALPAGE">$NAME</a></td>
    <td>$LANGUAGE</td>
    <td align="center">$TOTAL</td>
    <td align="center">$EDIT_TOTAL</td>
    <td align="center">$REVIEW_TOTAL</td>
    <td align="center">$IMPACT_1</td>
    <td align="center">$IMPACT_2</td>
    <td align="center">$IMPACT_3</td>
    <td align="center">$IMPACT_4</td>
</tr>
"""

def CreateTableLine(data) ->str:
    url = ""
    if str(data[0]) in translators.keys():
        url=translators[str(data[0])]["URL"]
    else:
        url="https://www.transifex.com/user/profile/"+data[0]
    
    return table_line_tempate.replace("$PERSONALPAGE",url).replace("$NAME",data[0]).replace("$LANGUAGE",data[1])\
                              .replace("$TOTAL",data[12]).replace("$EDIT_TOTAL",data[13]).replace("$REVIEW_TOTAL",data[14]).replace("$IMPACT_1",data[2])\
                                  .replace("$IMPACT_2", str(int(data[3])+int(data[4])+int(data[5])+int(data[6])+int(data[7])))\
                                 .replace("$IMPACT_3", str(int(data[8])+int(data[9]))).replace("$IMPACT_4", str(int(data[10])+int(data[11])))

# 从csdv读取数据
datas = []
with open(os.path.join(base_dir,"translator_data.csv"), "r") as f:
    for line in f:
        line = line.replace("\n", "")  # 每行去掉换行符
        datas.append(line.split(","))  # 按分隔符分割

table_lines=""
for i in range(len(datas)):
    if i==0:
        continue    # 跳过标题
    table_lines +=CreateTableLine(datas[i])

with open(os.path.join(base_dir,"declaration_zh_CN.html"), "w",encoding='utf-8') as fw:
    fw.write(html_page.replace("$WEBSITE",website).replace("$MIRROR_WEBSITE",mirror_website).replace("$TABLEDATAS",table_lines))

print("the contribute HTML is built in ",os.path.join(base_dir,"declaration_zh_CN.html"))