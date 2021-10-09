import json
import os,sys
import datetime

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


def CreateTableLine(titles,data) ->str:
    url = ""
    if str(data[0]) in translators.keys():
        url=translators[str(data[0])]["URL"]
    else:
        url="https://www.transifex.com/user/profile/"+data[0]
    
    return table_line_tempate.replace("$PERSONALPAGE",url).replace("$NAME",data[titles.index("Username")]).replace("$LANGUAGE",data[titles.index("Lang_code")])\
                              .replace("$TOTAL",data[titles.index("New_total")]).replace("$EDIT_TOTAL",data[titles.index("Edit_total")]).replace("$REVIEW_TOTAL",data[titles.index("Review_total")]).replace("$IMPACT_1",data[titles.index("Match (0-59%)")])\
                                  .replace("$IMPACT_2", str(int(data[titles.index("Match (60-64%)")])+int(data[titles.index("Match (65-69%)")])+int(data[titles.index("Match (70-74%)")])+int(data[titles.index("Match (75-79%)")])+int(data[titles.index("Match (80-84%)")])))\
                                 .replace("$IMPACT_3", str(int(data[titles.index("Match (85-89%)")])+int(data[titles.index("Match (90-94%)")]))).replace("$IMPACT_4", str(int(data[titles.index("Match (95-99%)")])+int(data[titles.index("Match (100%)")])))

with open(os.path.join(base_dir,"translator_data.csv"), "r") as f:
    for line in f:
        if '<' in line:
            print("fail to download the new csv from transifex.com, the contribute page will not update.")
            exit(1)

# 从csdv读取数据
datas = []
with open(os.path.join(base_dir,"translator_data.csv"), "r") as f:
    for line in f:
        line = line.replace("\n", "")                   # 每行去掉换行符
        datas.append(line.split(","))                   # 按分隔符分割

titles=datas.pop(0)                                     # 标题行删除
datas.sort(key=lambda data: int(data[titles.index("New_total")]),reverse=True)

table_lines=""
for i in range(len(datas)):
    table_lines +=CreateTableLine(titles,datas[i])

with open(os.path.join(base_dir,"declaration_zh_CN.html"), "w",encoding='utf-8') as fw:
    fw.write(html_page.replace("$WEBSITE",website).replace("$MIRROR_WEBSITE",mirror_website).replace("$TABLEDATAS",table_lines).replace("$UPDATETIME",str(datetime.date.today() - datetime.timedelta(days=1))))

print("the contribute HTML is built in ",os.path.join(base_dir,"declaration_zh_CN.html"))