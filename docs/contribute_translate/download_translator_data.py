import requests
import sys
import datetime
import os
import json

site_url = "https://www.transifex.com/signin/"
file_url = "https://www.transifex.com/TVMChinese/reports/?downloadcsv=1&from=2021-09-01&to="

base_dir=os.path.dirname(__file__)
sys.path.append(base_dir)

def getFullFileURL(): 
    yesterday = datetime.date.today() - datetime.timedelta(days=1) 
    return file_url + str(yesterday) + "&"

transifex_identify = {}
with open(os.path.join(base_dir,"transifex_identify.json"), "r") as f:
    transifex_identify = json.load(f)

headers = {
    'User-Agent': transifex_identify['User-Agent'],
    'Cookie': transifex_identify['Cookie']
}

def getFullFileURL(): 
    yesterday = datetime.date.today() - datetime.timedelta(days=1) 
    return file_url + str(yesterday) + "&expand_languages=1&"

data = requests.get(getFullFileURL(),headers=headers)

with open(os.path.join(base_dir,"translator_data.csv"), "wb") as code:
    code.write(data.content)