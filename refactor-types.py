import re
import os

q = re.compile(r'\{\s*\*ret\s*=\s*\w+')

print(q.match('{ ret = BEE'))
