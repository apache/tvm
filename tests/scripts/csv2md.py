import pandas as pd
# if you want to add datetime stamp -mkw
# import datetime
# import time
# date_string = time.strftime("%Y-%m-%d-%H:%M:%S")

df = pd.read_csv("rnotes.csv")

with open("RelNotes_9-0.md", 'w') as md:
  df.to_markdown(buf=md, tablefmt="grid")

