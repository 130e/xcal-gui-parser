import xcalparser
import pandas as pd
import code

tab = xcalparser.PDSCHMsg()

result = []
n = 1
while n > 0:
    if len(tab.data) > 0:
      result.append(tab.data[-1])
    n -= 1
    tab.parse_next()

df = pd.concat(result, ignore_index=True)

# debug
code.interact(local=locals())

v = df.columns.values
print(len(v))
for e in v:
    print(e)
#df.to_csv("./pdsch-0.csv")
# code.interact(local=locals())
