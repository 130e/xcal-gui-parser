# Filter in Qualcomm DM Message: 5GNR-MAC Layer-NR5G MAC PDSCH Status
import xcalparser
import pandas as pd
import code

tab = xcalparser.PDSCHMsg()

result = []
n = 4000
while n > 0:
    if len(tab.data) > 0:
      result.append(tab.data[-1])
    n -= 1
    tab.parse_next()

df = pd.concat(result, ignore_index=True)
df.to_csv("./pdsch-{}.csv".format("vz-120-ap-1"))
# df.to_csv("./pdsch-{}.csv".format("att-30-cell-705-620"))

exit()
# debug
code.interact(local=locals())

v = df.columns.values
print(len(v))
for e in v:
    print(e)
#df.to_csv("./pdsch-0.csv")
# code.interact(local=locals())
